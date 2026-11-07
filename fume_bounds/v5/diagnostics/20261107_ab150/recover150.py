"""GPU1 serial INRIA experiment. The tracked v5 trainer and original loss stay intact."""
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

FUME = Path("/inspire/ssd/project/video-generation/public/sunxincheng/FUME")
V5 = FUME / "local_ops/v5_20261106/repo/fume_bounds/v5"
LEGACY = FUME / "local_ops/inria_bound_20261101/frozen_fume"
DATA = LEGACY / "datasets/INRIA-Websearch.mat"
OUT = FUME / "runs/inria_ab150_seed2_20261107"
sys.path.insert(0, str(V5))
import run_evidence_ab as runner
from metrics import retrieval_map
from model import ModelConfig

EPOCHS = 150
ARMS = ("A_fume", "B_bce", "C_category", "D_matching")
DATA_SHA = "aae9ec202e06080b3b3c7c5ecab01ae1aa69483eb7ec93e2b47d65590cf50dbf"


def dump(path, obj):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def append(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def legacy_module(name):
    spec = importlib.util.spec_from_file_location("official_" + name, LEGACY / (name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_model():
    model = legacy_module("model").FUME("cpu", 4096, 1000, 100, 100, 4)
    w = model.W
    del model.W
    model.register_buffer("W", w)
    return model.to("cuda")


def legacy_loss_fn():
    # Official processor imports its sibling evaluate.py by its top-level name.
    sys.modules["evaluate"] = legacy_module("evaluate")
    return legacy_module("processor").calc_loss


def checked_data():
    datasets, receipt = runner.load_data(DATA, 100, 0)
    assert receipt["file_sha256"] == DATA_SHA
    assert [len(datasets[s]) for s in ("train","val","test")] == [9000,1332,4366]
    return datasets, receipt


def order_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def audited_loader(folder, dataset, *, batch_size, shuffle, generator, **kwargs):
    assert shuffle and batch_size == 300 and len(dataset) == 9000
    class LoggedSampler(RandomSampler):
        def __iter__(self):
            ids = list(super().__iter__())
            assert len(set(ids)) == len(dataset)
            path = folder / "order.jsonl"
            epoch = len(path.read_text().splitlines())+1 if path.exists() else 1
            row = dict(epoch=epoch, order=order_hash(ids), first_batch=order_hash(ids[:300]))
            if folder.name != ARMS[0]:
                expected = json.loads((OUT/ARMS[0]/"order.jsonl").read_text().splitlines()[epoch-1])
                assert row == expected, (folder.name, epoch, "sample order mismatch")
            append(path,row)
            return iter(ids)
    return DataLoader(dataset, batch_size=batch_size, sampler=LoggedSampler(dataset,generator=generator),
                      generator=generator, **kwargs)


@torch.no_grad()
def legacy_map(model, ds):
    model.eval()
    image, text, labels = [], [], []
    for xi, xt, y in DataLoader(ds, batch_size=128):
        image.append(model.img_net(xi.cuda()).cpu())
        text.append(model.text_net(xt.cuda()).cpu())
        labels.append(y)
    return retrieval_map(torch.cat(image),torch.cat(text),torch.cat(labels))


def precheck():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35"
    assert torch.cuda.device_count()==1 and "4090" in torch.cuda.get_device_name(0)
    ds, receipt = checked_data()
    original_loss = legacy_loss_fn()
    sample = tuple(t[:64].cuda() for t in ds["train"].tensors)
    runner.seed_all(2)
    original = legacy_model()
    initial = runner.backbone_hash(original)
    with torch.no_grad():
        result = original(*sample[:2])
        loss = original_loss(result["view1_feature"], result["view2_feature"], result["view1_membershipDegree"],
                             result["view2_membershipDegree"], sample[2], sample[2], .05)
        assert torch.isfinite(loss)
        ref = retrieval_map(result["view1_feature"].cpu(),result["view2_feature"].cpu(),sample[2].cpu())
        first_batches = {}
        for arm in ARMS:
            gen = torch.Generator().manual_seed(3)
            first_batches[arm] = order_hash(next(iter(DataLoader(range(9000), batch_size=300,shuffle=True,generator=gen))).tolist())
        assert len(set(first_batches.values()))==1
        for task in ("category", "matching"):
            runner.seed_all(2)
            model = runner.build_model(ModelConfig(),task).cuda()
            assert runner.backbone_hash(model)==initial, task
            zi,zt = model(*sample[:2])
            torch.testing.assert_close(zi["z"],result["view1_feature"])
            torch.testing.assert_close(zt["z"],result["view2_feature"])
            assert retrieval_map(zi["z"].cpu(),zt["z"].cpu(),sample[2].cpu())==ref
            del model
    dump(OUT/"acceptance.json", dict(worker=socket.gethostname(), GPU=os.environ["CUDA_VISIBLE_DEVICES"],
         torch=str(torch.__version__), initial_backbone_and_W_sha256=initial, first_batch_sha256=first_batches,
         same_retrieval_on_64_train_examples=ref, official_loss_on_64=float(loss),
         legacy_source={n:runner.sha256_file(LEGACY/n) for n in ("processor.py","model.py","evaluate.py")},
         data=receipt, seed=2, epochs=EPOCHS,steps_per_epoch=30))
    return initial


def run_original(folder, initial):
    data, _ = checked_data()
    runner.seed_all(2)
    model = legacy_model()
    assert runner.backbone_hash(model)==initial
    loss_fn = legacy_loss_fn()
    opt = torch.optim.Adam(model.parameters(),lr=5e-5,betas=(.5,.999))
    generator = torch.Generator().manual_seed(3)
    loader = audited_loader(folder,data["train"],batch_size=300,shuffle=True,generator=generator)
    dump(folder/"initial_validation.json", legacy_map(model,data["val"]))
    for epoch in range(1,EPOCHS+1):
        model.train()
        start=time.monotonic()
        total=0
        count=0
        for xi,xt,y in loader:
            xi,xt,y = xi.cuda(),xt.cuda(),y.cuda()
            opt.zero_grad(set_to_none=True)
            result = model(xi,xt)
            loss=loss_fn(result["view1_feature"], result["view2_feature"], result["view1_membershipDegree"],
                         result["view2_membershipDegree"], y,y,.05)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite original gradient")
            opt.step()
            total += float(loss.detach())*len(y)
            count += len(y)
        assert count==9000
        row=dict(epoch=epoch,train_rows=count,total=total/count,seconds=time.monotonic()-start)
        append(folder/"history.jsonl",row)
        report=legacy_map(model,data["val"])
        append(folder/"validation.jsonl",dict(epoch=epoch,**report))
        print(folder.name,epoch,row["total"],report["mean_map"],flush=True)
        if epoch%5==0:
            runner.atomic_save(folder/"last.pt",dict(epoch=epoch,model=model.state_dict(),
                optimizer=opt.state_dict(),loader_rng=generator.get_state(),rng=runner.get_rng()))
    final=legacy_map(model,data["test"])
    dump(folder/"test_metrics.json", final)
    return final


def run_v5(arm, folder, initial):
    args=["--data",str(DATA),"--output",str(folder),"--device","cuda","--classes","100","--label-base","0",
          "--epochs",str(EPOCHS),"--seed","2","--classification-source","direction",
          "--evidence-task","matching" if arm=="D_matching" else "category"]
    if arm=="B_bce":
        args += ["--vicinal-weight","0","--noise-weight","0","--kl-weight","0"]
    runner.DataLoader=lambda ds, **kw: audited_loader(folder,ds,**kw)
    runner.main(args)
    assert json.loads((folder/"provenance.json").read_text())["backbone_sha256"]==initial
    del runner.DataLoader
    runner.DataLoader=DataLoader
    gc.collect()
    torch.cuda.empty_cache()
    out = folder/"test"
    runner.main(["--data",str(DATA),"--output",str(out),"--device","cuda","--evaluate","test",
                 "--checkpoint",str(folder/"last.pt")])
    assert json.loads((out/"evaluation_manifest.json").read_text())["epoch"]==EPOCHS
    return json.loads((out/"metrics.json").read_text())["retrieval"]


def main():
    torch.set_num_threads(2)
    # Only one controller may create this run, even across duplicate launches.
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35"
    assert torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name(0)
    previous = json.loads((OUT/"test_map.json").read_text())
    assert list(previous) == [ARMS[0]] and previous[ARMS[0]] == json.loads((OUT/ARMS[0]/"test_metrics.json").read_text())
    assert previous[ARMS[0]]["queries"] == 4366
    assert len((OUT/ARMS[0]/"raw_curves.csv").read_text().splitlines()) == 151
    # Recover only this initial-validation failure, before B has trained.
    assert (OUT/"error.log").is_file() and "trapezoid" in (OUT/"error.log").read_text()
    failed = OUT/ARMS[1]
    assert set(p.name for p in failed.iterdir()) == {"provenance.json"}
    assert not any((OUT/arm).exists() for arm in ARMS[2:])
    archive = OUT/"first_attempt_initial_validation"
    archive.mkdir()
    failed.rename(archive/failed.name)
    (OUT/"error.log").rename(archive/"error.log")
    try:
        from environment_check import check
        environment = check("cuda")
        initial = json.loads((OUT/"acceptance.json").read_text())["initial_backbone_and_W_sha256"]
        assert initial == json.loads((archive/ARMS[1]/"provenance.json").read_text())["backbone_sha256"]
        dump(OUT/"environment.json", environment)
        print("ACCEPTED", initial, flush=True)
        results=previous
        for arm in ARMS[1:]:
            dump(OUT/"status.json",dict(active=arm,completed=list(results),epochs=EPOCHS))
            folder=OUT/arm
            folder.mkdir()
            if arm==ARMS[0]:
                report=run_original(folder,initial)
            else:
                report=run_v5(arm,folder,initial)
            assert report["queries"]==report["candidates"]==4366
            train=[json.loads(s) for s in (folder/"history.jsonl").read_text().splitlines()]
            val=[json.loads(s) for s in (folder/"validation.jsonl").read_text().splitlines()]
            assert [r["epoch"] for r in train]==[r["epoch"] for r in val]==list(range(1,EPOCHS+1))
            with (folder/"raw_curves.csv").open("w",newline="") as handle:
                writer=csv.writer(handle)
                writer.writerow(["epoch","train_loss","train_seconds","val_i2t_map","val_t2i_map","val_mean_map"])
                for t,v in zip(train,val):
                    writer.writerow([t["epoch"],t["total"],t["seconds"],v["i2t_map_all"],v["t2i_map_all"],v["mean_map"]])
            results[arm]=report
            dump(OUT/"test_map.json",results)
            gc.collect()
            torch.cuda.empty_cache()
        dump(OUT/"status.json",dict(active=None,completed=list(results),epochs=EPOCHS))
    except BaseException:
        (OUT/"error.log").write_text(traceback.format_exc())
        raise


if __name__=="__main__":
    main()
