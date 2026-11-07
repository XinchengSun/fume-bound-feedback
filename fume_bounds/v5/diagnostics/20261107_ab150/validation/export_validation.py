"""Re-export validation tensors from the four final checkpoints using CPU inference."""
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ARMS = ("A_fume", "B_bce", "C_category", "D_matching")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(fume, out):
    repo = fume / "local_ops/v5_20261106/repo/fume_bounds/v5"
    legacy = fume / "local_ops/inria_bound_20261101/frozen_fume"
    logs = fume / "runs/inria_ab150_seed2_20261107"
    sys.path.insert(0, str(repo))
    from run_evidence_ab import ModelConfig, build_model, load_data
    from metrics import retrieval_map
    spec = importlib.util.spec_from_file_location("frozen_fume", legacy / "model.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.set_num_threads(2)
    data, receipt = load_data(legacy/"datasets/INRIA-Websearch.mat", 100, 0)
    acceptance = json.loads((logs/"acceptance.json").read_text())
    assert receipt["val"] == acceptance["data"]["val"]
    val = data["val"]
    del data
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"split":"val", "samples":len(val), "sample_ids":"zero-based validation row offsets",
                "data_sha256":receipt["file_sha256"], "val_receipt":receipt["val"],
                "device":"cpu", "torch":str(torch.__version__), "numpy":np.__version__, "arms":{}}
    common = {}
    for arm in ARMS:
        loc = logs/arm
        ckfile = loc/"last.pt"
        ckhash = sha256(ckfile)
        ck = torch.load(ckfile, map_location="cpu", mmap=True, weights_only=True)
        assert ck["epoch"] == 150
        captured = {}
        if arm == "A_fume":
            model = official.FUME("cpu",4096,1000,100,100,4)
            w = model.W
            del model.W
            model.register_buffer("W",w)
            model.load_state_dict(ck["model"],strict=True)
            def capture(name):
                def hook(module, inputs, output):
                    captured[name] = output
                return hook
            model.img_net.layers[-1].register_forward_hook(capture("image"))
            model.text_net.layers[-1].register_forward_hook(capture("text"))
        else:
            ev = json.loads((loc/"test/evaluation_manifest.json").read_text())
            assert ev["checkpoint_sha256"] == ckhash and ev["epoch"] == ck["epoch"]
            cfg = ck["run_config"]
            assert cfg["classification_source"]=="direction"
            mc=ModelConfig(receipt["image_dim"], receipt["text_dim"], receipt["classes"],cfg["hidden"],cfg["layers"],cfg["initial_evidence"])
            model = build_model(mc,cfg["evidence_task"],cfg["evidence_mode"],cfg["pair_hidden"])
            assert model.configuration()==ck["model_config"]
            model.load_state_dict(ck["model_state"],strict=True)
        model.eval()
        buckets = {}
        def add(key, tensor):
            buckets.setdefault(key,[]).append(tensor.cpu().numpy())
        with torch.inference_mode():
            for xi, xt, labels in DataLoader(val,batch_size=128,shuffle=False):
                pred=model(xi,xt)
                add("labels",labels)
                for idx,side in enumerate(("image","text")):
                    if arm=="A_fume":
                        prefix="view"+str(idx+1)+"_"
                        h,z=captured[side],pred[prefix+"feature"]
                        m=pred[prefix+"membershipDegree"]
                        add(prefix+"membershipDegree",m)
                        add(prefix+"cred",model.get_test_category_credibility(m))
                        add(prefix+"uncertainty",pred[prefix+"uncertainty"])
                    else:
                        h,z=pred[idx]["h"],pred[idx]["z"]
                        logits=pred[idx]["logits"]
                        torch.testing.assert_close(logits,h@model.W)
                        add("logits_"+side,logits)
                        add("class_probability_"+side,logits.sigmoid())
                        if arm=="C_category":
                            add("beta_posterior_mean_"+side,pred[idx]["probability"])
                    torch.testing.assert_close(z,torch.nn.functional.normalize(h,dim=1))
                    add("h_"+side,h)
                    add("z_"+side,z)
        arrays={k:np.concatenate(v) for k,v in buckets.items()}
        arrays.update(sample_ids=np.arange(len(val),dtype=np.int64),W=model.W.detach().numpy())
        for key in ("labels","sample_ids","W"):
            if arm==ARMS[0]:
                common[key]=arrays[key].copy()
            else:
                assert np.array_equal(arrays[key],common[key]), (arm,key)
        assert np.array_equal(arrays["labels"],val.tensors[2].numpy())
        assert all(np.isfinite(x).all() for x in arrays.values())
        p=out/(arm+".npz")
        np.savez_compressed(p,**arrays)
        with np.load(p,allow_pickle=False) as z:
            assert all(np.array_equal(z[k],v) for k,v in arrays.items())
        rep=retrieval_map(*(torch.from_numpy(arrays[k]) for k in ("z_image","z_text","labels")))
        logged=json.loads((loc/"validation.jsonl").read_text().splitlines()[-1])
        assert logged["epoch"]==150 and rep["queries"]==len(val)==1332
        error=max(abs(rep[k]-logged[k]) for k in ("i2t_map_all","t2i_map_all","mean_map"))
        assert error<0.0001,(arm,error)
        manifest["arms"][arm]=dict(epoch=ck["epoch"],checkpoint_sha256=ckhash,npz_sha256=sha256(p),
                size_bytes=p.stat().st_size,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                cpu_map=rep,logged_val_map={k:logged[k] for k in ("i2t_map_all","t2i_map_all","mean_map")},
                max_abs_map_difference=error,optimizer_lr_in_checkpoint=[g["lr"] for g in ck["optimizer"]["param_groups"]])
        print(arm,ck["epoch"],len(val),p.stat().st_size,error,flush=True)
        del ck,model,buckets,arrays
        gc.collect()
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fume-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.fume_root,args.out)
