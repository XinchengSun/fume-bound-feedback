"""Run paired continuous FUME feedback training. All output directories must be new."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

if __package__:
    from .engine import ARMS, Experiment, make_batches
else:
    from engine import ARMS, Experiment, make_batches

HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "certification/replay"
SRC = REPLAY / "sources"
RECORD = REPLAY.parent / "records/20261110"
LEGACY_EXTRACTION_SHA256 = "b40924da7a6d75030bfcc5092e1d590c1a51f3d5b49dcaa96e3899124aca766a"
LEGACY_FEATURES_SHA256 = "f99ee36b2df15171a14764ae90972a35768f86f16bd8b9becec09c90d7edaabe"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_csv(path, row):
    fresh = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def source_check():
    for item in json.loads((REPLAY / "source_map.json").read_text()):
        if digest(SRC / item["published_file"]) != item["published_sha256"]:
            raise ValueError("Archived source changed: " + item["published_file"])


def prepare(checkpoint, dataset, out):
    """Extract the archived architecture/loader features, without evaluating test metrics."""
    source_check()
    cfg = json.loads((RECORD / "features_and_runtime.json").read_text())
    checkpoint, dataset = checkpoint.resolve(), dataset.resolve()
    for path, key in ((checkpoint, "checkpoint_sha256"), (dataset, "dataset_sha256")):
        if not path.is_file():
            raise FileNotFoundError(f"Required real input missing: {path}")
        if digest(path) != cfg[key]:
            raise ValueError(f"Input SHA-256 mismatch: {path.name}")
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "extraction"
    (stage / "datasets").mkdir(parents=True)
    target = stage / "datasets/nus_wide-10k_deep_doc2vec-corr-ae.h5py"
    try:
        os.link(dataset, target)
    except OSError:
        shutil.copyfile(dataset, target)
    from model import FUME
    from options import get_dataloader
    from main import set_seed
    torch.set_num_threads(3)
    set_seed(2)
    previous_cwd = Path.cwd()
    try:
        os.chdir(stage)
        with (out / "extraction.log").open("w") as log, contextlib.redirect_stdout(log):
            _, loaders = get_dataloader("nus_deep")
    finally:
        os.chdir(previous_cwd)
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck["epoch"] != 6:
        raise ValueError("Expected the registered epoch-6 checkpoint")
    model = FUME("cpu", 4096, 300, 10, 10, 4).eval()
    model.load_state_dict(ck["state"])
    model.W.copy_(ck["W"])
    pack = dict(W=ck["W"], wI=model.img_net.layers[-1].weight.detach(),
                wT=model.text_net.layers[-1].weight.detach())
    with torch.no_grad():
        for split, loader in loaders.items():
            ds = loader.dataset
            stored = torch.as_tensor(ds.labels)
            if not stored.sum(1).eq(1).all():
                raise ValueError("Expected the archived single-category labels")
            pen = []
            for net, data in zip((model.img_net, model.text_net), (ds.I, ds.T)):
                rows = []
                for x in torch.as_tensor(data).float().split(128):
                    for layer in net.layers[:-1]:
                        x = layer(x).relu()
                    rows.append(x)
                pen.append(torch.cat(rows))
            pack[split] = (*pen, stored.argmax(1))
    feature = out / "features.pt"
    torch.save(pack, feature)
    provenance = dict(source_kind="real", features_sha256=digest(feature), epoch=6,
                      checkpoint_sha256=cfg["checkpoint_sha256"], dataset_sha256=cfg["dataset_sha256"],
                      extraction_script_sha256=digest(HERE / "run.py"),
                      archived_extraction_reference_sha256=digest(SRC / "extract.py"),
                      extraction_change="Same loader/architecture/seed/forward; test metrics deferred until training ends",
                      independent_certification=False)
    write_json(out / "feature_manifest.json", provenance)
    return feature


def extraction_source_allowed(meta):
    """Accept current extraction or the exact published historical source/cache pair."""
    current = meta.get("extraction_script_sha256") == digest(HERE / "run.py")
    historical = (meta.get("extraction_script_sha256") == LEGACY_EXTRACTION_SHA256
                  and meta.get("features_sha256") == LEGACY_FEATURES_SHA256)
    return ((current or historical)
            and meta.get("archived_extraction_reference_sha256") == digest(SRC / "extract.py"))


def load_features(feature):
    meta = json.loads((feature.parent / "feature_manifest.json").read_text())
    if meta["source_kind"] != "real":
        raise ValueError("train/all only accept real inputs; use the explicit smoke command for synthetic data")
    expected = json.loads((RECORD / "features_and_runtime.json").read_text())
    for key in ("checkpoint_sha256", "dataset_sha256"):
        if meta.get(key) != expected[key]:
            raise ValueError("Feature provenance does not match the registered NUS checkpoint/data")
    if meta.get("features_sha256") != digest(feature) or meta.get("epoch") != 6:
        raise ValueError("Feature hash/epoch mismatch")
    if not extraction_source_allowed(meta):
        raise ValueError("Feature extraction source mismatch")
    pack = torch.load(feature, map_location="cpu", weights_only=True)
    validate_pack(pack, real=True)
    return pack, meta


def validate_pack(pack, real):
    for split in ("train", "val", "test"):
        i, t, y = pack[split]
        if i.ndim != 2 or t.ndim != 2 or y.ndim != 1 or len(i) != len(t) or len(i) != len(y):
            raise ValueError("Malformed feature split: " + split)
        if not torch.isfinite(i).all() or not torch.isfinite(t).all():
            raise ValueError("Nonfinite features: " + split)
        if y.dtype != torch.int64 or y.min() < 0 or y.max() >= pack["W"].shape[1]:
            raise ValueError("Invalid category labels")
        if real and len(y) != (8000 if split == "train" else 1000):
            raise ValueError("Real NUS protocol requires 8000/1000/1000 rows")
    if real and (pack["wI"].shape != (10, 4096) or pack["wT"].shape != (10, 4096)):
        raise ValueError("Unexpected real NUS last-layer dimensions")


def support_ids(n, synthetic):
    perm = torch.tensor(np.random.default_rng(41531).permutation(n))
    return perm[:80].reshape(2, -1) if synthetic else perm[5600:7648].reshape(2, -1)


def rng_snapshot():
    state = np.random.get_state()
    return dict(torch_cpu=torch.random.get_rng_state(), python=random.getstate(),
                numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]))


def train_run(pack, metadata, args, out):
    started = time.monotonic()
    synthetic = metadata["source_kind"] == "synthetic"
    validate_pack(pack, real=not synthetic)
    out.mkdir(parents=True, exist_ok=True)
    source_check()
    arms = tuple(getattr(args, "arms", ARMS))
    weighting = getattr(args, "feedback_weighting", "balanced")
    previous_run = getattr(args, "previous_run", None)
    if len(arms) < 2 or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("Paired training requires at least two unique registered arms")
    if previous_run and synthetic:
        raise ValueError("--previous-run is restricted to verified real inputs")
    support = support_ids(len(pack["train"][2]), synthetic)
    experiment = Experiment(pack, support, feedback_weighting=weighting)
    initial = [pack[k].detach().clone() for k in ("wI", "wT")]
    schedules = {seed: make_batches(len(pack["train"][2]), args.batch_size, args.epochs, seed)
                 for seed in args.seeds}
    config = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  feedback_weight=args.feedback_weight, adam_betas=[.5, .999], alpha=.5,
                  contrastive_temperature=1.0, arms=list(arms), threads=args.threads,
                  feedback_weighting=weighting, training_positive_prior=experiment.training_positive_prior,
                  feedback_positive_weight=experiment.positive_weight,
                  gradient_diagnostic_frequency="first batch of each epoch; extra autograd calls, no gradient modification",
                  support_seed=41531, support_pairs=40 if synthetic else 1024,
                  training_rows=len(pack["train"][2]), support_rows_also_train=True,
                  posterior="Laplace logistic; independent N(0,4) prior; 24-point GH",
                  primary_endpoint="fixed final epoch", secondary_endpoint="validation-best mAP, earliest tie",
                  checkpoint_initialization_count=1, order_seeds_are_model_seeds=False,
                  units="probabilities, mAP in [0,1]; entropy in nats; mAP differences multiply by 100 for pp")
    manifest = dict(source_kind=metadata["source_kind"], synthetic=synthetic, order_seeds=args.seeds,
                    config=config, features_sha256=metadata["features_sha256"], input_provenance=metadata,
                    evaluation={split+"_queries": len(pack[split][2]) for split in ("train", "val", "test")},
                    code_sha256={p.name: digest(p) for p in (HERE / "engine.py", HERE / "run.py")},
                    versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy")},
                    statistical_scope="empirical feedback pilot; no independent probability certification")
    if previous_run:
        if __package__:
            from .compare_weighting import check_previous
        else:
            from compare_weighting import check_previous
        initial_hashes = [hashlib.sha256(w.numpy().tobytes()).hexdigest() for w in initial]
        manifest["previous_run_control"] = check_previous(previous_run, manifest, support, schedules, initial_hashes)
        manifest["code_sha256"]["compare_weighting.py"] = digest(HERE / "compare_weighting.py")
    manifest["protocol_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "status.json", dict(status="running", source_kind=metadata["source_kind"],
                                         training_started=False, test_evaluated=False))
    torch.set_num_threads(args.threads)
    torch.manual_seed(41531)
    np.random.seed(41531)
    random.seed(41531)
    write_json(out / "support.json", support.tolist())
    base_val = experiment.evaluate(*initial, "val")
    if not synthetic:
        expected = json.loads((SRC / "official_result.json").read_text())["val"]
        if max(abs(base_val[k] - expected[k]) for k in ("i2t", "t2i")) > 1e-5:
            raise ValueError("Initial validation mAP differs from the archived checkpoint canary")
    write_json(out / "initial_validation.json", base_val)
    selected = []
    starts = []
    for seed in args.seeds:
        schedule = schedules[seed]
        write_json(out / f"batches_{seed}.json", [[b.tolist() for b in ep] for ep in schedule])
        for arm in arms:
            torch.manual_seed(41531)
            np.random.seed(41531)
            random.seed(41531)
            weights = [w.clone().requires_grad_() for w in initial]
            optimizer = torch.optim.Adam(weights, lr=args.lr, betas=(.5, .999))
            work = out / f"seed_{seed}" / arm
            work.mkdir(parents=True)
            best, best_epoch, step = base_val["map"], 0, 0

            def save(path, epoch):
                torch.save(dict(weights=[w.detach().clone() for w in weights], W=pack["W"],
                                optimizer=optimizer.state_dict(), rng=rng_snapshot(), epoch=epoch,
                                global_step=step, arm=arm, order_seed=seed,
                                protocol_sha256=manifest["protocol_sha256"],
                                features_sha256=metadata["features_sha256"]), path)

            starts.append(dict(seed=seed, arm=arm, optimizer_state_entries=len(optimizer.state),
                               weights_sha256=[hashlib.sha256(w.detach().numpy().tobytes()).hexdigest() for w in weights]))
            save(work / "val_best.pt", 0)
            append_csv(out / "validation.csv", dict(seed=seed, arm=arm, epoch=0, **base_val))
            for epoch, batches in enumerate(schedule, 1):
                for batch_no, ids in enumerate(batches):
                    optimizer.zero_grad(set_to_none=True)
                    loss, stats = experiment.loss(*weights, ids, arm, args.feedback_weight,
                                                  diagnostics=(batch_no == 0))
                    loss.backward()
                    if any(w.grad is None or not torch.isfinite(w.grad).all() for w in weights):
                        raise FloatingPointError("Missing/nonfinite gradient; no update applied")
                    grad_norm = [w.grad.norm().item() for w in weights]
                    optimizer.step()
                    step += 1
                    if step == 1:
                        write_json(out / "status.json", dict(status="training", source_kind=metadata["source_kind"],
                                   training_started=True, test_evaluated=False, seed=seed, arm=arm))
                    append_csv(out / "training.csv", dict(seed=seed, arm=arm, epoch=epoch, step=step,
                               batch=batch_no, rows=len(ids), **stats,
                               image_grad_norm=grad_norm[0], text_grad_norm=grad_norm[1]))
                value = experiment.evaluate(*weights, "val")
                append_csv(out / "validation.csv", dict(seed=seed, arm=arm, epoch=epoch, **value))
                if value["map"] > best:
                    best, best_epoch = value["map"], epoch
                    save(work / "val_best.pt", epoch)
                save(work / "latest.pt", epoch)
                print(json.dumps(dict(seed=seed, arm=arm, epoch=epoch, step=step,
                                      validation_map=value["map"], validation_brier=value["brier"],
                                      source_kind=metadata["source_kind"])), flush=True)
            save(work / "final.pt", args.epochs)
            selected.append(dict(seed=seed, arm=arm, final_epoch=args.epochs, best_epoch=best_epoch))
    write_json(out / "branch_starts.json", starts)
    write_json(out / "selections.json", selected)
    # No test evaluation occurs before every arm/order has finished and all selections are fixed.
    write_json(out / "status.json", dict(status="evaluating_fixed_selections",
                                         source_kind=metadata["source_kind"], training_started=True,
                                         test_evaluation_started=True, test_evaluated=False))
    initial_test = experiment.evaluate(*initial, "test")
    for item in selected:
        seed, arm = item["seed"], item["arm"]
        append_csv(out / "final.csv", dict(seed=seed, arm=arm, selection="epoch0", epoch=0,
                                           split="test", **initial_test))
        for selection, file, epoch in (("final", "final.pt", args.epochs),
                                       ("val_best", "val_best.pt", item["best_epoch"])):
            state = torch.load(out / f"seed_{seed}" / arm / file, map_location="cpu", weights_only=True)
            value = experiment.evaluate(*state["weights"], "test")
            append_csv(out / "final.csv", dict(seed=seed, arm=arm, selection=selection, epoch=epoch,
                                               split="test", **value))
    if __package__:
        from .summarize import summarize_run
    else:
        from summarize import summarize_run
    summarize_run(out, out / "summary")
    write_json(out / "status.json", dict(status="completed", source_kind=metadata["source_kind"],
               real_training_completed=not synthetic, synthetic_training_completed=synthetic,
               training_started=True, test_evaluated=True, training_steps_per_arm=sum(map(len, schedule)),
               seconds=time.monotonic()-started, protocol_sha256=manifest["protocol_sha256"]))
    if previous_run:
        if __package__:
            from .compare_weighting import compare_runs
        else:
            from compare_weighting import compare_runs
        compare_runs(previous_run, out, out / "weighting_comparison")


def synthetic_pack():
    """Explicit integration fixture, never used as a fallback for missing real inputs."""
    gen = torch.Generator().manual_seed(811)
    centers_i, centers_t = torch.randn(4, 16, generator=gen), torch.randn(4, 12, generator=gen)
    pack = {"wI": .1 * torch.randn(10, 16, generator=gen),
            "wT": .1 * torch.randn(10, 12, generator=gen),
            "W": torch.linalg.qr(torch.randn(10, 10, generator=gen))[0][:, :4]}
    for split, n in (("train", 96), ("val", 32), ("test", 32)):
        y = torch.arange(n) % 4
        pack[split] = (centers_i[y] + .8*torch.randn(n, 16, generator=gen),
                       centers_t[y] + .8*torch.randn(n, 12, generator=gen), y.long())
    return pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "train", "all", "smoke"):
        sub = modes.add_parser(name)
        sub.add_argument("--out", required=True, type=Path)
        if name in ("prepare", "all"):
            sub.add_argument("--checkpoint", required=True, type=Path)
            sub.add_argument("--dataset", required=True, type=Path)
        if name == "train":
            sub.add_argument("--features", required=True, type=Path)
        if name in ("train", "all"):
            sub.add_argument("--previous-run", type=Path,
                             help="Completed balanced run to validate before training and compare afterwards")
        if name != "prepare":
            sub.add_argument("--epochs", type=int, default=2 if name == "smoke" else 5)
            sub.add_argument("--batch-size", type=int, default=16 if name == "smoke" else 300)
            sub.add_argument("--seeds", type=int, nargs="+", default=[110104, 110105] if name == "smoke" else [110104, 110105, 110106])
            sub.add_argument("--lr", type=float, default=5e-5)
            sub.add_argument("--feedback-weight", type=float, default=1.0)
            sub.add_argument("--threads", type=int, default=3)
            sub.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
            sub.add_argument("--feedback-weighting", choices=("balanced", "train_prior"), default="balanced")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Output directory already exists; refusing to overwrite it")
    if args.mode != "prepare" and (args.epochs < 1 or args.batch_size < 2 or args.lr <= 0
                                   or args.feedback_weight < 0 or args.threads < 1
                                   or len(set(args.seeds)) != len(args.seeds)
                                   or len(set(args.arms)) != len(args.arms) or len(args.arms) < 2):
        parser.error("Invalid training configuration")
    args.out.mkdir(parents=True)
    try:
        if args.mode == "prepare":
            feature = prepare(args.checkpoint, args.dataset, args.out)
            write_json(args.out / "status.json", dict(status="prepared", real_training_completed=False,
                                                      features=str(feature)))
        elif args.mode == "smoke":
            pack = synthetic_pack()
            feature = args.out / "synthetic_features.pt"
            torch.save(pack, feature)
            train_run(pack, dict(source_kind="synthetic", features_sha256=digest(feature)), args, args.out)
        else:
            feature = (prepare(args.checkpoint, args.dataset, args.out / "prepared")
                       if args.mode == "all" else args.features.resolve())
            pack, metadata = load_features(feature)
            train_run(pack, metadata, args, args.out)
    except Exception as error:
        status_path = args.out / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        began = previous.get("training_started", False)
        write_json(status_path, dict(status="not_run" if isinstance(error, FileNotFoundError) and not began else "failed",
                   source_kind="synthetic" if args.mode == "smoke" else "real", real_training_completed=False,
                   training_started=began, test_evaluated=previous.get("test_evaluated", False),
                   test_evaluation_started=previous.get("test_evaluation_started", False),
                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    main()
