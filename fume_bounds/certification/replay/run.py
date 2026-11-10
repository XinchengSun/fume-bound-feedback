"""Portable replay of archived NUS scripts; outputs always go to a new directory."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SRC=HERE/"sources"
RECORD=HERE.parent/"records"/"20261110"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def checked(path,sha):
    if digest(path)!=sha:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def compare(got,ref,keys):
    import pandas as pd
    import numpy as np
    left,right=[pd.read_csv(p).set_index(keys).sort_index() for p in (got,ref)]
    if not left.index.is_unique or not left.index.equals(right.index) or not left.columns.equals(right.columns):
        raise ValueError(f"CSV schema/key mismatch: {ref.name}")
    np.testing.assert_allclose(left,right,rtol=0,atol=1e-12,equal_nan=True)
    return dict(rows=len(left),max_absolute_difference=float(np.nanmax(abs(left-right))))


def summary(work,raw):
    shutil.copyfile(raw,work/"raw.csv")
    shutil.copyfile(SRC/"summarize.py",work/"summarize.py")
    subprocess.run([sys.executable,str(work/"summarize.py")],check=True)
    return {n:compare(work/(n+".csv"),RECORD/("historical_"+n+".csv"),keys)
            for n,keys in [("averages",["split","group","kind"]),("paired",["split","group","contrast","metric"])]}


def stage_input(src,dst):
    try:
        os.link(src,dst)
    except OSError:
        shutil.copyfile(src,dst)


def replay(args,work):
    import importlib.metadata as metadata
    import torch
    from platform import python_version
    cfg=json.loads((RECORD/"features_and_runtime.json").read_text())
    checked(args.checkpoint,cfg["checkpoint_sha256"])
    checked(args.dataset,cfg["dataset_sha256"])
    official=work/"2025-CVPR-FUME"
    shutil.copytree(SRC/"upstream",official)
    old=work/"runs"/"nus_official_seed2_replay_20261108"
    (old/"datasets").mkdir(parents=True)
    stage_input(args.checkpoint,old/"best.pt")
    stage_input(args.dataset,old/"datasets"/"nus_wide-10k_deep_doc2vec-corr-ae.h5py")
    shutil.copyfile(SRC/"official_result.json",old/"result.json")
    base=old.parent/"real_feedback_20261110"
    base.mkdir()
    for name in ("experiment.py","extract.py","result.json"):
        shutil.copyfile(SRC/name,base/name)
    env=dict(os.environ,FUME_REPLAY_ROOT=str(work),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="3",
             CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
    def call(path):
        subprocess.run([sys.executable,"-Bu",str(path)],env=env,check=True)
    call(base/"extract.py")
    # Inspect the actual frozen architecture. Its fixed W is a plain tensor.
    import sys as _sys
    _sys.path.insert(0,str(official))
    from model import FUME
    with torch.device("meta"):
        skeleton=FUME("meta",4096,300,10,10,4)
    buffers=list(dict(skeleton.named_buffers()))
    if buffers:
        raise RuntimeError(f"Unaccounted buffers: {buffers}")
    run=old.parent/"feedback_mechanism_20261110"
    run.mkdir()
    shutil.copyfile(SRC/"mechanism.py",run/"mechanism.py")
    shutil.copyfile(HERE/"observe.py",run/"observe.py")
    call(run/"observe.py")
    if json.loads((run/"batches.json").read_text())!=json.loads((RECORD/"mechanism/batches.json").read_text()):
        raise ValueError("Generated batch IDs differ")
    result={"source_buffers":buffers,"observed_step_starts":len(json.loads((run/"starts.json").read_text())),
            "python":python_version(),"versions":{p:metadata.version(p) for p in ["numpy","scipy","pandas","torch","h5py"]},
            "batch_indices_match":True,"inputs":{"checkpoint":cfg["checkpoint_sha256"],"dataset":cfg["dataset_sha256"]}}
    result["raw"]=compare(run/"raw.csv",RECORD/"mechanism/raw.csv",["batch","group","kind","split"])
    result["gradients"]=compare(run/"gradients.csv",RECORD/"mechanism/gradients.csv",["batch","tower"])
    result.update(summary(work,run/"raw.csv"))
    return result


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="mode",required=True)
    for mode in ("summary","replay"):
        p=sub.add_parser(mode)
        p.add_argument("--out",required=True,type=Path)
        if mode=="replay":
            p.add_argument("--checkpoint",required=True,type=Path)
            p.add_argument("--dataset",required=True,type=Path)
    args=parser.parse_args()
    for entry in json.loads((HERE/"source_map.json").read_text()):
        checked(SRC/entry["published_file"],entry["published_sha256"])
    work=args.out.resolve()
    work.mkdir(parents=True,exist_ok=False)
    result=summary(work,RECORD/"mechanism/raw.csv") if args.mode=="summary" else replay(args,work)
    (work/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()

