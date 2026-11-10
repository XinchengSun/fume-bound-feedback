"""Inspect the original NUS feature container and the historical, frozen support cache."""
import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import h5py
import numpy as np
import torch
from eligibility import classify, load_manifest


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        (output / name).write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")

    approved = {
        args.dataset: "338d710bd99bdfca28075d87ea0bf5875cdae83cd4b0c5bb931d390ce3c6091b",
        args.cache: "f99ee36b2df15171a14764ae90972a35768f86f16bd8b9becec09c90d7edaabe",
        args.checkpoint: "01eacabdde0e8f9ee88e506bb1c763a4a55e0e3210f6b904bd19fba6f67da290",
    }
    for path, expected in approved.items():
        if digest(path) != expected:
            raise ValueError(f"Input differs from the audited historical source: {path.name}")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    split_keys = ("train", "valid", "test")
    by_modality = {}
    label_counts = {}
    with h5py.File(args.dataset, "r") as f:
        keys = sorted(f.keys())
        attrs = {"file": sorted(f.attrs), **{key:sorted(f[key].attrs) for key in keys}}
        for split in split_keys:
            image_y = f[split+"_imgs_labels"][:].ravel()
            np.testing.assert_array_equal(image_y, f[split+"_texts_labels"][:].ravel())
            counts = Counter(image_y.tolist())
            label_counts[split] = dict(sorted(counts.items()))
        for modality in ("imgs_deep", "texts"):
            hashes = {}
            for split in split_keys:
                x = f[split+"_"+modality][:]
                assert np.isfinite(x).all()
                hashes[split] = Counter(hashlib.sha256(row.tobytes()).digest() for row in x)
            by_modality[modality] = {
                "within_split_repeated_rows_beyond_first":{s:sum(c.values())-len(c) for s,c in hashes.items()},
                "between_split_shared_unique_vectors":{a+"__"+b:len(hashes[a].keys() & hashes[b].keys())
                                                       for a,b in combinations(split_keys,2)}}
    manifest_path = Path(__file__).with_name("historical_manifest.json")
    records, references = load_manifest(manifest_path)
    eligibility, statuses = classify(records, references)
    assert len(statuses) == sum(sum(v.values()) for v in label_counts.values())
    used = [set(item["documented_uses"]) for item in statuses]
    write("entity_overlap_report.json",dict(fields=keys,attributes=attrs,label_counts=label_counts,
          exact_feature_hash_audit=by_modality,alias_verification="unknown; no canonical source IDs",
          row_role_intersections={"historical_head_fit_and_encoder_fit":sum({"head_fit","encoder_fit"} <= u for u in used),
                                  "test_queries_and_gallery":sum({"test_query","gallery"} <= u for u in used)},
          eligibility=eligibility, eligible_cert_rows=eligibility["candidate_rows"]["eligible"],
          historical_manifest_sha256=digest(manifest_path)))
    # The archived feedback study sampled from permuted official loader rows, not HDF5 row order.
    order = np.random.default_rng(41531).permutation(8000)[5600:7648].reshape(2,-1)
    assert np.unique(order).size == 2048
    i,t=order
    im,tx,y=cache["train"]
    def encode(x,w):
        return torch.nn.functional.normalize(torch.nn.functional.linear(x,w),dim=1)
    product=(encode(im[i],cache["wI"])*encode(tx[t],cache["wT"])).double().numpy()*10
    design=np.column_stack([np.ones(len(i)),product])
    singular=np.linalg.svd(design,compute_uv=False)
    np.savetxt(output/"historical_fit_indices.csv",order.T,fmt="%d",delimiter=",",header="image_loader_row,text_loader_row",comments="")
    write("pair_sampling_manifest.json",dict(historical_fit_pairs=len(i),historical_fit_entity_rows=2048,
          historical_fit_positive=int((y[i]==y[t]).sum()),index_space="official train loader seed 2",
          seed=41531,selection="default_rng(seed).permutation(8000)[5600:7648].reshape(2,1024)",
          indices_sha256=digest(output/"historical_fit_indices.csv"),new_fit_pairs=0,new_fit_entities=0,
          cert_pairs=0,cert_entities=0,cert_entity_reuse=0,cert_sampling_seed=None,cert_positive_rate=None))
    packages={}
    for name in ("torch","numpy","scipy","h5py","mpmath","cvxpy","clarabel","python-flint","gmpy2"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    write("features_and_runtime.json",dict(shape=list(design.shape),rank=int(np.linalg.matrix_rank(design)),
          singular_values=singular.tolist(),rank_scope="historical fit only",cert_rank=None,packages=packages,
          frozen_output_dtype="float32",design_dtype="float64",design_sha256=hashlib.sha256(design.tobytes()).hexdigest(),
          dataset_sha256=digest(args.dataset),checkpoint_sha256=digest(args.checkpoint),cache_sha256=digest(args.cache)))

if __name__=="__main__":
    main()

