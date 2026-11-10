"""Classify candidate records using registered uses and linked source blocks."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

USED = {"encoder_fit", "epoch_selection", "method_selection", "hyperparameter_selection",
        "feature_fit", "head_fit", "feedback", "test_query", "gallery"}
REVIEWS = ("identity_review", "history_review", "population_review", "prospective_isolation",
           "observation_unit_review")


def classify(records, evidence):
    by_id = {}
    owner = {}
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in records:
        key = row["record_id"]
        if not isinstance(key,str) or not key or key in by_id:
            raise ValueError("record_id must be unique and nonempty")
        by_id[key] = row
        parent[key] = key
        # canonical and alias identifiers share one namespace; block IDs have their own.
        for field in ("aliases","uses"):
            if not isinstance(row.get(field,[]),list):
                raise ValueError(f"{field} must be a list")
        ids = [row.get("canonical_id"), *row.get("aliases",[])]
        tokens = [("entity",v) for v in ids if v]
        if row.get("source_block"):
            tokens.append(("block",row["source_block"]))
        for token in tokens:
            other = owner.setdefault(token,key)
            parent[find(other)] = find(key)
    groups = defaultdict(list)
    for k,row in by_id.items():
        groups[find(k)].append(row)
    counts = Counter()
    reasons = Counter()
    eligible_entities = set()
    details = []
    for members in groups.values():
        known, missing = set(), set()
        for r in members:
            if not r.get("canonical_id") or not r.get("source_block"):
                missing.add("identity_or_source_missing")
            if r.get("modality") not in {"image","text","image_text"}:
                missing.add("modality_missing")
            for f in REVIEWS:
                if r.get(f) not in evidence:
                    missing.add(f)
            if "uses" not in r:
                missing.add("uses_missing")
            for use in r.get("uses",[]):
                if use["role"] in USED and use.get("ref") in evidence:
                    known.add(use["role"])
                else:
                    missing.add("unverified_use")
        for r in members:
            if type(r.get("candidate")) is not bool:
                raise ValueError("candidate must be boolean")
            if not r["candidate"]:
                continue
            status = "ineligible" if known else "unknown" if missing else "eligible"
            counts[status] += 1
            if status=="eligible":
                eligible_entities.add(find(r["record_id"]))
            reasons.update(known|missing)
            details.append({"record_id":r["record_id"],"status":status,
                            "documented_uses":sorted(known),"missing_information":sorted(missing)})
    return {"candidate_rows":{k:counts[k] for k in ("eligible","ineligible","unknown")},
            "eligible_source_blocks":len(eligible_entities),"reason_counts":dict(reasons)},details


def load_manifest(path):
    raw=json.loads(path.read_text(encoding="utf-8"))
    refs=set()
    for key,ref in raw["evidence"].items():
        if hashlib.sha256((path.parent/ref["file"]).read_bytes()).hexdigest()!=ref["sha256"]:
            raise ValueError(f"evidence hash mismatch: {key}")
        refs.add(key)
    rows=list(raw.get("records",[]))
    for group in raw.get("ranges",[]):
        for i in range(group["count"]):
            rows.append({**group["fields"],"record_id":f"{group['split']}:{i}"})
    for extra in raw.get("indexed_uses",[]):
        lookup={r["record_id"]:r for r in rows}
        with (path.parent/raw["evidence"][extra["index_ref"]]["file"]).open() as f:
            for pair in csv.DictReader(f):
                for column in extra["columns"]:
                    row=lookup[f"{extra['split']}:{int(pair[column])}"]
                    row["uses"]=[*row["uses"],extra["use"]]
    return rows,refs


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--details",type=Path)
    args=parser.parse_args()
    counts,details=classify(*load_manifest(args.manifest))
    if args.details:
        args.details.write_text("\n".join(json.dumps(d) for d in details)+"\n")
    print(json.dumps(counts,indent=2))

