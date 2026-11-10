"""Summarize preselected endpoints without using test scores to select an epoch."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

ARMS = ("original", "predictive_mean", "stop_sd", "full")
SELECTIONS = ("final", "val_best")
CONTRASTS = (
    ("full", "predictive_mean"), ("stop_sd", "predictive_mean"),
    ("full", "stop_sd"), ("predictive_mean", "original"),
    ("stop_sd", "original"), ("full", "original"),
)
REQUIRED_METRICS = ("i2t", "t2i", "map", "brier", "nll", "ece")
OPTIONAL_METRICS = ("pos_L", "neg_L", "AU", "EU", "event_loss", "lower", "upper", "width")
HIGHER = {"i2t", "t2i", "map"}
LOWER = {"brier", "nll", "ece", "event_loss"}


def integer(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if str(result) != str(value):
        raise ValueError(f"Invalid integer {field}: {value!r}")
    return result


def load_results(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source = manifest.get("source_kind")
    if source not in {"real", "synthetic"}:
        raise ValueError("Manifest must explicitly declare source_kind=real or synthetic")
    if "synthetic" in manifest and manifest["synthetic"] is not (source == "synthetic"):
        raise ValueError("Contradictory synthetic and source_kind fields")
    seeds = [integer(s, "order seed") for s in manifest.get("order_seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Manifest order_seeds must be nonempty and unique")
    with (run_dir / "final.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {"seed", "arm", "selection", "epoch", "split", *REQUIRED_METRICS}
        if not required <= fields:
            raise ValueError(f"Missing final.csv columns: {sorted(required - fields)}")
        metrics = REQUIRED_METRICS + tuple(m for m in OPTIONAL_METRICS if m in fields)
        rows = list(reader)
    if not rows:
        raise ValueError("final.csv is empty")
    indexed = {}
    expected_epochs = manifest.get("config", {}).get("epochs")
    if expected_epochs is not None:
        expected_epochs = integer(expected_epochs, "epochs")
        if expected_epochs < 1:
            raise ValueError("epochs must be positive")
    for row in rows:
        row["seed"] = integer(row["seed"], "seed")
        row["epoch"] = integer(row["epoch"], "epoch")
        if row["seed"] not in seeds:
            raise ValueError("CSV seed is not in manifest order_seeds")
        if row["arm"] not in ARMS or row["selection"] not in (*SELECTIONS, "epoch0"):
            raise ValueError("Unknown arm or endpoint selection")
        if row["split"] != "test":
            raise ValueError("final.csv must contain test endpoints only; no epoch selection is performed here")
        if row["epoch"] < 0 or (row["selection"] == "epoch0" and row["epoch"] != 0):
            raise ValueError("Invalid endpoint epoch")
        if expected_epochs is not None:
            if row["epoch"] > expected_epochs:
                raise ValueError("Endpoint epoch exceeds fixed training budget")
            if row["selection"] == "final" and row["epoch"] != expected_epochs:
                raise ValueError("Final endpoint must equal the prespecified training epoch")
        for metric in metrics:
            try:
                row[metric] = float(row[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid metric {metric}") from exc
            if not math.isfinite(row[metric]):
                raise ValueError(f"Non-finite metric {metric}")
            if metric in HIGHER | {"brier", "ece", "pos_L", "neg_L", "lower", "upper", "width"}:
                if not 0 <= row[metric] <= 1:
                    raise ValueError(f"{metric} must be stored in [0, 1], not percent")
            if metric in {"nll", "event_loss"} and row[metric] < 0:
                raise ValueError(f"Negative {metric}")
        if not math.isclose(row["map"], (row["i2t"] + row["t2i"]) / 2, rel_tol=0, abs_tol=1e-10):
            raise ValueError("map must equal the mean of i2t and t2i")
        for field in ("queries", "pairs"):
            if field in fields:
                row[field] = integer(row[field], field)
                if row[field] < 1:
                    raise ValueError(f"{field} must be positive")
        key = (row["seed"], row["arm"], row["selection"])
        if key in indexed:
            raise ValueError(f"Duplicate endpoint key: {key}")
        indexed[key] = row
    selections = set(row["selection"] for row in rows)
    if not set(SELECTIONS) <= selections:
        raise ValueError("Both final and val_best endpoints are required")
    for selection in selections:
        required_keys = {(s, a, selection) for s in seeds for a in ARMS}
        actual_keys = {key for key in indexed if key[2] == selection}
        if required_keys != actual_keys:
            raise ValueError(f"Unpaired or missing seed/arm endpoints for {selection}: {sorted(required_keys - actual_keys)}")
    if len({r["epoch"] for r in rows if r["selection"] == "final"}) != 1:
        raise ValueError("Final epochs differ across paired arms/seeds")
    for field in ("queries", "pairs"):
        if field in fields and len({r[field] for r in rows}) != 1:
            raise ValueError(f"Evaluation {field} differ across paired endpoints")
    return manifest, seeds, metrics, indexed


def summarize_run(run_dir, out_dir=None):
    manifest, seeds, metrics, indexed = load_results(run_dir)
    per_seed, aggregates = [], []
    for selection in SELECTIONS:
        for left, right in CONTRASTS:
            for metric in metrics:
                factor = 100.0 if metric in HIGHER else 1.0
                direction = "higher" if metric in HIGHER else "lower" if metric in LOWER else "descriptive"
                paired = []
                for seed in seeds:
                    left_value = factor * indexed[seed, left, selection][metric]
                    right_value = factor * indexed[seed, right, selection][metric]
                    delta = factor * (indexed[seed, left, selection][metric] - indexed[seed, right, selection][metric])
                    win = (delta > 0 if direction == "higher" else delta < 0) if direction != "descriptive" else None
                    record = dict(seed=seed, selection=selection, contrast=f"{left}-minus-{right}",
                                  metric=metric, value_left=left_value, value_right=right_value,
                                  difference=delta, value_unit="percent" if factor == 100 else "raw",
                                  difference_unit="percentage_points" if factor == 100 else "raw",
                                  direction=direction, win=win)
                    per_seed.append(record)
                    paired.append(record)
                differences = [r["difference"] for r in paired]
                zeros = sum(d == 0 for d in differences)
                wins = sum(r["win"] for r in paired) if direction != "descriptive" else None
                aggregates.append(dict(selection=selection, contrast=f"{left}-minus-{right}", metric=metric,
                    n_seeds=len(seeds), left_mean=mean(r["value_left"] for r in paired),
                    right_mean=mean(r["value_right"] for r in paired), mean_difference=mean(differences),
                    min_difference=min(differences), max_difference=max(differences),
                    n_positive=sum(d > 0 for d in differences), n_negative=sum(d < 0 for d in differences),
                    n_zero=zeros, n_wins=wins, n_losses=len(seeds)-wins-zeros if wins is not None else None,
                    value_unit=paired[0]["value_unit"], difference_unit=paired[0]["difference_unit"],
                    direction=direction))
    rows = list(indexed.values())
    synthetic = manifest["source_kind"] == "synthetic"
    summary = dict(
        title="SYNTHETIC INTEGRATION TEST — NOT NUS PERFORMANCE" if synthetic else "NUS empirical multistep feedback experiment",
        source_kind=manifest["source_kind"], synthetic=synthetic,
        primary_selection="final", secondary_selection="val_best",
        selection_rule="Read already selected endpoint labels; never choose an epoch using test metrics.",
        replication_unit="minibatch order seed; not independent encoder initializations, support libraries, or datasets",
        inference_scope="Descriptive paired differences at one starting checkpoint; no confidence interval or significance claim.",
        uncertainty_scope="Changes in AU, EU, endpoints and width are descriptive; shrinking width is not a coverage certificate.",
        order_seeds=seeds, n_order_seeds=len(seeds),
        protocol_sha256=manifest.get("protocol_sha256", "unknown"),
        features_sha256=manifest.get("features_sha256", "unknown"),
        manifest_test_queries=manifest.get("evaluation", {}).get("test_queries", "unknown"),
        observed_queries=sorted({r["queries"] for r in rows}) if "queries" in rows[0] else "unknown",
        observed_pairs=sorted({r["pairs"] for r in rows}) if "pairs" in rows[0] else "unknown",
        paired_summary=aggregates,
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
        for name, records in (("per_seed", per_seed), ("paired_summary", aggregates)):
            with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Training directory containing final.csv and run_manifest.json")
    parser.add_argument("--out", type=Path, help="New output directory; default: RUN/summary")
    args = parser.parse_args()
    result = summarize_run(args.run, args.out if args.out is not None else args.run / "summary")
    print(json.dumps({k: v for k, v in result.items() if k != "paired_summary"}, indent=2))
    print("Primary endpoint: fixed final epoch; full minus predictive_mean")
    for row in result["paired_summary"]:
        if row["selection"] == "final" and row["contrast"] == "full-minus-predictive_mean" and row["metric"] in REQUIRED_METRICS:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
