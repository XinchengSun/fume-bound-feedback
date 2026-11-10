"""Check a prior run before training and compare the fixed weighting experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

if __package__:
    from .summarize import ARMS, HIGHER, LOWER, REQUIRED_METRICS, SELECTIONS, load_results
else:
    from summarize import ARMS, HIGHER, LOWER, REQUIRED_METRICS, SELECTIONS, load_results

FIXED_CONFIG = (
    "epochs", "batch_size", "lr", "feedback_weight", "adam_betas", "alpha",
    "contrastive_temperature", "threads", "support_seed", "support_pairs",
    "training_rows", "support_rows_also_train", "posterior", "primary_endpoint",
    "secondary_endpoint", "checkpoint_initialization_count", "order_seeds_are_model_seeds", "units",
)
NEW_ARMS = {"predictive_mean", "full"}
PREVIOUS_REQUIRED_ARMS = {"original", "predictive_mean", "full"}
RAW_DIAGNOSTICS = ("TU", "mean_probability", "logit_sd")
REGISTERED_PREVIOUS_PROTOCOL_SHA256 = "397f9ee04081e00ca08631813aba14107695fd001443b7fc6835ed0cf36d0e7f"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def protocol_sha256(manifest):
    payload = {key: value for key, value in manifest.items() if key != "protocol_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_protocol(manifest):
    actual = protocol_sha256(manifest)
    if actual != manifest.get("protocol_sha256"):
        raise ValueError("Run manifest fields do not match its protocol hash")
    return actual


def arms(manifest):
    result = manifest.get("config", {}).get("arms", list(ARMS))
    if not isinstance(result, list) or not result or len(result) != len(set(result)) or not set(result) <= set(ARMS):
        raise ValueError("Invalid declared arms")
    return result


def completed(path, manifest):
    status = read_json(Path(path) / "status.json")
    kind = manifest.get("source_kind")
    if (kind not in {"real", "synthetic"} or status.get("source_kind") != kind
            or status.get("status") != "completed" or status.get("test_evaluated") is not True
            or status.get(kind + "_training_completed") is not True):
        raise ValueError("Comparison requires a completed run with test endpoints and explicit source kind")
    if status.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise ValueError("Completed status and run manifest protocol hashes differ")
    return status


def compare_manifests(previous, new):
    if previous.get("source_kind") != new.get("source_kind"):
        raise ValueError("Real and synthetic runs cannot be compared")
    if previous.get("order_seeds") != new.get("order_seeds") or not previous.get("order_seeds"):
        raise ValueError("Order seeds differ or are missing")
    if len(previous["order_seeds"]) != len(set(previous["order_seeds"])):
        raise ValueError("Duplicate order seed")
    old_config, config = previous.get("config", {}), new.get("config", {})
    if old_config.get("feedback_weighting", "balanced") != "balanced" or config.get("feedback_weighting") != "train_prior":
        raise ValueError("Expected previous balanced and new train_prior runs")
    if not PREVIOUS_REQUIRED_ARMS <= set(arms(previous)) or set(arms(new)) != NEW_ARMS:
        raise ValueError("Weighting control needs previous baseline/mean/full and exactly two new mean/full arms")
    prior = config.get("training_positive_prior")
    if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not math.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("Missing or invalid training positive prior")
    if config.get("feedback_positive_weight", prior) != prior:
        raise ValueError("New feedback positive weight differs from the training positive prior")
    if old_config.get("feedback_positive_weight", .5) != .5:
        raise ValueError("Previous balanced feedback positive weight must equal 0.5")
    for key in FIXED_CONFIG:
        if key not in config or key not in old_config or config[key] != old_config[key]:
            raise ValueError("Fixed training setting differs: " + key)
    if not previous.get("features_sha256") or previous["features_sha256"] != new.get("features_sha256"):
        raise ValueError("Feature hashes differ or are missing")
    old_input, new_input = previous.get("input_provenance", {}), new.get("input_provenance", {})
    for key in ("checkpoint_sha256", "dataset_sha256"):
        if old_input.get(key) != new_input.get(key) or (new.get("source_kind") == "real" and not old_input.get(key)):
            raise ValueError("Input provenance differs or is missing: " + key)
    if new.get("source_kind") == "real":
        if verify_protocol(previous) != REGISTERED_PREVIOUS_PROTOCOL_SHA256:
            raise ValueError("Previous real run is not the registered NUS weighting reference")
        for package in ("torch", "numpy", "scipy"):
            old_version = previous.get("versions", {}).get(package)
            if not old_version or old_version != new.get("versions", {}).get(package):
                raise ValueError("Numerical dependency versions differ: " + package)


def check_starts(path, manifest, weights_hashes):
    entries = read_json(Path(path) / "branch_starts.json")
    expected = {(s, a) for s in manifest["order_seeds"] for a in arms(manifest)}
    actual = [(row["seed"], row["arm"]) for row in entries]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Branch starts are duplicated, missing or unexpected")
    if (len(weights_hashes) != 2 or any(row.get("weights_sha256") != list(weights_hashes)
                                      or row.get("optimizer_state_entries") != 0 for row in entries)):
        raise ValueError("Initial weights or Adam states differ")


def _check_reference(previous_path, new_manifest, support, schedules, weights_hashes):
    """Read-only identity checks shared by the real preflight and result comparison."""
    previous_path = Path(previous_path)
    previous = read_json(previous_path / "run_manifest.json")
    completed(previous_path, previous)
    compare_manifests(previous, new_manifest)
    check_starts(previous_path, previous, weights_hashes)
    if read_json(previous_path / "support.json") != plain(support):
        raise ValueError("Support pair identities differ")
    refs = ["run_manifest.json", "status.json", "branch_starts.json", "support.json", "final.csv"]
    expected_seeds = set(previous["order_seeds"])
    if set(map(int, schedules)) != expected_seeds:
        raise ValueError("Planned schedule seed set differs")
    for seed in previous["order_seeds"]:
        name = f"batches_{seed}.json"
        schedule = schedules[seed] if seed in schedules else schedules[str(seed)]
        if read_json(previous_path / name) != plain(schedule):
            raise ValueError(f"Batch identities or order differ for seed {seed}")
        refs.append(name)
    return dict(checked=True, previous_protocol_sha256=previous.get("protocol_sha256"),
                previous_source_kind=previous["source_kind"],
                previous_files_sha256={name: sha256(previous_path / name) for name in refs},
                invariant_scope="Same inputs, initial weights, empty Adam, support, order seeds, all batches and fixed training settings; no test-based parameter choice")


def check_previous(previous_path, new_manifest, support, schedules, weights_hashes):
    """Validate real run identity and every planned batch before an optimizer update."""
    if new_manifest.get("source_kind") != "real":
        raise ValueError("Training with --previous-run requires registered real inputs")
    return _check_reference(previous_path, new_manifest, support, schedules, weights_hashes)


def add_raw_diagnostics(metrics, indexed):
    """Parse shared raw columns without changing the historical summary schema."""
    result = list(metrics)
    first = next(iter(indexed.values()))
    for metric in RAW_DIAGNOSTICS:
        if metric not in first or metric in result:
            continue
        for row in indexed.values():
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid raw diagnostic: " + metric) from error
            if not math.isfinite(value) or value < 0 or (metric == "mean_probability" and value > 1):
                raise ValueError("Nonfinite or out-of-range raw diagnostic: " + metric)
            row[metric] = value
        result.append(metric)
    return tuple(result)


def compare_runs(previous_run, new_run, out_dir=None):
    previous_run, new_run = Path(previous_run), Path(new_run)
    old_manifest, old_seeds, old_metrics, old = load_results(previous_run)
    manifest, seeds, new_metrics, new = load_results(new_run)
    old_metrics, new_metrics = add_raw_diagnostics(old_metrics, old), add_raw_diagnostics(new_metrics, new)
    completed(new_run, manifest)
    verify_protocol(manifest)
    if seeds != old_seeds:
        raise ValueError("Endpoint seeds differ")
    starts = read_json(new_run / "branch_starts.json")
    if not starts:
        raise ValueError("Missing new branch starts")
    weights = starts[0]["weights_sha256"]
    check_starts(new_run, manifest, weights)
    basis = _check_reference(previous_run, manifest, read_json(new_run / "support.json"),
                            {s: read_json(new_run / f"batches_{s}.json") for s in seeds}, weights)
    recorded = manifest.get("previous_run_control")
    if manifest["source_kind"] == "real" and not recorded:
        raise ValueError("Real weighting comparison requires the recorded pre-training reference check")
    if recorded is not None and (recorded.get("previous_files_sha256") != basis["previous_files_sha256"]
                                 or recorded.get("previous_protocol_sha256") != basis["previous_protocol_sha256"]):
        raise ValueError("Reference files changed after the pre-training check")
    # Additional new diagnostics have no fabricated historical counterpart.
    metrics = tuple(m for m in new_metrics if m in old_metrics)
    if not set(REQUIRED_METRICS) <= set(metrics):
        raise ValueError("Missing a core comparison metric")
    for seed in seeds:
        for arm in NEW_ARMS:
            if (seed, arm, "epoch0") not in new or (seed, arm, "epoch0") not in old:
                raise ValueError("Both runs must record their common epoch-zero endpoints")
            for metric in metrics:
                if not math.isclose(new[seed, arm, "epoch0"][metric], old[seed, arm, "epoch0"][metric],
                                    rel_tol=0, abs_tol=1e-10):
                    raise ValueError("Common starting endpoint differs: " + metric)
            for selection in (*SELECTIONS, "epoch0"):
                for field in ("queries", "pairs"):
                    if field not in new[seed, arm, selection] or field not in old[seed, arm, selection]:
                        raise ValueError("Missing evaluation population counts")
                    if new[seed, arm, selection][field] != old[seed, arm, selection][field]:
                        raise ValueError("Evaluation populations differ")
    values, aggregates = [], []
    for selection in SELECTIONS:
        for metric in metrics:
            factor = 100.0 if metric in HIGHER else 1.0
            direction = "higher" if metric in HIGHER else "lower" if metric in LOWER else "descriptive"
            by_contrast = {}
            for seed in seeds:
                nf, nm = (new[seed, arm, selection][metric] for arm in ("full", "predictive_mean"))
                of, om, original = (old[seed, arm, selection][metric] for arm in ("full", "predictive_mean", "original"))
                terms = {
                    "new_full-minus-new_predictive_mean": (nf, nm),
                    "previous_full-minus-previous_predictive_mean": (of, om),
                    "full_minus_mean_new-minus-previous": (nf-nm, of-om),
                    "new_full-minus-previous_original": (nf, original),
                    "new_predictive_mean-minus-previous_original": (nm, original),
                }
                for contrast, (left, right) in terms.items():
                    delta = factor * (left-right)
                    win = (delta > 0 if direction == "higher" else delta < 0) if direction != "descriptive" else None
                    row = dict(seed=seed, selection=selection, contrast=contrast, metric=metric,
                               value_left=factor*left, value_right=factor*right, difference=delta,
                               difference_unit="percentage_points" if factor == 100 else "raw",
                               value_unit=("percentage_points" if contrast == "full_minus_mean_new-minus-previous" else "percent") if factor == 100 else "raw",
                               value_kind="within-run differences" if contrast == "full_minus_mean_new-minus-previous" else "endpoint values",
                               direction=direction, win=win)
                    values.append(row)
                    by_contrast.setdefault(contrast, []).append(row)
            for contrast, records in by_contrast.items():
                differences = [row["difference"] for row in records]
                wins = sum(row["win"] for row in records) if direction != "descriptive" else None
                zeros = sum(d == 0 for d in differences)
                aggregates.append(dict(selection=selection, contrast=contrast, metric=metric,
                    n_seeds=len(seeds), mean_difference=mean(differences), min_difference=min(differences),
                    max_difference=max(differences), n_positive=sum(d > 0 for d in differences),
                    n_negative=sum(d < 0 for d in differences), n_zero=zeros, n_wins=wins,
                    n_losses=len(seeds)-wins-zeros if wins is not None else None,
                    direction=direction, difference_unit=records[0]["difference_unit"]))
    result = dict(
        title="SYNTHETIC WEIGHTING CONTROL — NOT NUS PERFORMANCE" if manifest["source_kind"] == "synthetic" else "NUS fixed train-prior weighting control",
        source_kind=manifest["source_kind"], primary_selection="final", secondary_selection="val_best",
        previous_protocol_sha256=old_manifest.get("protocol_sha256"), new_protocol_sha256=manifest.get("protocol_sha256"),
        previous_weighting="balanced", new_weighting="train_prior", training_positive_prior=manifest["config"]["training_positive_prior"],
        order_seeds=seeds, verification=basis, comparison_metrics=list(metrics),
        pre_training_check_recorded=recorded is not None,
        new_only_metrics=[m for m in new_metrics if m not in old_metrics],
        previous_only_metrics=[m for m in old_metrics if m not in new_metrics],
        missing_metric_policy="Only explicitly listed shared metrics are compared; absent historical diagnostics are never filled with zero.",
        interpretation="Descriptive control at one checkpoint. Reweighting changes the objective and gradient scale; a contrast interaction is not proof of a pure weighting mechanism or probability coverage.",
        paired_summary=aggregates,
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
        for name, rows in (("per_seed", values), ("paired_summary", aggregates)):
            with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (out_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.previous_run, args.run, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "paired_summary"}, indent=2))
    for row in result["paired_summary"]:
        if row["selection"] == "final" and row["metric"] in REQUIRED_METRICS:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
