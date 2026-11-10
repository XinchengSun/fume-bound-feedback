"""Additional declared comparisons from frozen records; no model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def analyze(out):
    out = Path(out)
    summary = json.loads((out/"summary.json").read_text())
    decisions = [json.loads(line) for line in (out/"decisions.jsonl").read_text().splitlines()]
    attempts = [json.loads(line) for line in (out/"attempts.jsonl").read_text().splitlines()]
    output = {}
    worlds = sorted({r["world"] for r in decisions})
    for policy in ("ordinary_latest", "point_guard", "certified_guard"):
        rows = [r for r in decisions if r["policy"] == policy]
        output[policy] = {
            "accepted": sum(r["accepted_nonzero_update"] for r in rows),
            "expected_decreases": sum(r["accepted_expected_decrease"] for r in rows),
            "realized_decreases": sum(r["accepted_nonzero_update"] and
                                      r["mean_realized_increment"] < -1e-12 for r in rows),
            "min_accepted_expected_gain": min(r["mean_expected_increment"] for r in rows
                                               if r["accepted_nonzero_update"]),
        }
    final_n = max(r["n"] for r in decisions)
    final = [r for r in decisions if r["n"] == final_n]
    for metric in ("mean_expected_ap", "mean_realized_ap"):
        difference = []
        for world in worlds:
            cert = np.mean([r[metric] for r in final if r["world"] == world and r["policy"] == "certified_guard"])
            point = np.mean([r[metric] for r in final if r["world"] == world and r["policy"] == "point_guard"])
            difference.append(cert-point)
        values = np.asarray(difference)
        mean = float(values.mean())
        half = float(t.ppf(.975, len(worlds)-1)*values.std(ddof=1)/np.sqrt(len(worlds)))
        output["certified_minus_point_"+metric] = {
            "mean": mean, "ci95": [mean-half, mean+half],
            "world_wins": int((values > 1e-12).sum()),
            "world_losses": int((values < -1e-12).sum()),
            "world_ties": int((abs(values) <= 1e-12).sum()),
        }
    output["max_certificate_minus_truth"] = max(r["lower_bound"]-r["true_expected_difference"] for r in attempts)
    output["beneficial_candidates_not_certified_fraction"] = (
        summary["positive_expected_candidates_without_positive_certificate"]/summary["true_positive_expected_candidates"])
    output["width_reduction"] = 1-summary["interval_curve"][-1]["mean_width"]["mean"]/summary["interval_curve"][0]["mean_width"]["mean"]
    output["radius_reduction"] = 1-summary["interval_curve"][-1]["mean_information_radius"]["mean"]/summary["interval_curve"][0]["mean_information_radius"]["mean"]
    (out/"critical_comparisons.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = analyze(parser.parse_args().out)
    print(json.dumps(result, ensure_ascii=False))
