"""Plot the recorded controlled-pilot results without smoothing or retraining."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.out/"summary.json").read_text())
    config = json.loads((args.out/"config.json").read_text())
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "evidence_feedback_20261110",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    sizes = np.asarray(config["support_prefixes"])
    curve = summary["interval_curve"]
    for ax, key, label, color in (
        (axes[0, 0], "mean_width", "Mean probability-interval width", "#1f7185"),
        (axes[0, 1], "mean_information_radius", "Worst log-loss regret (nats)", "#935b1f"),
    ):
        means = np.array([row[key]["mean"] for row in curve])
        ci = np.array([row[key]["ci95"] for row in curve])
        ax.plot(sizes, means, marker="o", color=color, lw=2)
        ax.fill_between(sizes, ci[:, 0], ci[:, 1], color=color, alpha=.16)
        ax.set_ylabel(label)
    names = {"ordinary_latest": "Ordinary latest checkpoint", "point_guard": "MAP point selection",
             "certified_guard": "Certified selection"}
    colors = {"ordinary_latest": "#7e8792", "point_guard": "#bb6d28", "certified_guard": "#176f85"}
    for policy in config["policies"]:
        rows = [row for row in summary["groups"] if row["policy"] == policy]
        ap = np.array([row["mean_expected_ap"]["mean"] for row in rows])
        acceptance = [100*row["accepted_nonzero_updates"]/row["decisions"] for row in rows]
        axes[1, 0].plot(sizes, 100*ap, marker="o", label=names[policy], color=colors[policy], lw=2)
        axes[1, 1].plot(sizes, acceptance, marker="o", label=names[policy], color=colors[policy], lw=2)
    axes[1, 0].axhline(100*summary["initial_expected_ap"]["mean"], color="#aaa", ls=":", label="Common warm-up mean")
    axes[1, 0].set_ylabel("True conditional expected bidirectional mAP (%)")
    axes[1, 0].legend(fontsize=9, loc="lower right")
    axes[1, 1].set_ylabel("Nonzero updates accepted (%)")
    axes[1, 1].set_ylim(0, 105)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, labels=[str(n) for n in sizes])
        ax.set_xlabel("Unique newly observed support entities")
        ax.grid(alpha=.18)
    fig.suptitle("Controlled known-model pilot: 32 worlds, 3 student seeds per world\n"
                 "One shared training path per student; three checkpoint-selection rules", fontsize=13)
    fig.savefig(args.out/"curves.svg", metadata={"Date": None})
    fig.savefig(args.out/"curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
