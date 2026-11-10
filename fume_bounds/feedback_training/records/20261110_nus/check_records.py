"""Verify the archived raw NUS run and optionally render its unsmoothed validation curve."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
ARMS = ("original", "predictive_mean", "stop_sd", "full")
SEEDS = (110104, 110105, 110106)


def read_csv(name):
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(name):
    return json.loads((ROOT / name).read_text())


def selected(rows, seed, arm):
    return [r for r in rows if int(r["seed"]) == seed and r["arm"] == arm]


def verify():
    manifest = read_json("run_manifest.json")
    status = read_json("status.json")
    assert status["real_training_completed"] and status["test_evaluated"]
    assert manifest["source_kind"] == "real" and manifest["order_seeds"] == list(SEEDS)
    training, validation, test = map(read_csv, ("training.csv", "validation.csv", "final.csv"))
    assert (len(training), len(validation), len(test)) == (1620, 72, 36)
    starts = read_json("branch_starts.json")
    assert {(s["seed"], s["arm"]) for s in starts} == {(s, a) for s in SEEDS for a in ARMS}
    assert len({tuple(s["weights_sha256"]) for s in starts}) == 1
    assert all(s["optimizer_state_entries"] == 0 for s in starts)
    paired = read_csv("summary/per_seed.csv")
    for seed in SEEDS:
        schedule = read_json(f"batches_{seed}.json")
        assert len(schedule) == 5
        for epoch in schedule:
            assert len(epoch) == 27
            assert sorted(i for batch in epoch for i in batch) == list(range(8000))
        for arm in ARMS:
            tr = selected(training, seed, arm)
            assert [int(r["step"]) for r in tr] == list(range(1, 136))
            for r in tr:
                assert len(schedule[int(r["epoch"]) - 1][int(r["batch"])]) == int(r["rows"])
            val = selected(validation, seed, arm)
            assert [int(r["epoch"]) for r in val] == list(range(6))
            best = int(max(val, key=lambda r: float(r["map"]))["epoch"])
            rows = selected(test, seed, arm)
            assert {r["selection"]: int(r["epoch"]) for r in rows} == dict(epoch0=0, final=5, val_best=best)
            for r in rows:
                assert int(r["queries"]) == 1000 and int(r["pairs"]) == 1000000
                assert abs(float(r["map"]) - mean(float(r[k]) for k in ("i2t", "t2i"))) < 1e-12
    endpoints = {(int(r["seed"]), r["arm"], r["selection"]): r for r in test}
    for r in paired:
        left, right = r["contrast"].split("-minus-")
        seed, key, sel = int(r["seed"]), r["metric"], r["selection"]
        factor = 100 if key in ("map", "i2t", "t2i") else 1
        diff = float(endpoints[seed, left, sel][key]) - float(endpoints[seed, right, sel][key])
        assert abs(float(r["difference"]) - factor * diff) < 1e-12
    print(f"Checked {len(training)} steps, {len(validation)} validation rows, "
          f"{len(test)} test endpoints and {len(paired)} paired values.")


def plot():
    import matplotlib.pyplot as plt
    rows = read_csv("validation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for seed, ax in zip(SEEDS, axes):
        for arm, marker in zip(ARMS, ("o", "^", "s", "x")):
            values = selected(rows, seed, arm)
            ax.plot([int(r["epoch"]) for r in values], [100 * float(r["map"]) for r in values],
                    marker=marker, markersize=4, linewidth=1, label=arm)
        ax.set(title=f"Order seed {seed}", xlabel="Continuation epoch", xticks=range(6))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation bidirectional mAP (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(ROOT / "validation_map.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    verify()
    if args.plot:
        plot()
