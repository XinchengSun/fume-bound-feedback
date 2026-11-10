import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "feedback_training" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("feedback_summary", MODULE)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = dict(source_kind="synthetic", synthetic=True, order_seeds=[11, 12], config={"epochs": 5})
        self.rows = []
        for seed in self.manifest["order_seeds"]:
            for selection in ("epoch0", "final", "val_best"):
                for arm in summary.ARMS:
                    # val_best deliberately has worse TEST mAP than final.
                    score = .5 if selection == "val_best" else .6
                    if arm == "full":
                        score += .01
                    self.rows.append(dict(seed=seed, arm=arm, selection=selection,
                        epoch=0 if selection == "epoch0" else 5 if selection == "final" else 2,
                        split="test", i2t=score, t2i=score, map=score,
                        brier=.1 - (.01 if arm == "full" else 0), nll=.3, ece=.02, AU=.1, EU=.01,
                        queries=20, pairs=400))
        self.write()

    def write(self):
        (self.root / "run_manifest.json").write_text(json.dumps(self.manifest))
        with (self.root / "final.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_units_direction_and_synthetic_scope(self):
        result = summary.summarize_run(self.root, self.root / "summary")
        table = {(r["selection"], r["contrast"], r["metric"]): r for r in result["paired_summary"]}
        map_row = table["final", "full-minus-predictive_mean", "map"]
        self.assertAlmostEqual(map_row["mean_difference"], 1.0)
        self.assertEqual(map_row["difference_unit"], "percentage_points")
        self.assertEqual(map_row["n_wins"], 2)
        self.assertEqual(table["final", "full-minus-predictive_mean", "brier"]["n_wins"], 2)
        self.assertIsNone(table["final", "full-minus-predictive_mean", "EU"]["n_wins"])
        self.assertTrue(result["synthetic"])
        self.assertIn("NOT NUS PERFORMANCE", result["title"])
        self.assertEqual(result["manifest_test_queries"], "unknown")
        self.assertEqual(result["observed_queries"], [20])

    def test_no_test_based_epoch_selection(self):
        result = summary.summarize_run(self.root)
        rows = {(r["selection"], r["contrast"], r["metric"]): r for r in result["paired_summary"]}
        self.assertAlmostEqual(rows["val_best", "full-minus-predictive_mean", "map"]["left_mean"], 51)
        self.assertAlmostEqual(rows["final", "full-minus-predictive_mean", "map"]["left_mean"], 61)
        self.rows[4]["epoch"] = 4  # A renamed test-selected endpoint cannot masquerade as final.
        self.write()
        with self.assertRaisesRegex(ValueError, "prespecified"):
            summary.summarize_run(self.root)

    def test_missing_or_duplicate_pair_is_rejected(self):
        removed = self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            summary.summarize_run(self.root)
        self.rows.extend([removed, removed.copy()])
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summary.summarize_run(self.root)

    def test_nonfinite_percentage_and_seed_mismatch_are_rejected(self):
        for field, value in (("nll", "nan"), ("map", "60"), ("seed", "999")):
            with self.subTest(field=field):
                original = self.rows[0][field]
                self.rows[0][field] = value
                self.write()
                with self.assertRaises(ValueError):
                    summary.summarize_run(self.root)
                self.rows[0][field] = original


if __name__ == "__main__":
    unittest.main()
