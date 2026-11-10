"""Unit fixtures exercise comparison guards; they do not represent model training."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "feedback_training"
sys.path.insert(0, str(MODULE))
SPEC = importlib.util.spec_from_file_location("feedback_weighting_compare", MODULE / "compare_weighting.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class WeightingComparison(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.old, self.new = self.root / "balanced", self.root / "train_prior"
        self.seeds = [11, 12]
        self.weights = ["a"*64, "b"*64]
        self.support = [[0, 1], [2, 3]]
        self.schedules = {s: [[[0, 1], [2, 3]], [[3, 2], [1, 0]]] for s in self.seeds}
        self.config = dict(epochs=2, batch_size=2, lr=5e-5, feedback_weight=1., adam_betas=[.5, .999],
            alpha=.5, contrastive_temperature=1., threads=3, support_seed=41531, support_pairs=2,
            training_rows=4, support_rows_also_train=True, posterior="fixed test definition",
            primary_endpoint="fixed final epoch", secondary_endpoint="validation best",
            checkpoint_initialization_count=1, order_seeds_are_model_seeds=False, units="raw probability")
        self.old_manifest, self.old_rows = self.fixture(self.old, False)
        self.new_manifest, self.new_rows = self.fixture(self.new, True)

    def fixture(self, path, new):
        path.mkdir()
        config = dict(self.config, arms=["predictive_mean", "full"] if new else list(compare.ARMS),
                      feedback_positive_weight=.1 if new else .5)
        if new:
            config.update(feedback_weighting="train_prior", training_positive_prior=.1)
        manifest = dict(source_kind="synthetic", synthetic=True, order_seeds=self.seeds,
            config=config, features_sha256="f"*64,
            versions=dict(torch="fixture-torch", numpy="fixture-numpy", scipy="fixture-scipy"),
            input_provenance=dict(checkpoint_sha256="1"*64, dataset_sha256="2"*64))
        manifest["protocol_sha256"] = compare.protocol_sha256(manifest)
        (path / "run_manifest.json").write_text(json.dumps(manifest))
        (path / "status.json").write_text(json.dumps(dict(status="completed", source_kind="synthetic",
            synthetic_training_completed=True, real_training_completed=False, test_evaluated=True,
            protocol_sha256=manifest["protocol_sha256"])))
        (path / "support.json").write_text(json.dumps(self.support))
        for seed, batches in self.schedules.items():
            (path / f"batches_{seed}.json").write_text(json.dumps(batches))
        starts = [dict(seed=s, arm=a, weights_sha256=self.weights, optimizer_state_entries=0)
                  for s in self.seeds for a in config["arms"]]
        (path / "branch_starts.json").write_text(json.dumps(starts))
        rows = []
        score = {"original": .59, "predictive_mean": .62 if new else .60,
                 "full": .61 if new else .58, "stop_sd": .57}
        for seed in self.seeds:
            for arm in config["arms"]:
                for selection in ("epoch0", "final", "val_best"):
                    value = .5 if selection == "epoch0" else score[arm] - (.01 if selection == "val_best" else 0)
                    row = dict(seed=seed, arm=arm, selection=selection,
                        epoch=0 if selection == "epoch0" else 2 if selection == "final" else 1,
                        split="test", i2t=value, t2i=value, map=value, brier=.1, nll=.3, ece=.02,
                        AU=.1, queries=4, pairs=16, TU=.11, mean_probability=.1, logit_sd=.5)
                    if new:
                        row["EU"] = .01  # Intentionally absent from old data; must not be filled with zero.
                    rows.append(row)
        self.write_rows(path, rows)
        return manifest, rows

    @staticmethod
    def write_rows(path, rows):
        with (path / "final.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def mark_real(self):
        # Mock metadata for preflight unit checks only; no training or data loader is invoked.
        for path, manifest in ((self.old, self.old_manifest), (self.new, self.new_manifest)):
            manifest.update(source_kind="real", synthetic=False)
            manifest["protocol_sha256"] = compare.protocol_sha256(manifest)
            (path / "run_manifest.json").write_text(json.dumps(manifest))
            (path / "status.json").write_text(json.dumps(dict(status="completed", source_kind="real",
                real_training_completed=True, synthetic_training_completed=False, test_evaluated=True,
                protocol_sha256=manifest["protocol_sha256"])))
        self.enterContext(patch.object(compare, "REGISTERED_PREVIOUS_PROTOCOL_SHA256",
                                      self.old_manifest["protocol_sha256"]))

    def preflight(self, manifest=None, support=None, schedules=None, weights=None):
        return compare.check_previous(self.old, manifest or self.new_manifest,
            self.support if support is None else support,
            self.schedules if schedules is None else schedules, self.weights if weights is None else weights)

    def test_real_preflight_records_identity_and_rejects_synthetic_training(self):
        with self.assertRaisesRegex(ValueError, "registered real"):
            self.preflight()
        self.mark_real()
        result = self.preflight()
        self.assertTrue(result["checked"])
        self.assertEqual(result["previous_protocol_sha256"], self.old_manifest["protocol_sha256"])
        self.assertIn("final.csv", result["previous_files_sha256"])
        self.assertEqual(result["previous_files_sha256"]["support.json"], compare.sha256(self.old / "support.json"))

    def test_declared_weighting_matches_effective_positive_weight(self):
        compare.compare_manifests(self.old_manifest, self.new_manifest)
        legacy = copy.deepcopy(self.old_manifest)
        del legacy["config"]["feedback_positive_weight"]
        compare.compare_manifests(legacy, self.new_manifest)
        changed = copy.deepcopy(self.new_manifest)
        changed["config"]["feedback_positive_weight"] = .5
        with self.assertRaisesRegex(ValueError, "differs from the training positive prior"):
            compare.compare_manifests(self.old_manifest, changed)
        changed = copy.deepcopy(self.old_manifest)
        changed["config"]["feedback_positive_weight"] = .1
        with self.assertRaisesRegex(ValueError, "must equal 0.5"):
            compare.compare_manifests(changed, self.new_manifest)

    def test_preflight_rejects_changed_inputs_protocol_support_schedule_or_start(self):
        self.mark_real()
        for key in ("epochs", "lr", "feedback_weight", "support_seed"):
            altered = copy.deepcopy(self.new_manifest)
            altered["config"][key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Fixed training setting"):
                self.preflight(manifest=altered)
        altered = copy.deepcopy(self.new_manifest)
        altered["features_sha256"] = "e"*64
        with self.assertRaisesRegex(ValueError, "Feature hashes"):
            self.preflight(manifest=altered)
        altered = copy.deepcopy(self.new_manifest)
        altered["input_provenance"]["checkpoint_sha256"] = "e"*64
        with self.assertRaisesRegex(ValueError, "Input provenance"):
            self.preflight(manifest=altered)
        altered = copy.deepcopy(self.new_manifest)
        altered["versions"]["torch"] = "different-torch"
        with self.assertRaisesRegex(ValueError, "dependency versions"):
            self.preflight(manifest=altered)
        with self.assertRaisesRegex(ValueError, "Support"):
            self.preflight(support=[[1, 0], [2, 3]])
        schedule = copy.deepcopy(self.schedules)
        schedule[11][0][0] = [1, 0]
        with self.assertRaisesRegex(ValueError, "Batch identities"):
            self.preflight(schedules=schedule)
        with self.assertRaisesRegex(ValueError, "Initial weights"):
            self.preflight(weights=["e"*64, self.weights[1]])
        starts = compare.read_json(self.old / "branch_starts.json")
        starts.append(starts[0])
        (self.old / "branch_starts.json").write_text(json.dumps(starts))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.preflight()

    def test_comparison_units_interaction_and_explicit_absent_metrics(self):
        result = compare.compare_runs(self.old, self.new, self.root / "comparison")
        self.assertIn("NOT NUS PERFORMANCE", result["title"])
        self.assertEqual(result["new_only_metrics"], ["EU"])
        self.assertNotIn("EU", result["comparison_metrics"])
        self.assertTrue(set(compare.RAW_DIAGNOSTICS) <= set(result["comparison_metrics"]))
        table = {(r["selection"], r["contrast"], r["metric"]): r for r in result["paired_summary"]}
        for contrast, expected in (("new_full-minus-new_predictive_mean", -1),
                                   ("previous_full-minus-previous_predictive_mean", -2),
                                   ("full_minus_mean_new-minus-previous", 1),
                                   ("new_full-minus-previous_original", 2),
                                   ("new_predictive_mean-minus-previous_original", 3)):
            row = table["final", contrast, "map"]
            self.assertAlmostEqual(row["mean_difference"], expected)
            self.assertEqual(row["difference_unit"], "percentage_points")
        self.assertIsNone(table["final", "new_full-minus-new_predictive_mean", "AU"]["n_wins"])
        self.assertEqual(result["primary_selection"], "final")
        self.assertEqual(result["secondary_selection"], "val_best")

    def test_missing_endpoint_or_different_start_is_rejected(self):
        removed = self.new_rows.pop()
        self.write_rows(self.new, self.new_rows)
        with self.assertRaises(ValueError):
            compare.compare_runs(self.old, self.new)
        self.new_rows.append(removed)
        self.new_rows[0]["brier"] += .001
        self.write_rows(self.new, self.new_rows)
        with self.assertRaisesRegex(ValueError, "starting endpoint"):
            compare.compare_runs(self.old, self.new)

    def test_registered_protocol_and_reference_mutation_guards(self):
        self.mark_real()
        with patch.object(compare, "REGISTERED_PREVIOUS_PROTOCOL_SHA256", "0"*64):
            with self.assertRaisesRegex(ValueError, "registered NUS"):
                self.preflight()
        changed = copy.deepcopy(self.old_manifest)
        changed["config"]["lr"] *= 2
        (self.old / "run_manifest.json").write_text(json.dumps(changed))
        # Match the caller's LR, so failure must come from the stale protocol hash.
        caller = copy.deepcopy(self.new_manifest)
        caller["config"]["lr"] = changed["config"]["lr"]
        with self.assertRaisesRegex(ValueError, "protocol hash"):
            self.preflight(manifest=caller)
        (self.old / "run_manifest.json").write_text(json.dumps(self.old_manifest))
        self.new_manifest["previous_run_control"] = self.preflight()
        self.new_manifest["protocol_sha256"] = compare.protocol_sha256(self.new_manifest)
        (self.new / "run_manifest.json").write_text(json.dumps(self.new_manifest))
        status = compare.read_json(self.new / "status.json")
        status["protocol_sha256"] = self.new_manifest["protocol_sha256"]
        (self.new / "status.json").write_text(json.dumps(status))
        self.assertTrue(compare.compare_runs(self.old, self.new)["pre_training_check_recorded"])
        self.old_rows[1]["brier"] += .001
        self.write_rows(self.old, self.old_rows)
        with self.assertRaisesRegex(ValueError, "Reference files changed"):
            compare.compare_runs(self.old, self.new)


if __name__ == "__main__":
    unittest.main()
