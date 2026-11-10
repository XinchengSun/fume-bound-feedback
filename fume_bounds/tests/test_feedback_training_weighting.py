"""Tests for the bounded train-prior weighting control, not retrieval performance."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fume_bounds.feedback_training import engine, run, summarize


class TrainingPriorWeighting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rho_uses_training_population_not_test_or_batch(self):
        pack = run.synthetic_pack()
        image, text, _ = pack["train"]
        labels = torch.cat((torch.zeros(72, dtype=torch.long), torch.ones(24, dtype=torch.long)))
        pack["train"] = image, text, labels
        support = run.support_ids(96, True)
        control = engine.Experiment(pack, support, "train_prior")
        self.assertEqual(control.training_positive_prior, .625)
        # Changing test labels cannot affect the training population weight.
        image, text, labels = pack["test"]
        pack["test"] = image, text, torch.zeros_like(labels)
        self.assertEqual(engine.Experiment(pack, support, "train_prior").positive_weight, .625)
        values = torch.tensor([2., 4., 8., 12.], dtype=torch.float64, requires_grad=True)
        pair_labels = torch.tensor([True, False, False, False])
        reduced = control.reduce_feedback(values, pair_labels)
        self.assertEqual(reduced.item(), .625*2 + .375*8)
        torch.testing.assert_close(torch.autograd.grad(reduced, values)[0],
                                   torch.tensor([.625, .125, .125, .125], dtype=torch.float64), rtol=0, atol=0)
        self.assertNotEqual(reduced.item(), values.mean().item())

    def test_sampled_gradient_diagnostics_do_not_change_actual_update(self):
        pack = run.synthetic_pack()
        ex = engine.Experiment(pack, run.support_ids(96, True), "train_prior")
        ids = torch.arange(16)
        results = []
        for diagnostics in (False, True):
            weights = [pack[k].clone().requires_grad_() for k in ("wI", "wT")]
            value, stats = ex.loss(*weights, ids, "full", diagnostics=diagnostics)
            grads = torch.autograd.grad(value, weights)
            results.append((value.detach(), grads, stats))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(list(results[0][2]), list(results[1][2]))
        self.assertIsNone(results[0][2]["feedback_grad_norm"])
        self.assertGreater(results[1][2]["feedback_grad_norm"], 0)
        self.assertGreaterEqual(results[1][2]["feedback_base_grad_cosine"], -1)
        self.assertLessEqual(results[1][2]["feedback_base_grad_cosine"], 1)

    def test_conditional_metrics_reconstruct_natural_metrics(self):
        pack = run.synthetic_pack()
        image, text, _ = pack["test"]
        labels = torch.tensor([0]*20 + [1]*4 + [2]*4 + [3]*4)
        pack["test"] = image, text, labels
        ex = engine.Experiment(pack, run.support_ids(96, True), "train_prior")
        result = ex.evaluate(pack["wI"], pack["wT"], "test")
        fraction = result["positive_pairs"] / result["pairs"]
        self.assertEqual(fraction, (20**2+3*4**2)/32**2)
        self.assertEqual(ex.training_positive_prior, .25)
        self.assertNotEqual(fraction, ex.training_positive_prior)
        for overall, conditional in (("brier", "brier"), ("nll", "nll"), ("mean_probability", "mean_q")):
            reconstructed = (fraction*result["pos_"+conditional]
                             +(1-fraction)*result["neg_"+conditional])
            self.assertAlmostEqual(reconstructed, result[overall], places=13)

    def test_legacy_cache_allowlist_is_exact(self):
        reference = run.digest(run.SRC / "extract.py")
        meta = dict(extraction_script_sha256=run.LEGACY_EXTRACTION_SHA256,
                    features_sha256=run.LEGACY_FEATURES_SHA256,
                    archived_extraction_reference_sha256=reference)
        self.assertTrue(run.extraction_source_allowed(meta))
        for field in tuple(meta):
            invalid = {**meta, field: "0"*64}
            self.assertFalse(run.extraction_source_allowed(invalid), field)

    def test_two_arm_training_and_summary_are_complete(self):
        args = SimpleNamespace(epochs=2, batch_size=32, lr=5e-5, feedback_weight=1., threads=1,
                               seeds=[110104], arms=["predictive_mean", "full"],
                               feedback_weighting="train_prior", previous_run=None)
        pack = run.synthetic_pack()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            features = folder / "synthetic_features.pt"
            torch.save(pack, features)
            output = folder / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                run.train_run(pack, dict(source_kind="synthetic", features_sha256=run.digest(features)), args, output)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["config"]["arms"], args.arms)
            self.assertEqual(manifest["config"]["feedback_weighting"], "train_prior")
            self.assertEqual(manifest["config"]["training_positive_prior"], .25)
            with (output / "training.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(int(r["gradient_diagnostic"]) for r in rows), 4)
            self.assertTrue(all(None not in r for r in rows))
            self.assertTrue(all(float(r["training_positive_prior"]) == .25 for r in rows))
            for r in rows:
                self.assertEqual(int(r["positive_pairs"])+int(r["negative_pairs"]), int(r["rows"])**2)
                self.assertEqual(bool(r["feedback_grad_norm"]), bool(int(r["gradient_diagnostic"])))
            summary = summarize.summarize_run(output)
            self.assertEqual({r["contrast"] for r in summary["paired_summary"]}, {"full-minus-predictive_mean"})
            self.assertEqual(summary["source_kind"], "synthetic")
            # Removing one declared endpoint must fail; the comparison never silently intersects seeds/arms.
            final = output / "final.csv"
            lines = final.read_text().splitlines()
            final.write_text("\n".join(lines[:-1])+"\n")
            with self.assertRaises(ValueError):
                summarize.load_results(output)


if __name__ == "__main__":
    unittest.main()
