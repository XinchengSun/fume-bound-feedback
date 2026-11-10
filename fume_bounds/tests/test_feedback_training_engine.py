"""Numerical compatibility with the archived teacher; synthetic inputs only."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fume_bounds.feedback_training import engine


def archived_reference(pack, support, batch):
    """Execute unmodified archived functions without their data-loading globals."""
    source = ROOT / "fume_bounds/certification/replay/sources/experiment.py"
    names = {"emb", "design", "posterior", "state", "sufficient", "entropy_from_logit",
             "entropy", "teacher", "crossed", "event", "objective", "batch_terms"}
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise AssertionError("Archived reference functions changed; audit the test reference")
    nodes, mass = hermgauss(24)
    context = dict(torch=torch, np=np, F=F, DATA=pack, support=support, batch=batch,
                   I=pack["train"][0], T=pack["train"][1], Y=pack["train"][2],
                   calc_loss=engine.calc_loss, Z95=1.6448536269514722,
                   nodes=torch.tensor(nodes*np.sqrt(2)), mass=torch.tensor(mass/np.sqrt(np.pi)))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), context)
    return context


def fixture():
    generator = torch.Generator().manual_seed(94173)
    image = torch.randn(80, 13, generator=generator, dtype=torch.float64)
    text = torch.randn(80, 13, generator=generator, dtype=torch.float64)
    labels = torch.arange(80) % 10
    pack = {"train": (image, text, labels), "W": torch.eye(10, dtype=torch.float64)}
    wi = torch.randn(10, 13, generator=generator, dtype=torch.float64)
    wt = torch.randn(10, 13, generator=generator, dtype=torch.float64)
    # Include both positive and negative support observations without using test labels.
    support = torch.stack((torch.arange(40), torch.cat((torch.arange(20), torch.arange(21, 41)))))
    return pack, support, torch.arange(20), wi, wt


class EngineCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_rows_kept_and_order_reproducible(self):
        batches = engine.make_batches(8000, 300, 2, 110104)
        again = engine.make_batches(8000, 300, 2, 110104)
        for first, second in zip(batches, again):
            joined = torch.cat(first)
            self.assertEqual(len(first), 27)
            torch.testing.assert_close(joined.sort().values, torch.arange(8000))
            torch.testing.assert_close(joined, torch.cat(second))
        tail = engine.make_batches(601, 300, 1, 91)[0]
        self.assertEqual([len(x) for x in tail], [300, 301])
        torch.testing.assert_close(torch.cat(tail).sort().values, torch.arange(601))

    def test_posterior_values_and_implicit_gradient_match_archive(self):
        pack, support, batch, _, _ = fixture()
        old = archived_reference(pack, support, batch)
        generator = torch.Generator().manual_seed(291)
        initial = torch.cat((torch.ones(64, 1, dtype=torch.float64),
                             torch.randn(64, 10, generator=generator, dtype=torch.float64)), 1)
        labels = torch.arange(64).remainder(3) == 0
        actual_x = initial.clone().requires_grad_()
        expected_x = initial.clone().requires_grad_()
        actual = engine.posterior(actual_x, labels)
        expected = old["posterior"](expected_x, labels)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
        actual_loss = actual[0].square().sum() + .17*actual[1].square().sum()
        expected_loss = expected[0].square().sum() + .17*expected[1].square().sum()
        actual_grad = torch.autograd.grad(actual_loss, actual_x)[0]
        expected_grad = torch.autograd.grad(expected_loss, expected_x)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)

    def test_objectives_and_encoder_gradients_match_archive(self):
        pack, support, ids, initial_i, initial_t = fixture()
        experiment = engine.Experiment(pack, support)
        old = archived_reference(pack, support, ids)
        actual_gradients = {}
        actual_values = {}
        for arm in engine.ARMS:
            with self.subTest(arm=arm):
                wi, wt = [x.clone().requires_grad_() for x in (initial_i, initial_t)]
                ri, rt = [x.clone().requires_grad_() for x in (initial_i, initial_t)]
                actual, _ = experiment.loss(wi, wt, ids, arm)
                base, _, full, _ = old["batch_terms"](ri, rt, old["teacher"])
                a, b = old["emb"](pack["train"][0][ids], ri), old["emb"](pack["train"][1][ids], rt)
                mu, sd = old["sufficient"](old["crossed"](a, b), old["state"](ri, rt)).T
                labels = (pack["train"][2][ids, None] == pack["train"][2][ids][None, :]).flatten()
                if arm == "original":
                    expected = base
                elif arm == "full":
                    expected = base + full
                elif arm == "stop_sd":
                    point = F.softplus(-(2*labels.double()-1)*mu + old["Z95"]*sd.detach())
                    expected = base + .5*(point[labels].mean() + point[~labels].mean())
                else:
                    probability = old["teacher"](torch.stack((mu, sd), 1))[:, 2]
                    point = torch.where(labels, -probability.log(), -torch.log1p(-probability))
                    expected = base + .5*(point[labels].mean() + point[~labels].mean())
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                ga = torch.autograd.grad(actual, (wi, wt))
                ge = torch.autograd.grad(expected, (ri, rt))
                for a, b in zip(ga, ge):
                    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
                actual_gradients[arm] = torch.cat([g.flatten() for g in ga])
                actual_values[arm] = actual.item()
        self.assertEqual(actual_values["full"], actual_values["stop_sd"])
        self.assertGreater((actual_gradients["full"]-actual_gradients["stop_sd"]).norm().item(), 1e-6)

    def test_predictive_loss_remains_finite_at_extreme_logits(self):
        pack, support, _, _, _ = fixture()
        experiment = engine.Experiment(pack, support)
        mu = torch.tensor([1000., -1000., 1000., -1000.], requires_grad=True, dtype=torch.float64)
        sd = torch.ones(4, requires_grad=True, dtype=torch.float64)
        labels = torch.tensor([True, True, False, False])
        loss = experiment.feedback_loss(mu, sd, labels, "predictive_mean")
        self.assertTrue(torch.isfinite(loss))
        for gradient in torch.autograd.grad(loss, (mu, sd)):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_actual_runner_keeps_adam_history_and_defers_test_evaluation(self):
        from fume_bounds.feedback_training import run
        config = SimpleNamespace(epochs=2, batch_size=32, lr=5e-5,
                                 feedback_weight=1., threads=1, seeds=[110104])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            pack = run.synthetic_pack()
            feature = root / "synthetic_features.pt"
            torch.save(pack, feature)
            metadata = dict(source_kind="synthetic", features_sha256=run.digest(feature))
            original_evaluate = run.Experiment.evaluate
            test_calls = []

            def guarded_evaluate(instance, wi, wt, split, chunk=32):
                if split == "test":
                    # This inspects the actual trainer's execution order, not a dummy optimizer.
                    self.assertTrue((output / "selections.json").is_file())
                    for arm in engine.ARMS:
                        self.assertTrue((output / "seed_110104" / arm / "final.pt").is_file())
                    test_calls.append(split)
                return original_evaluate(instance, wi, wt, split, chunk=chunk)

            with mock.patch.object(run.Experiment, "evaluate", guarded_evaluate), contextlib.redirect_stdout(io.StringIO()):
                run.train_run(pack, metadata, config, output)
            self.assertEqual(len(test_calls), 1 + 2*len(engine.ARMS))
            for arm in engine.ARMS:
                state = torch.load(output / "seed_110104" / arm / "final.pt", weights_only=True)
                self.assertEqual(state["global_step"], 6)
                self.assertEqual(state["epoch"], 2)
                self.assertEqual(len(state["optimizer"]["state"]), 2)
                self.assertEqual({int(s["step"]) for s in state["optimizer"]["state"].values()}, {6})
                for key, weight in zip(("wI", "wT"), state["weights"]):
                    self.assertGreater((weight-pack[key]).abs().max().item(), 0)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertFalse(status["real_training_completed"])
            self.assertTrue(status["synthetic_training_completed"])


if __name__ == "__main__":
    unittest.main()
