"""Independent gold standards for confidence-set projection and state safety."""
import unittest
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from fume_bounds.evidence_feedback.confidence import ConfidenceSetError, LogisticConfidenceSequence


class ConfidenceSequenceTests(unittest.TestCase):
    def test_scalar_projection_encloses_independent_exact_roots(self):
        rng = np.random.default_rng(81031)
        labels = rng.binomial(1, .7, size=512)
        model = LogisticConfidenceSequence(1, radius=4.)
        exact_lower, exact_upper, previous = -4., 4., None
        for n in (32, 128, 512):
            metadata = model.update(np.ones((n, 1)), labels[:n])
            threshold, ones = metadata["threshold"], int(labels[:n].sum())
            def value(w):
                return ones*np.logaddexp(0., -w)+(n-ones)*np.logaddexp(0., w)-threshold
            center = float(np.log(ones/(n-ones)))
            left = -4. if value(-4.) <= 0 else brentq(value, -4., center, xtol=1e-13)
            right = 4. if value(4.) <= 0 else brentq(value, center, 4., xtol=1e-13)
            exact_lower, exact_upper = max(exact_lower, left), min(exact_upper, right)
            result = model.project([[1.], [-.7], [0.]])
            expected_lower = np.array([exact_lower, -.7*exact_upper, 0.])
            expected_upper = np.array([exact_upper, -.7*exact_lower, 0.])
            self.assertTrue(np.all(result["logit_lower"] <= expected_lower+1e-11))
            self.assertTrue(np.all(result["logit_upper"] >= expected_upper-1e-11))
            self.assertLess(np.max(expected_lower-result["logit_lower"]), 1e-5)
            self.assertLess(np.max(result["logit_upper"]-expected_upper), 1e-5)
            if previous is not None:
                self.assertTrue(np.all(result["lower"] >= previous["lower"]))
                self.assertTrue(np.all(result["upper"] <= previous["upper"]))
            previous = result

    def test_multidimensional_feasible_grid_is_inside_projection(self):
        rng = np.random.default_rng(91031)
        X = rng.normal(size=(256, 2))
        y = rng.binomial(1, expit(X @ [.7, -.5]))
        model = LogisticConfidenceSequence(2, radius=2.)
        for n in (64, 256):
            self.assertEqual(model.update(X[:n], y[:n])["status"], "feasible")
        queries = np.array([[1., 0.], [0., 1.], [.3, -.7], [-.8, -.1]])
        result = model.project(queries)
        feasible = np.array([w for w in rng.uniform(-2., 2., size=(12000, 2))
                             if model.contains(w)])
        self.assertGreater(len(feasible), 10)
        projections = feasible @ queries.T
        self.assertTrue(np.all(projections >= result["logit_lower"]-1e-10))
        self.assertTrue(np.all(projections <= result["logit_upper"]+1e-10))
        self.assertLess(result["diagnostics"]["maximum_primal_dual_gap"], 1e-4)

    def test_prefix_mutation_repetition_and_ids(self):
        X, y = np.ones((16, 1)), np.tile([0, 1], 8)
        model = LogisticConfidenceSequence(1)
        first = model.update(X, y, observation_ids=np.arange(16))
        repeat = model.update(X.copy(), y.copy(), observation_ids=np.arange(16))
        self.assertEqual(first["threshold"], repeat["threshold"])
        self.assertEqual(repeat["checkpoint_count"], 1)
        self.assertTrue(repeat["reused_identical_prefix"])
        changed = X.copy()
        changed[0, 0] = 2
        with self.assertRaises(ValueError):
            model.update(changed, y, observation_ids=np.arange(16))
        with self.assertRaises(ValueError):
            model.update(X[:8], y[:8], observation_ids=np.arange(8))
        with self.assertRaises(ValueError):
            model.update(X, y, observation_ids=[0]*16)
        with self.assertRaises(ValueError):
            model.update(X, y)
        X[0, 0] = 10
        self.assertNotEqual(float(model._X[0, 0]), 10.)

    def test_adversarial_prefix_can_certify_empty_and_blocks_reset(self):
        X, y = np.ones((800, 1)), np.r_[np.zeros(400), np.ones(400)]
        model = LogisticConfidenceSequence(1, radius=8.)
        self.assertEqual(model.update(X[:400], y[:400])["status"], "feasible")
        result = model.update(X, y)
        self.assertEqual(result["status"], "empty_certified")
        self.assertGreater(result["feasibility"]["empty_set_dual_lower"], 0.)
        with self.assertRaises(ConfidenceSetError):
            model.project([[1.]])
        with self.assertRaises(ConfidenceSetError):
            model.update(X, y)

    def test_zero_queries_ball_and_invalid_inputs(self):
        model = LogisticConfidenceSequence(2, radius=3.)
        result = model.project([[3., 4.], [0., 0.]])
        np.testing.assert_allclose(result["logit_lower"], [-15., 0.])
        np.testing.assert_allclose(result["logit_upper"], [15., 0.])
        self.assertTrue(np.all(result["logit_lower"] <= [-15., 0.]))
        self.assertTrue(np.all(result["logit_upper"] >= [15., 0.]))
        self.assertLessEqual(result["lower"][1], .5)
        self.assertGreaterEqual(result["upper"][1], .5)
        with self.assertRaises(ValueError):
            model.update([[1., 1.]], [.5])
        with self.assertRaises(ValueError):
            model.update([[1., np.inf]], [1])
        with self.assertRaises(ValueError):
            model.project([[np.nan, 0.]])
        with self.assertRaises(AttributeError):
            model.delta = .1
        with self.assertRaises(AttributeError):
            model.radius = 10.

    def test_any_nonnegative_multiplier_gives_global_witness_bound(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(80, 2))
        y = rng.binomial(1, expit(X @ [.4, -.6]))
        model = LogisticConfidenceSequence(2, radius=2.)
        model.update(X[:40], y[:40])
        model.update(X, y)
        feasible = [model._witness.copy()]
        feasible += [w for w in rng.normal(size=(500, 2)) if model.contains(w)]
        self.assertGreater(len(feasible), 3)
        for _ in range(20):
            objective, point = rng.normal(size=2), rng.normal(size=2)*5
            mu = rng.exponential(size=2)
            bound, _, _ = model._dual_bound(objective, point, mu)
            self.assertLessEqual(bound, np.min(np.asarray(feasible) @ objective)+1e-10)

    def test_previous_outer_bound_survives_weaker_later_solver_bound(self):
        model = LogisticConfidenceSequence(1, radius=4.)
        labels = np.tile([0., 1., 1., 1.], 128)
        model.update(np.ones((128, 1)), labels[:128])
        before = model.project([[1.], [-1.]])
        model.update(np.ones((512, 1)), labels)
        ball_only = LogisticConfidenceSequence(1, radius=4.)
        # Emulate a solver that only returns the still-valid known-ball bound.
        model._project_linear = ball_only._project_linear
        after = model.project([[1.], [-1.]])
        np.testing.assert_array_equal(before["logit_lower"], after["logit_lower"])
        np.testing.assert_array_equal(before["logit_upper"], after["logit_upper"])
        self.assertEqual(after["diagnostics"]["cached_lower_reused"], 2)
        self.assertEqual(after["diagnostics"]["cached_upper_reused"], 2)


if __name__ == "__main__":
    unittest.main()
