"""Independent checks of the AP/capacity interface; no empirical certification."""

import math
import unittest

import numpy as np
from scipy.optimize import linprog, minimize_scalar

from fume_bounds.evidence_feedback.ranking import (
    average_precision,
    binary_probability_box,
    class_ap_from_scores,
    cosine_scores,
    expected_ap,
    realized_ap_from_scores,
    same_model_ap_certificate,
    simplex_box_minimum,
    stable_order,
)
from fume_bounds.evidence_feedback.uncertainty import binary_entropy, information_radius


class RankingTests(unittest.TestCase):
    def test_actual_cosine_and_stable_ties(self):
        q = np.array([[2.0, 0.0], [0.0, 1.0]])
        g = np.array([[3.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
        scores = cosine_scores(q, g)
        np.testing.assert_allclose(scores, [[1, 1, 0, -1], [0, 0, 1, 0]])
        np.testing.assert_array_equal(stable_order(scores), [[0, 1, 2, 3], [2, 0, 1, 3]])
        table = class_ap_from_scores(scores, [0, 1, 1, 0], 3)
        self.assertEqual(table[0, 0], (1 + 2 / 4) / 2)
        self.assertEqual(table[0, 1], (1 / 2 + 2 / 3) / 2)
        np.testing.assert_array_equal(table[:, 2], [0, 0])
        np.testing.assert_array_equal(
            realized_ap_from_scores(scores, [0, 1], [0, 1, 1, 0], 3),
            table[[0, 1], [0, 1]],
        )

    def test_two_item_exact_difference(self):
        old = class_ap_from_scores([[0, 1]], [0, 1])
        new = class_ap_from_scores([[1, 0]], [0, 1])
        # Bounds are expressed for p(class=1), so p(class=0) in [.4,.9].
        lo, hi = binary_probability_box([.1], [.6])
        wide = same_model_ap_certificate(new, old, lo, hi)
        lo, hi = binary_probability_box([.1], [.3])
        narrow = same_model_ap_certificate(new, old, lo, hi)
        self.assertAlmostEqual(wide["nominal_minimum"], -.1)
        self.assertAlmostEqual(narrow["nominal_minimum"], .2)
        self.assertFalse(wide["certifies_non_decrease"])
        self.assertTrue(narrow["certifies_positive"])
        self.assertLessEqual(narrow["lower_bound"], .2)
        self.assertGreater(narrow["lower_bound"], .2 - 2e-12)

    def test_random_boxes_against_independent_linear_program(self):
        rng = np.random.default_rng(1101003)
        maximum_error = 0.0
        for _ in range(220):
            k = int(rng.integers(2, 17))
            p = rng.dirichlet(np.ones(k))
            lo = p * rng.uniform(size=k)
            hi = p + (1 - p) * rng.uniform(size=k)
            c = rng.uniform(-1, 1, size=k)
            result = simplex_box_minimum(c, lo, hi)
            reference = linprog(
                c, A_eq=np.ones((1, k)), b_eq=[1], bounds=list(zip(lo, hi)), method="highs"
            )
            self.assertTrue(reference.success)
            maximum_error = max(maximum_error, abs(reference.fun - result["nominal_minimum"]))
            self.assertLessEqual(result["lower_bound"], reference.fun + 2e-15)
            self.assertLess(abs(result["nominal_minimum"] - reference.fun), 2e-12)
            self.assertLess(abs(sum(result["probabilities"]) - 1), 2e-14)
        self.assertLess(maximum_error, 2e-12)

    def test_fixed_weights_keep_all_queries_and_allow_negative_contribution(self):
        old = np.array([[.5, .5], [.6, .6]])
        new = np.array([[.9, .9], [.5, .5]])
        lo = np.zeros((2, 2))
        hi = np.ones((2, 2))
        result = same_model_ap_certificate(new, old, lo, hi, [.5, .5])
        self.assertEqual(result["query_count"], 2)
        self.assertLess(result["per_query_lower"][1], 0)
        self.assertAlmostEqual(result["nominal_minimum"], .15)
        self.assertTrue(result["certifies_positive"])
        np.testing.assert_array_equal(result["weights"], [.5, .5])

    def test_unchanged_zero_and_tiny_positive_not_certified(self):
        old = np.array([[.5, .7]])
        lo, hi = binary_probability_box([.2], [.8])
        same = same_model_ap_certificate(old, old.copy(), lo, hi)
        self.assertEqual(same["lower_bound"], 0.0)
        self.assertTrue(same["certifies_non_decrease"])
        self.assertFalse(same["certifies_positive"])
        tiny = same_model_ap_certificate(old + 1e-14, old, lo, hi)
        self.assertGreater(tiny["nominal_minimum"], 0)
        self.assertLess(tiny["lower_bound"], 0)
        self.assertFalse(tiny["certifies_positive"])

    def test_shared_mechanism_not_difference_of_separate_extrema(self):
        old = np.array([[.4, .8]])
        new = np.array([[.5, .9]])
        result = same_model_ap_certificate(new, old, [[0, 0]], [[1, 1]])
        self.assertAlmostEqual(result["nominal_minimum"], .1)
        self.assertTrue(result["certifies_positive"])
        self.assertLess(float(new.min() - old.max()), 0)

    def test_point_mean_improvement_does_not_certify(self):
        old = np.array([[.2, .9]])
        new = np.array([[.3, .8]])
        self.assertGreater(
            float(expected_ap(new, [[.9, .1]])[0] - expected_ap(old, [[.9, .1]])[0]), 0
        )
        result = same_model_ap_certificate(new, old, [[0, 0]], [[1, 1]])
        self.assertLess(result["lower_bound"], 0)

    def test_input_errors_are_not_repaired(self):
        with self.assertRaises(ValueError):
            cosine_scores([[0, 0]], [[1, 0]])
        with self.assertRaises(ValueError):
            class_ap_from_scores([[1, 2]], [0.0, 1.0])
        with self.assertRaises(ValueError):
            class_ap_from_scores([[1, 2]], [0, 2])
        with self.assertRaises(ValueError):
            simplex_box_minimum([0, 1], [.6, .6], [1, 1])
        with self.assertRaises(ValueError):
            simplex_box_minimum([0, 1], [0, 0], [.4, .4])
        with self.assertRaises(ValueError):
            binary_probability_box([.8], [.7])
        with self.assertRaises(ValueError):
            same_model_ap_certificate([[.5, .5]], [[.3, .3]], [[0, 0]], [[1, 1]], [.9])
        with self.assertRaises(ValueError):
            expected_ap([[.5, .5]], [[.4, .4]])
        with self.assertRaises(ValueError):
            average_precision([1, 2])
        self.assertEqual(average_precision([0, 0]), 0)


class InformationRadiusTests(unittest.TestCase):
    def test_closed_form_against_independent_mixture_optimization(self):
        rng = np.random.default_rng(1101004)
        intervals = [(0., 1.), (0., .1), (.2, .8), (.8, .9), (.5, .5)]
        intervals += [tuple(sorted(rng.uniform(size=2))) for _ in range(180)]
        for lower, upper in intervals:
            value = information_radius(lower, upper)
            if lower == upper:
                self.assertEqual(value, 0)
                continue
            reference = minimize_scalar(
                lambda weight: -(
                    binary_entropy((1 - weight) * lower + weight * upper)
                    - (1 - weight) * binary_entropy(lower)
                    - weight * binary_entropy(upper)
                ),
                bounds=(0, 1), method="bounded", options={"xatol": 1e-13},
            )
            self.assertTrue(reference.success)
            self.assertLess(abs(value + reference.fun), 5e-12)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, (upper - lower) * math.log(2) + 2e-14)

    def test_nested_intervals_and_tiny_width(self):
        outer = information_radius([0, .2, .4], [1, .8, .6])
        inner = information_radius([.2, .3, .45], [.8, .7, .55])
        self.assertTrue(np.all(inner <= outer))
        value = information_radius(.5 - 1e-9, .5 + 1e-9)
        self.assertGreater(value, 0)
        self.assertAlmostEqual(value / (2e-18), 1, places=6)
        self.assertGreater(information_radius(0, 1e-100), 0)
        self.assertGreater(information_radius(1e-100, 1.0001e-100), 0)
        self.assertGreater(information_radius(0, 1e-320), 0)
        # Complement calculation avoids a rounded-to-one minimax center.
        self.assertGreater(information_radius(np.nextafter(1., 0.), 1.), 0)

    def test_radius_is_not_the_eu_of_every_supported_posterior(self):
        lower, upper = .2, .8
        deterministic_posterior_eu = 0.0
        self.assertGreater(information_radius(lower, upper), deterministic_posterior_eu)
        self.assertEqual(information_radius(.5, .5), 0)
        self.assertAlmostEqual(binary_entropy(.5), math.log(2))

    def test_radius_validation_and_endpoint_entropy(self):
        np.testing.assert_array_equal(binary_entropy([0, 1]), [0, 0])
        with self.assertRaises(ValueError):
            information_radius(.8, .7)
        with self.assertRaises(ValueError):
            information_radius(-.1, .8)
        with self.assertRaises(ValueError):
            binary_entropy(np.nan)


if __name__ == "__main__":
    unittest.main()
