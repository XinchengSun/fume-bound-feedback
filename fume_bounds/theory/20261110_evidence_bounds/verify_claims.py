#!/usr/bin/env python3
"""Finite numerical audits of the accompanying mathematical claims.

This script neither trains a retrieval model nor certifies an empirical dataset.
Run with Python, NumPy, and SciPy.  A failure is recorded and exits nonzero;
the script never repairs a failed inequality by clipping its reported values.
"""

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import brentq, linprog, minimize_scalar
from scipy.special import betaln, digamma, expit, rel_entr, xlogy


SEED = 20261110
ATOL = 2e-10


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def entropy(p):
    p = np.asarray(p, dtype=float)
    require(np.all((0 <= p) & (p <= 1)), "Entropy argument outside [0,1]")
    return -xlogy(p, p) - xlogy(1 - p, 1 - p)


def bernoulli_kl(p, r):
    if p == r:
        return 0.0
    if r == 0 or r == 1:
        return math.inf
    if p == 0:
        return -math.log1p(-r)
    if p == 1:
        return -math.log(r)

    def remainder(t):
        # (1+t) log(1+t)-t. The convergent series avoids cancellation
        # for close Bernoulli parameters; it is not a nonnegativity clip.
        if abs(t) < .05:
            return math.fsum((-1) ** n * t ** n / (n * (n - 1)) for n in range(2, 22))
        return (1 + t) * math.log1p(t) - t

    return r * remainder((p - r) / r) + (1 - r) * remainder((r - p) / (1 - r))


def information_radius(lower, upper):
    require(0 <= lower <= upper <= 1, "Invalid probability interval")
    if lower == upper:
        return 0.0, lower, 0.0
    slope = float((entropy(upper) - entropy(lower)) / (upper - lower))
    center = float(expit(-slope))
    mix = (center - lower) / (upper - lower)
    value = float(entropy(center) - entropy(lower) - (center - lower) * slope)
    return value, center, mix


def check_information_radius(rng):
    endpoints = [0.0, 1e-10, 1e-6, .001, .01, .1, .25, .45,
                 .5, .75, .9, .99, .999, 1 - 1e-6, 1 - 1e-10, 1.0]
    intervals = [(a, b) for i, a in enumerate(endpoints) for b in endpoints[i:]]
    intervals.extend(tuple(sorted(rng.uniform(size=2))) for _ in range(120))
    intervals.extend([(.5 - 1e-9, .5 + 1e-9), (0.0, .1), (.8, .9)])
    errors = dict(kl_balance=0.0, mixture_optimization=0.0,
                  endpoint_equalization=0.0, mixture_identity=0.0)
    largest_negative = 0.0
    max_monotonicity_violation = -math.inf
    max_posterior_bound_violation = -math.inf
    max_width_bound_violation = -math.inf
    degenerate = 0
    for lower, upper in intervals:
        value, center, mix = information_radius(lower, upper)
        require(np.isfinite(value), "Nonfinite information radius")
        require(value >= -ATOL, "Negative information radius beyond tolerance")
        largest_negative = min(largest_negative, value)
        require(lower - ATOL <= center <= upper + ATOL, "Center outside interval")
        if lower == upper:
            degenerate += 1
            require(value == 0 and center == lower, "Degenerate interval failure")
            continue
        require(-ATOL <= mix <= 1 + ATOL, "Mixture weight outside [0,1]")
        root = brentq(lambda r: bernoulli_kl(lower, r) - bernoulli_kl(upper, r),
                      lower, upper, xtol=5e-15, rtol=4 * np.finfo(float).eps)
        root_value = max(bernoulli_kl(lower, root), bernoulli_kl(upper, root))
        opt = minimize_scalar(
            lambda w: -float(entropy((1 - w) * lower + w * upper)
                             - (1 - w) * entropy(lower) - w * entropy(upper)),
            bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12})
        require(opt.success, "Independent mixture optimization failed")
        errors["kl_balance"] = max(errors["kl_balance"], abs(value - root_value))
        errors["mixture_optimization"] = max(errors["mixture_optimization"],
                                              abs(value + opt.fun))
        errors["endpoint_equalization"] = max(errors["endpoint_equalization"],
            abs(bernoulli_kl(lower, center) - bernoulli_kl(upper, center)))
        via_mix = float(entropy(center) - (1 - mix) * entropy(lower)
                        - mix * entropy(upper))
        errors["mixture_identity"] = max(errors["mixture_identity"], abs(value - via_mix))
        inner = information_radius(lower + .3 * (upper - lower),
                                   upper - .2 * (upper - lower))[0]
        max_monotonicity_violation = max(max_monotonicity_violation, inner - value)
        points = rng.uniform(lower, upper, size=17)
        weights = rng.dirichlet(np.ones(17))
        posterior_eu = float(entropy(weights @ points) - weights @ entropy(points))
        max_posterior_bound_violation = max(max_posterior_bound_violation,
                                            posterior_eu - value)
        max_width_bound_violation = max(max_width_bound_violation,
                                        value - (upper - lower) * math.log(2))
    require(max(errors.values()) <= ATOL, "Information-radius independent checks disagree")
    require(max_monotonicity_violation <= ATOL, "Nested interval radius increased")
    require(max_posterior_bound_violation <= ATOL, "Supported posterior exceeds radius")
    require(max_width_bound_violation <= ATOL, "Width bound failed")
    examples = []
    for lower, upper in [(0, 1), (.2, .8), (.4, .6), (.45, .55), (.8, .9), (0, .1), (.5, .5)]:
        value, center, mix = information_radius(lower, upper)
        examples.append(dict(lower=lower, upper=upper, radius_nats=value,
                             minimax_center=center, upper_endpoint_weight=mix))
    return dict(status="passed", interval_count=len(intervals),
                degenerate_count=degenerate, nondegenerate_count=len(intervals) - degenerate,
                max_absolute_errors=errors, smallest_computed_value=largest_negative,
                max_nested_increase=max_monotonicity_violation,
                max_supported_posterior_excess=max_posterior_bound_violation,
                max_linear_width_bound_excess=max_width_bound_violation,
                tolerance=ATOL, examples=examples)


def beta_eu(alpha, beta):
    total = alpha + beta
    mean = alpha / total
    au = digamma(total + 1) - mean * digamma(alpha + 1) - (1 - mean) * digamma(beta + 1)
    return float(entropy(mean) - au)


def check_beta_supermartingale():
    grid = [.05, .1, .25, .5, 1, 2, 5, 10, 20, 100]
    max_error = 0.0
    max_expected_increase = -math.inf
    rising_examples = []
    for alpha, beta in itertools.product(grid, repeat=2):
        total = alpha + beta
        mean = alpha / total
        before = beta_eu(alpha, beta)
        after_zero = beta_eu(alpha, beta + 1)
        after_one = beta_eu(alpha + 1, beta)
        expected_after = (1 - mean) * after_zero + mean * after_one
        joint = np.array([[beta * (beta + 1), alpha * beta],
                          [alpha * beta, alpha * (alpha + 1)]], dtype=float) / (total * (total + 1))
        marginal = np.array([1 - mean, mean])
        future_label_mi = float(np.sum(rel_entr(joint, marginal[:, None] * marginal[None, :])))
        max_error = max(max_error, abs(before - expected_after - future_label_mi))
        max_expected_increase = max(max_expected_increase, expected_after - before)
        require(min(before, after_zero, after_one, future_label_mi) >= -ATOL,
                "Negative Beta information beyond numerical tolerance")
        for observation, after, probability in [(0, after_zero, 1 - mean), (1, after_one, mean)]:
            if after > before + 1e-6:
                rising_examples.append(dict(alpha=alpha, beta=beta, observation=observation,
                    observation_predictive_probability=probability, eu_before=before,
                    eu_after=after, increase=after - before,
                    expected_eu_after=expected_after, future_label_mutual_information=future_label_mi))
    require(max_error <= ATOL, "Beta supermartingale information identity failed")
    require(max_expected_increase <= ATOL, "Expected Beta EU increased")
    require(rising_examples, "No realized-path EU increase found")
    selected = next((r for r in rising_examples if r["alpha"] == .1 and r["beta"] == 2
                     and r["observation"] == 1), max(rising_examples, key=lambda r: r["increase"]))
    return dict(status="passed", posterior_count=len(grid) ** 2,
                realized_updates_checked=2 * len(grid) ** 2,
                max_identity_error=max_error, max_expected_increase=max_expected_increase,
                realized_increase_count=len(rising_examples), example_realized_increase=selected,
                expectation_law="Current Beta posterior predictive, not every fixed-parameter sampling law",
                tolerance=ATOL)


def average_precision(relevance):
    relevance = np.asarray(relevance, dtype=int)
    positives = int(relevance.sum())
    if positives == 0:
        return 0.0
    return float(np.sum(relevance * np.cumsum(relevance) / np.arange(1, len(relevance) + 1)) / positives)


def check_correlated_relevance():
    outcomes = np.array([[1, 0, 1], [0, 1, 0]])
    mass = np.array([.6, .4])
    orders = {"ABC": [0, 1, 2], "BAC": [1, 0, 2], "ACB": [0, 2, 1], "CAB": [2, 0, 1]}
    expected = {name: float(sum(p * average_precision(y[order]) for p, y in zip(mass, outcomes)))
                for name, order in orders.items()}
    targets = {"ABC": .70, "BAC": .75, "ACB": 11 / 15, "CAB": 11 / 15}
    error = max(abs(expected[k] - targets[k]) for k in targets)
    require(error <= ATOL, "Correlated relevance expected AP mismatch")
    require(expected["BAC"] > expected["ACB"], "Counterexample to marginal sorting failed")
    return dict(status="passed", outcome_count=2, orders_checked=len(orders),
                joint_outcomes=outcomes.tolist(), joint_masses=mass.tolist(),
                marginal_relevance=(mass @ outcomes).tolist(), expected_ap=expected,
                max_absolute_error=error,
                note="ABC is not a full marginal-descending order; ACB and CAB are, and both remain worse than BAC.")


def greedy_box_min(coefficients, lower, upper):
    coefficients, lower, upper = map(lambda x: np.asarray(x, dtype=float),
                                     (coefficients, lower, upper))
    require(np.all((0 <= lower) & (lower <= upper) & (upper <= 1)), "Invalid simplex box")
    require(lower.sum() <= 1 + ATOL and upper.sum() >= 1 - ATOL, "Infeasible simplex box")
    solution = lower.copy()
    remaining = 1.0 - float(solution.sum())
    for index in np.argsort(coefficients, kind="stable"):
        if remaining <= 0:
            break
        take = min(float(upper[index] - solution[index]), remaining)
        solution[index] += take
        remaining -= take
    require(abs(solution.sum() - 1) <= ATOL, "Greedy solution has invalid total mass")
    require(np.all(solution >= lower - ATOL) and np.all(solution <= upper + ATOL),
            "Greedy solution outside box")
    return float(coefficients @ solution), solution


def check_simplex_box_difference(rng):
    max_error = 0.0
    max_mass_error = 0.0
    count = 220
    dimensions = []
    for _ in range(count):
        classes = int(rng.integers(2, 17))
        gallery_size = int(rng.integers(classes, 3 * classes + 10))
        gallery = np.concatenate((np.arange(classes), rng.integers(0, classes, gallery_size - classes)))
        old_order, new_order = rng.permutation(gallery_size), rng.permutation(gallery_size)
        old_ap = np.array([average_precision(gallery[old_order] == c) for c in range(classes)])
        new_ap = np.array([average_precision(gallery[new_order] == c) for c in range(classes)])
        difference = new_ap - old_ap
        feasible = rng.dirichlet(np.ones(classes))
        lower = feasible * rng.uniform(size=classes)
        upper = feasible + (1 - feasible) * rng.uniform(size=classes)
        value, solution = greedy_box_min(difference, lower, upper)
        reference = linprog(difference, A_eq=np.ones((1, classes)), b_eq=[1.0],
                            bounds=list(zip(lower, upper)), method="highs")
        require(reference.success, "Independent simplex-box LP failed")
        max_error = max(max_error, abs(value - reference.fun))
        max_mass_error = max(max_mass_error, abs(float(solution.sum()) - 1))
        dimensions.append(classes)
    edge_cases = [([.1, -.2, .3], [.2, .3, .5], [.2, .3, .5]),
                  ([.5, -.5], [0, 0], [0, 1]), ([0, 0, 0], [0, .1, .2], [.8, .5, .6])]
    for coefficients, lower, upper in edge_cases:
        value, _ = greedy_box_min(coefficients, lower, upper)
        reference = linprog(coefficients, A_eq=np.ones((1, len(lower))), b_eq=[1],
                            bounds=list(zip(lower, upper)), method="highs")
        require(reference.success, "Edge-case LP failed")
        max_error = max(max_error, abs(value - reference.fun))
    # Same query-class law is used for both rankings. No independent pair-label assumption.
    gallery = np.array([0, 1])
    old_order, new_order = [1, 0], [0, 1]
    difference = np.array([average_precision(gallery[new_order] == c)
                           - average_precision(gallery[old_order] == c) for c in [0, 1]])
    examples = []
    for name, lower, upper, target in [("wide", [.4, .1], [.9, .6], -.1),
                                      ("narrow", [.7, .1], [.9, .3], .2)]:
        value, solution = greedy_box_min(difference, lower, upper)
        require(abs(value - target) <= ATOL, "Two-item gallery certificate mismatch")
        examples.append(dict(name=name, lower=lower, upper=upper,
            minimum_expected_ap_difference=value, attaining_probability=solution.tolist(),
            certifies_positive_difference=value > 0))
    require(max_error <= ATOL, "Greedy LP differs from scipy.linprog")
    return dict(status="passed", random_boxes=count, edge_boxes=len(edge_cases),
                min_class_count=min(dimensions), max_class_count=max(dimensions),
                max_objective_error=max_error, max_mass_error=max_mass_error,
                two_item_gallery=dict(gallery_classes=gallery.tolist(), old_order=old_order,
                    new_order=new_order, per_class_ap_difference=difference.tolist(), examples=examples),
                tolerance=ATOL)


def check_ap_inversion_bound():
    count = 0
    max_ap_identity_error = 0.0
    max_bound_violation = -math.inf
    for size in range(1, 9):
        for pattern in itertools.product([0, 1], repeat=size):
            positive_positions = np.flatnonzero(pattern) + 1
            positives = len(positive_positions)
            if positives == 0:
                continue
            positive_index = np.arange(1, positives + 1)
            negative_before = positive_positions - positive_index
            inversions = int(negative_before.sum())
            ap = average_precision(pattern)
            alternate_ap = float(np.mean(positive_index / (positive_index + negative_before)))
            max_ap_identity_error = max(max_ap_identity_error, abs(ap - alternate_ap))
            bound = inversions / (positives + inversions)
            max_bound_violation = max(max_bound_violation, 1 - ap - bound)
            count += 1
    require(max_ap_identity_error <= ATOL, "AP positive-rank identity failed")
    require(max_bound_violation <= ATOL, "AP inversion bound failed")
    return dict(status="passed", maximum_gallery_size=8, binary_rankings_checked=count,
                empty_positive_rankings_excluded=8, max_ap_identity_error=max_ap_identity_error,
                max_bound_violation=max_bound_violation, bound="1 - AP <= K / (M + K)", tolerance=ATOL)


def beta_mixture_crossing_probability(true_probability, delta, horizon):
    # Dynamic programming removes already-crossed paths. At time t, the likelihood
    # ratio depends only on successes, so this computes finite-horizon hitting mass.
    survival = np.array([1.0])
    ever_crossed = 0.0
    max_mass_error = 0.0
    for time in range(1, horizon + 1):
        next_mass = np.zeros(time + 1)
        next_mass[:-1] += (1 - true_probability) * survival
        next_mass[1:] += true_probability * survival
        successes = np.arange(time + 1)
        log_mixture = betaln(successes + 1, time - successes + 1)
        log_likelihood = (xlogy(successes, true_probability)
                          + xlogy(time - successes, 1 - true_probability))
        crossed = log_mixture - log_likelihood >= math.log(1 / delta)
        ever_crossed += float(next_mass[crossed].sum())
        next_mass[crossed] = 0.0  # Absorbing boundary, not numerical repair.
        survival = next_mass
        max_mass_error = max(max_mass_error, abs(ever_crossed + float(survival.sum()) - 1))
    return ever_crossed, max_mass_error


def check_finite_horizon_confidence_sequence():
    probabilities = np.unique(np.concatenate((np.linspace(0, 1, 101), [.001, .005, .995, .999])))
    deltas = [.01, .05, .1]
    horizon = 128
    max_mass_error = 0.0
    summaries = []
    for delta in deltas:
        worst_probability, worst_parameter = -1.0, None
        for true_probability in probabilities:
            crossing, mass_error = beta_mixture_crossing_probability(float(true_probability), delta, horizon)
            require(crossing <= delta + ATOL, "Finite-horizon crossing exceeds Ville threshold")
            require(crossing >= 0, "Negative crossing probability")
            max_mass_error = max(max_mass_error, mass_error)
            if crossing > worst_probability:
                worst_probability, worst_parameter = crossing, float(true_probability)
        summaries.append(dict(delta=delta, maximum_crossing_probability=worst_probability,
                              maximizing_grid_probability=worst_parameter,
                              maximum_crossing_to_delta_ratio=worst_probability / delta))
    require(max_mass_error <= ATOL, "Confidence-sequence DP lost probability mass")
    return dict(status="passed", prior="Beta(1,1) sequential predictive mixture",
                likelihood_ratio="B(s+1,t-s+1) / (p**s * (1-p)**(t-s))",
                probability_grid=probabilities.tolist(), parameter_grid_count=len(probabilities),
                confidence_levels=len(deltas), scenarios=len(probabilities) * len(deltas),
                maximum_time=horizon, max_mass_conservation_error=max_mass_error,
                results=summaries, tolerance=ATOL,
                scope="Finite-grid, finite-horizon numerical audit; not a proof of Ville's inequality, not real-data certification.")


def main():
    source = Path(__file__).resolve()
    output = source.with_name("verification.json")
    report = dict(status="running", purpose="pure_mathematical_verification",
                  real_training_completed=False, real_dataset_used=False,
                  proofs_established_by_numerics=False, seed=SEED,
                  executed_at_utc=datetime.now(timezone.utc).isoformat(),
                  code_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  checks={})
    rng = np.random.default_rng(SEED)
    checks = [
        ("information_radius", lambda: check_information_radius(rng)),
        ("beta_eu_supermartingale", check_beta_supermartingale),
        ("correlated_relevance", check_correlated_relevance),
        ("simplex_box_same_parameter_difference", lambda: check_simplex_box_difference(rng)),
        ("ap_inversion_bound", check_ap_inversion_bound),
        ("scalar_mixture_confidence_sequence", check_finite_horizon_confidence_sequence),
    ]
    try:
        for name, check in checks:
            report["checks"][name] = check()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), check=name)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], checks_completed=len(report["checks"]),
                         output=str(output), real_training_completed=False,
                         error=report.get("error")), ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
