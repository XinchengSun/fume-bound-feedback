"""Actual cosine AP and same-model expected-AP improvement certificates.

The caller must supply valid simultaneous probability bounds for the declared
latent query classes.  This module does not establish the statistical validity
of those bounds, infer labels, remove queries, or change the retrieval score.
All arrays use rows for queries and columns for classes or gallery items.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return a


def _class_labels(labels: Any, length: int, n_classes: int) -> np.ndarray:
    a = np.asarray(labels)
    if (
        a.ndim != 1
        or len(a) != length
        or a.dtype.kind not in "iu"
        or np.any(a < 0)
        or np.any(a >= n_classes)
    ):
        raise ValueError("Class labels must be integer IDs in [0,n_classes)")
    return a.astype(np.int64, copy=False)


def cosine_scores(query_embeddings: Any, gallery_embeddings: Any) -> np.ndarray:
    """Return actual cosine scores; a zero vector is an input error."""
    q = _finite_matrix(query_embeddings, "query_embeddings")
    g = _finite_matrix(gallery_embeddings, "gallery_embeddings")
    if q.shape[1] != g.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ")

    def normalize(a: np.ndarray) -> np.ndarray:
        # Scaling first avoids overflow for finite, very large coordinates.
        scale = np.max(np.abs(a), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("Cosine scores are undefined for zero embeddings")
        scaled = a / scale
        return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)

    return normalize(q) @ normalize(g).T


def stable_order(scores: Any) -> np.ndarray:
    """Descending score order; ties preserve the declared gallery input order."""
    s = _finite_matrix(scores, "scores")
    return np.argsort(-s, axis=1, kind="stable")


def average_precision(relevance_in_rank_order: Any) -> float:
    """Full-gallery binary AP, with zero positives defined to have AP=0."""
    y = np.asarray(relevance_in_rank_order)
    if y.ndim != 1 or y.size == 0 or not np.all((y == 0) | (y == 1)):
        raise ValueError("Relevance must be a nonempty binary vector")
    positions = np.flatnonzero(y) + 1
    if len(positions) == 0:
        return 0.0
    return math.fsum((i + 1) / int(rank) for i, rank in enumerate(positions)) / len(positions)


def class_ap_from_scores(scores: Any, gallery_labels: Any, n_classes: int = 2) -> np.ndarray:
    """AP[query,class] for each hypothetical query class, using actual ranks.

    This interface uses single-class equality relevance.  Multilabel retrieval
    needs a declared joint-label-state AP table instead of argmax conversion.
    """
    s = _finite_matrix(scores, "scores")
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError("n_classes must be an integer >=2")
    labels = _class_labels(gallery_labels, s.shape[1], n_classes)
    ordered_labels = labels[stable_order(s)]
    table = np.empty((len(s), n_classes), dtype=np.float64)
    for q in range(len(s)):
        for c in range(n_classes):
            table[q, c] = average_precision(ordered_labels[q] == c)
    return table


def class_ap_table(query_embeddings: Any, gallery_embeddings: Any,
                   gallery_labels: Any, n_classes: int = 2) -> np.ndarray:
    """Convenience wrapper retaining cosine as the retrieval score."""
    return class_ap_from_scores(
        cosine_scores(query_embeddings, gallery_embeddings), gallery_labels, n_classes
    )


def realized_ap_from_scores(scores: Any, query_labels: Any, gallery_labels: Any,
                            n_classes: int = 2) -> np.ndarray:
    """Evaluation-only AP with realized labels; do not pass hidden labels to training."""
    table = class_ap_from_scores(scores, gallery_labels, n_classes)
    labels = _class_labels(query_labels, len(table), n_classes)
    return table[np.arange(len(table)), labels]


def binary_probability_box(lower_p1: Any, upper_p1: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build class-0/class-1 probability boxes from bounds on Pr(class=1)."""
    lo = np.atleast_1d(np.asarray(lower_p1, dtype=np.float64))
    hi = np.atleast_1d(np.asarray(upper_p1, dtype=np.float64))
    if lo.ndim != 1 or lo.shape != hi.shape or not len(lo):
        raise ValueError("Binary bounds must be nonempty vectors with identical shape")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("Binary bounds must be finite")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
        raise ValueError("Invalid binary probability interval")
    complement_lo = np.where(hi == 1, 0.0, np.nextafter(1 - hi, -np.inf))
    complement_hi = np.where(lo == 0, 1.0, np.nextafter(1 - lo, np.inf))
    return np.column_stack((complement_lo, lo)), np.column_stack((complement_hi, hi))


def _validate_box(lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.ndim != 1 or lo.shape != hi.shape or len(lo) < 2:
        raise ValueError("Probability boxes must be equal-shaped vectors with >=2 entries")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("Probability bounds must be finite")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
        raise ValueError("Invalid probability bounds")
    if math.fsum(map(float, lo)) > 1 or math.fsum(map(float, hi)) < 1:
        raise ValueError("Probability box has no simplex intersection")
    return lo, hi


def _down(value: float) -> float:
    return math.nextafter(value, -math.inf)


def simplex_box_minimum(coefficients: Any, lower: Any, upper: Any,
                        numerical_slack: float = 1e-12) -> dict[str, Any]:
    """Minimize c.p on a box intersected with the probability simplex.

    The greedy primal gives the nominal value.  An independently evaluated
    feasible LP dual gives the conservative value:
        nu + sum_j min_{l_j<=p_j<=u_j} (c_j-nu)*p_j.
    Basic arithmetic in that dual is rounded outward with nextafter.  The
    additional absolute slack covers floating AP-table and subtraction error;
    it is a declared numeric guard, not a statistical failure probability.
    Exactly zero coefficients have an exact zero result.
    """
    lo, hi = _validate_box(lower, upper)
    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape != lo.shape or not np.all(np.isfinite(c)):
        raise ValueError("Coefficients must be finite and match the probability box")
    if not np.isfinite(numerical_slack) or numerical_slack < 0:
        raise ValueError("numerical_slack must be finite and nonnegative")

    p = lo.copy()
    remaining = 1.0 - math.fsum(map(float, lo))
    order = np.argsort(c, kind="stable")
    # Any nu is dual feasible; the greedy threshold is optimal in exact arithmetic.
    nu = float(c[order[0]])
    for j in order:
        if remaining <= 0:
            break
        take = min(float(hi[j] - lo[j]), remaining)
        p[j] += take
        remaining -= take
        nu = float(c[j])
    if remaining > 32 * np.finfo(float).eps * len(p):
        raise ArithmeticError("Greedy allocation left unassigned probability mass")
    nominal = math.fsum(float(ci) * float(pi) for ci, pi in zip(c, p))
    unchanged = bool(np.all(c == 0))
    if unchanged:
        return dict(nominal_minimum=0.0, lower_bound=0.0, probabilities=p,
                    dual_threshold=0.0, numerical_guard=0.0, exact_zero=True)

    dual = nu
    for cj, lj, uj in zip(c, lo, hi):
        coefficient_lower = _down(float(cj) - nu)
        endpoint = float(lj if coefficient_lower >= 0 else uj)
        term_lower = _down(coefficient_lower * endpoint)
        dual = _down(dual + term_lower)
    guarded = _down(dual - float(numerical_slack))
    return dict(
        nominal_minimum=nominal,
        lower_bound=guarded,
        probabilities=p,
        dual_threshold=nu,
        numerical_guard=float(nominal - guarded),
        exact_zero=False,
    )


def same_model_ap_certificate(ap_new: Any, ap_old: Any, lower_probs: Any,
                              upper_probs: Any, weights: Any = None,
                              numerical_slack: float = 1e-12) -> dict[str, Any]:
    """Certify the full, fixed-weight mean conditional expected AP difference.

    All queries are retained.  The probability vector in each difference is
    shared by old and new AP.  Per-query worst cases are more conservative than
    a single common parameter across all queries, but remain valid outer bounds.
    Equal AP vectors return exact zero; their rankings need not be identical.
    """
    new = _finite_matrix(ap_new, "ap_new")
    old = _finite_matrix(ap_old, "ap_old")
    lo = _finite_matrix(lower_probs, "lower_probs")
    hi = _finite_matrix(upper_probs, "upper_probs")
    if new.shape != old.shape or lo.shape != new.shape or hi.shape != new.shape:
        raise ValueError("AP tables and probability boxes must have the same shape")
    if new.shape[1] < 2 or np.any(new < 0) or np.any(new > 1) or np.any(old < 0) or np.any(old > 1):
        raise ValueError("AP entries must lie in [0,1] with >=2 latent classes")
    if not np.isfinite(numerical_slack) or numerical_slack < 0:
        raise ValueError("numerical_slack must be finite and nonnegative")
    n = len(new)
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Weights must be a finite nonnegative vector over every query")
    weight_sum = math.fsum(map(float, w))
    if abs(weight_sum - 1.0) > 16 * np.finfo(float).eps:
        raise ValueError("Declared weights must sum to one; weights are never normalized")

    differences = new - old
    results = [
        simplex_box_minimum(differences[q], lo[q], hi[q], numerical_slack)
        for q in range(n)
    ]
    per_lower = np.array([r["lower_bound"] for r in results])
    nominal = math.fsum(float(wq) * r["nominal_minimum"] for wq, r in zip(w, results))
    all_zero = all(r["exact_zero"] for r in results)
    if all_zero:
        lower = 0.0
    else:
        lower = 0.0
        for weight, bound in zip(w, per_lower):
            lower = _down(lower + _down(float(weight) * float(bound)))
    return dict(
        lower_bound=lower,
        nominal_minimum=nominal,
        per_query_lower=per_lower,
        per_query_nominal=np.array([r["nominal_minimum"] for r in results]),
        worst_case_probabilities=np.vstack([r["probabilities"] for r in results]),
        class_ap_difference=differences,
        weights=w.copy(),
        weight_sum=weight_sum,
        query_count=n,
        all_ap_vectors_unchanged=all_zero,
        certifies_non_decrease=bool(lower >= 0),
        certifies_positive=bool(lower > 0),
        numerical_slack=float(numerical_slack),
    )


def expected_ap(ap_table: Any, probabilities: Any) -> np.ndarray:
    """Per-query expected AP for an explicit probability vector (diagnostics)."""
    ap = _finite_matrix(ap_table, "ap_table")
    p = _finite_matrix(probabilities, "probabilities")
    if ap.shape != p.shape or np.any(ap < 0) or np.any(ap > 1) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("AP and probability tables must share shape and lie in [0,1]")
    if np.any(np.abs(p.sum(axis=1) - 1) > 16 * np.finfo(float).eps):
        raise ValueError("Probabilities must sum to one without normalization")
    return np.sum(ap * p, axis=1)
