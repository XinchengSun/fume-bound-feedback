"""Binary information radius for valid probability ranges, in natural-log units.

This is the worst-case conditional log-loss regret of the interval family,
not the EU of a specified Bayesian/Laplace posterior.  Bounds on a marginal
class event do not by themselves give the whole categorical channel capacity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import xlogy, xlog1py


def binary_entropy(probability: Any) -> np.ndarray | float:
    p = np.asarray(probability, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must be finite and lie in [0,1]")
    result = -xlogy(p, p) - xlog1py(1 - p, -p)
    # xlog1py(0,-1) may be platform-dependent; use the mathematical endpoints.
    result = np.where((p == 0) | (p == 1), 0.0, result)
    return float(result) if result.ndim == 0 else result


def _bernoulli_kl(p: float, r: float) -> float:
    if p == r:
        return 0.0
    if r == 0 or r == 1:
        return math.inf
    if p == 0:
        return -math.log1p(-r)
    if p == 1:
        return -math.log(r)

    def remainder(t: float) -> float:
        # Stable (1+t)log(1+t)-t, without a nonnegativity clip.
        if abs(t) < 0.05:
            return math.fsum((-1) ** k * t**k / (k * (k - 1)) for k in range(2, 24))
        return (1 + t) * math.log1p(t) - t

    return r * remainder((p - r) / r) + (1 - r) * remainder((r - p) / (1 - r))


def _radius_scalar(lower: float, upper: float) -> float:
    if lower == upper:
        return 0.0
    # Complement symmetry keeps extremely asymmetric upper-end intervals stable.
    if lower + upper > 1.0:
        return _radius_scalar(1.0 - upper, 1.0 - lower)
    width = upper - lower
    midpoint = lower + width / 2
    distance = min(lower, 1 - upper)
    if distance > 0 and width < 1e-3 * distance:
        # The mean entropy derivative, integrated over a symmetric interval.
        # Terms through width^4 avoid subtraction of nearly equal entropies.
        slope = (
            math.log1p(-midpoint) - math.log(midpoint)
            + ((width / midpoint)**2 - (width / (1 - midpoint))**2) / 24
            + ((width / midpoint)**4 - (width / (1 - midpoint))**4) / 320
        )
    else:
        slope = (float(binary_entropy(upper)) - float(binary_entropy(lower))) / width
    # This branch also retains representable subnormal probabilities; the
    # 1/(1+exp(slope)) form can overflow before the result itself underflows.
    if slope >= 0:
        exp_negative = math.exp(-slope)
        center = exp_negative / (1 + exp_negative)
    else:
        center = 1 / (1 + math.exp(slope))
    if not lower <= center <= upper:
        # Fail explicitly rather than silently clipping an unstable closed form.
        raise ArithmeticError("Information-radius center left its probability interval")
    # At the exact center both values equal capacity. Their maximum also gives
    # a conservative minimax value when the computed center has rounding error.
    value = max(_bernoulli_kl(lower, center), _bernoulli_kl(upper, center))
    if not np.isfinite(value) or value < 0:
        raise ArithmeticError("Invalid information radius")
    return value


def information_radius(lower: Any, upper: Any) -> np.ndarray | float:
    """Closed-form binary radius, with stable small-width evaluation.

    Scalar or broadcast-compatible array inputs are accepted. This is a
    floating-point numerical evaluation; no interval-arithmetic coverage claim
    is attached to this diagnostic. The statistical validity belongs to the
    interval-producing procedure.
    """
    lo, hi = np.broadcast_arrays(
        np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    )
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("Probability bounds must be finite")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
        raise ValueError("Invalid probability interval")
    result = np.empty(lo.shape, dtype=np.float64)
    for index in np.ndindex(lo.shape):
        result[index] = _radius_scalar(float(lo[index]), float(hi[index]))
    return float(result) if result.ndim == 0 else result
