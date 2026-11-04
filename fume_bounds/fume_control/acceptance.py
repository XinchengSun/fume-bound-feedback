"""Verified-reference acceptance; old dual sensitivities are diagnostics only.

These are numerical certificates for a declared reference objective, not
confidence intervals for unknown population probabilities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence
import numpy as np


class ProtocolMismatch(ValueError):
    """Old/new objectives, events or risk definitions are not comparable."""


def protocol_id(spec: Mapping[str, Any]) -> str:
    """Hash the *fixed recipe*: event IDs/weights, reference IDs, kappa, etc.

    Include data/label hashes and the reference-family definition. Do NOT
    hash only a label such as 'seed2'; do not exclude kappa or event weights.
    Model parameters may change, but the mathematical recipe must not.
    """
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class BoundBracket:
    lower: float
    upper: float
    objective_id: str
    checked: bool
    note: str = ''

    def validate(self) -> None:
        if not self.objective_id or not isinstance(self.objective_id, str):
            raise ValueError('A stable, nonempty objective_id is required.')
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError('Nonfinite objective enclosure.')
        if self.lower > self.upper:
            raise ValueError('Inverted objective enclosure.')
        if not self.checked:
            raise ValueError('Unverified solver outputs cannot authorize an update.')


@dataclass(frozen=True)
class Evaluation:
    bound: BoundBracket
    risks: tuple[float, ...]
    risk_id: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.bound.validate()
        if not self.risk_id or not self.risks:
            raise ValueError('Nonempty, ordered risk components and risk_id required.')
        if not all(math.isfinite(float(x)) for x in self.risks):
            raise ValueError('Nonfinite risk.')


@dataclass(frozen=True)
class RiskGuard:
    """Budgets are fixed before retries and must come from the existing protocol.

    component_budget is an absolute risk increment, NOT a percentage of a
    candidate's own risk. No automatic budget widening is implemented.
    """
    component_budget: tuple[float, ...]
    mean_budget: float = 0.0
    risk_atol: float = 1e-10
    mean_weights: tuple[float, ...] | None = None
    anchor_risks: tuple[float, ...] | None = None
    anchor_component_budget: tuple[float, ...] | None = None

    def arrays(self, n: int):
        budget = np.asarray(self.component_budget, dtype=np.float64)
        if budget.shape != (n,) or not np.isfinite(budget).all() or (budget < 0).any():
            raise ValueError('Expected one finite nonnegative budget per risk component.')
        if self.mean_budget < 0 or self.risk_atol < 0 or not all(
                math.isfinite(x) for x in (self.mean_budget, self.risk_atol)):
            raise ValueError('Invalid mean budget or numerical tolerance.')
        if self.mean_weights is None:
            weights = np.full(n, 1.0/n)
        else:
            weights = np.asarray(self.mean_weights, dtype=np.float64)
            if (weights.shape != (n,) or not np.isfinite(weights).all()
                    or (weights < 0).any() or weights.sum() <= 0):
                raise ValueError('Invalid fixed risk aggregation weights.')
            weights = weights/weights.sum()
        if (self.anchor_risks is None) != (self.anchor_component_budget is None):
            raise ValueError('An anchor and its cumulative budgets must be supplied together.')
        if self.anchor_risks is not None:
            a = np.asarray(self.anchor_risks, dtype=np.float64)
            b = np.asarray(self.anchor_component_budget, dtype=np.float64)
            if (a.shape != (n,) or b.shape != (n,) or not np.isfinite(a).all()
                    or not np.isfinite(b).all() or (b < 0).any()):
                raise ValueError('Invalid cumulative/anchor risk guard.')
        return budget, weights


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reasons: tuple[str, ...]
    guaranteed_gain: float
    mean_risk_change: float
    max_component_increase: float
    max_budget_fraction: float
    old_hint: float | None


def decide(old: Evaluation, new: Evaluation, guard: RiskGuard, *,
           min_gain: float, old_dual_hint: float | None = None) -> Decision:
    """Accept only if a newly solved enclosure verifies the actual objective.

    lower(new) - upper(old) is the relevant conservative difference.
    lower(new) - lower(old) is NOT a certificate of improvement.
    The hint is recorded but never a veto (nor an authorization).
    """
    old.validate(); new.validate()
    if (old.bound.objective_id != new.bound.objective_id
            or old.risk_id != new.risk_id or len(old.risks) != len(new.risks)):
        raise ProtocolMismatch('Changed events/weights/reference recipe/risk schema.')
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError('min_gain must be finite and nonnegative.')
    if old_dual_hint is not None and not math.isfinite(old_dual_hint):
        raise ValueError('Nonfinite diagnostic dual hint.')
    budget, weights = guard.arrays(len(old.risks))
    delta = np.asarray(new.risks) - np.asarray(old.risks)
    mean_delta = float(weights @ delta)
    gain = float(new.bound.lower - old.bound.upper)
    reasons = []
    if not gain > min_gain:
        reasons.append('fresh_bound_gain_not_resolved')
    if mean_delta > guard.mean_budget + guard.risk_atol:
        reasons.append('mean_risk_budget_exceeded')
    if np.any(delta > budget + guard.risk_atol):
        reasons.append('component_risk_budget_exceeded')
    if guard.anchor_risks is not None:
        anchor_delta = np.asarray(new.risks) - np.asarray(guard.anchor_risks)
        if np.any(anchor_delta > np.asarray(guard.anchor_component_budget) + guard.risk_atol):
            reasons.append('cumulative_risk_budget_exceeded')
    # Only a diagnostic ratio; zero budget increments are checked separately.
    fractions = np.divide(np.maximum(delta, 0), budget,
                          out=np.zeros_like(delta), where=budget > 0)
    zero_budget_breach = (budget == 0) & (delta > guard.risk_atol)
    max_fraction = float(fractions.max())
    if zero_budget_breach.any():
        max_fraction = float('inf')
    return Decision(not reasons, tuple(reasons), gain, mean_delta,
                    float(max(0.0, delta.max())), max_fraction, old_dual_hint)


def bracket_from_bound_results(results: Sequence[Any], weights: Sequence[float], *,
                               objective_id: str, feasibility_tol: float = 2e-8,
                               max_gap: float = 2e-5,
                               numerical_padding: float = 1e-8) -> BoundBracket:
    """Adapter for the existing BoundResult(primal, dual_lower, max_violation).

    Each result solves the SAME fixed event in its current reference problem.
    Require valid primal feasibility and a checked dual lower endpoint. This
    checks numerical residuals, not formal interval-arithmetic validity. The
    producer remains responsible for actual feasibility/duality semantics.
    """
    w = np.asarray(weights, dtype=np.float64)
    if (not results or w.shape != (len(results),) or not np.isfinite(w).all()
            or (w < 0).any() or w.sum() <= 0):
        raise ValueError('Invalid fixed event weights.')
    if min(feasibility_tol, max_gap, numerical_padding) < 0:
        raise ValueError('Negative solver tolerance.')
    w = w/w.sum()
    lows, highs = [], []
    for r in results:
        lo, hi, violation = float(r.dual_lower), float(r.primal), float(r.max_violation)
        if not all(math.isfinite(x) for x in (lo, hi, violation)):
            raise ValueError('Nonfinite bound solver output.')
        if violation < 0 or violation > feasibility_tol:
            raise ValueError('Primal solution is not numerically feasible.')
        if hi - lo < -numerical_padding or hi - lo > max_gap:
            raise ValueError('Unresolved or inconsistent primal-dual gap.')
        lows.append(lo - numerical_padding)
        highs.append(hi + numerical_padding)
    out = BoundBracket(float(w @ lows), float(w @ highs), objective_id, True,
                       'numerically_checked_reference_enclosure_not_population_coverage')
    out.validate()
    return out
