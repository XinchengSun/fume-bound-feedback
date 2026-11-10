"""Fixed-feature logistic confidence sequences with analytic dual projections.

Statistical validity requires the declared conditional Bernoulli model, an
ex-ante bound ||w*|| <= radius, and a fixed feature map. Each registered prefix
adds F_n(w) <= J_n(v) + .5 logdet(I+G_n/(4*lambda)) + log(1/delta).
Any finite numerical v gives the Gaussian-mixture lower bound, so coverage
does not require an exact MAP optimizer.

For g_j(w)=F_j(w)/n_j-threshold_j/n_j, mu>=0 and arbitrary v, convexity gives
  inf_{w feasible} c'w >= c'v+mu'g(v)-r'v-S||r||,
  r=c+sum_j mu_j grad(g_j(v)).
Thus a primal feasible objective is never substituted for a lower bound.
Explicit float64 arithmetic margins are applied and logged. These analytic
certificates are numerically safeguarded, not formal interval arithmetic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import expit


class ConfidenceSetError(RuntimeError):
    """An empty or unresolved evidence chain cannot produce intervals."""


@dataclass(frozen=True)
class _Checkpoint:
    n: int
    threshold: float
    threshold_raw: float
    threshold_margin: float
    map_point: np.ndarray
    map_success: bool
    map_gradient_norm: float
    prefix_sha256: str


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


class LogisticConfidenceSequence:
    """Receive complete nested data prefixes; never reset a failed chain."""

    _fixed_configuration = frozenset({
        "dimension", "delta", "regularization", "radius", "outward_tolerance", "maxiter"
    })

    def __setattr__(self, name, value):
        if name in self._fixed_configuration and name in self.__dict__:
            raise AttributeError("confidence-sequence configuration is immutable")
        super().__setattr__(name, value)

    def __init__(self, dimension, delta=0.05, regularization=1.0, radius=4.0,
                 outward_tolerance=1e-8, maxiter=250):
        if isinstance(dimension, bool) or int(dimension) != dimension or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if not (np.isfinite(delta) and 0 < delta < 1):
            raise ValueError("delta must lie in (0,1)")
        for name, value in (("regularization", regularization), ("radius", radius),
                            ("outward_tolerance", outward_tolerance)):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(name + " must be finite and positive")
        if int(maxiter) != maxiter or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        self.dimension, self.delta = int(dimension), float(delta)
        self.regularization, self.radius = float(regularization), float(radius)
        self.outward_tolerance, self.maxiter = float(outward_tolerance), int(maxiter)
        self._X, self._y = np.empty((0, self.dimension)), np.empty(0)
        self._ids, self._id_mode = None, None
        self._checkpoints = []
        self._witness = np.zeros(self.dimension)
        self._status = "ball_only"
        self._feasibility = {"status": "ball_only"}
        self._projection_cache = {}

    @property
    def num_observations(self):
        return len(self._y)

    @property
    def map_estimate(self):
        return (self._checkpoints[-1].map_point.copy() if self._checkpoints
                else np.zeros(self.dimension))

    @property
    def status(self):
        return self._status

    def _validate_xy(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError("X must have shape (observations, dimension)")
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError("one binary label is required per nonempty observation")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("observations must be finite")
        if not np.isin(y, [0., 1.]).all():
            raise ValueError("fractional targets are not Bernoulli observations")
        return np.ascontiguousarray(X), np.ascontiguousarray(y)

    @staticmethod
    def _loss_grad(X, y, w):
        z = X @ w
        loss = float(np.logaddexp(0., (1. - 2. * y) * z).sum())
        return loss, X.T @ (expit(z) - y)

    def _constraints(self, w):
        values, gradients = [], []
        for cp in self._checkpoints:
            loss, grad = self._loss_grad(self._X[:cp.n], self._y[:cp.n], w)
            # Algebraic whole-constraint rescaling; no observation reweighting.
            values.append((loss - cp.threshold) / cp.n)
            gradients.append(grad / cp.n)
        return np.asarray(values), np.asarray(gradients).reshape((-1, self.dimension))

    def _ball(self, w):
        return float(1. - w @ w / self.radius**2)

    def _ball_jac(self, w):
        return -2. * w / self.radius**2

    def _margin(self, scale):
        return self.outward_tolerance * (1. + abs(float(scale)))

    def update(self, X, y, observation_ids=None):
        if self._status in {"empty_certified", "feasibility_unresolved"}:
            raise ConfidenceSetError("failed evidence chains cannot be silently reset")
        X, y = self._validate_xy(X, y)
        old_n, n = self.num_observations, len(y)
        if n < old_n:
            raise ValueError("the data prefix cannot shrink")
        if old_n and not (np.array_equal(X[:old_n], self._X)
                          and np.array_equal(y[:old_n], self._y)):
            raise ValueError("old features or labels changed")
        id_mode = observation_ids is not None
        if self._id_mode is not None and id_mode != self._id_mode:
            raise ValueError("observation ID mode must remain fixed")
        ids = None
        if id_mode:
            ids = tuple(str(v) for v in observation_ids)
            if len(ids) != n or len(set(ids)) != n:
                raise ValueError("one distinct observation ID is required per row")
            if self._ids is not None and ids[:old_n] != self._ids:
                raise ValueError("old observation IDs changed")
        if n == old_n:
            result = self.snapshot()
            result["reused_identical_prefix"] = True
            return result

        def objective(w):
            loss, gradient = self._loss_grad(X, y, w)
            return (loss + .5 * self.regularization * (w @ w),
                    gradient + self.regularization * w)

        def hessian(w):
            p = expit(X @ w)
            return X.T @ ((p * (1. - p))[:, None] * X) + self.regularization * np.eye(self.dimension)

        fit = minimize(objective, self.map_estimate, jac=True, hess=hessian,
                       method="Newton-CG", options={"xtol": 1e-11, "maxiter": self.maxiter})
        v = np.asarray(fit.x, dtype=np.float64)
        j_value, gradient = objective(v)
        gram = X.T @ X
        sign, logdet = np.linalg.slogdet(np.eye(self.dimension) + gram / (4. * self.regularization))
        raw = float(j_value + .5 * logdet + np.log(1. / self.delta))
        if sign <= 0 or not (np.isfinite(v).all() and np.isfinite(raw)):
            raise FloatingPointError("invalid finite mixture threshold")
        margin = self._margin(abs(j_value) + abs(logdet) + abs(np.log(self.delta)))
        point = v.copy()
        point.setflags(write=False)
        cp = _Checkpoint(n, raw + margin, raw, margin, point, bool(fit.success),
                         float(np.linalg.norm(gradient)), _array_hash(X, y))
        self._X, self._y = X.copy(), y.copy()
        self._X.setflags(write=False)
        self._y.setflags(write=False)
        self._ids, self._id_mode = ids, id_mode
        self._checkpoints.append(cp)
        self._feasibility = self._find_witness(v)
        self._status = self._feasibility["status"]
        return self.snapshot()

    def _find_witness(self, candidate):
        tolerance = 10. * self.outward_tolerance
        for point in (candidate, self._witness, np.zeros(self.dimension)):
            vals, _ = self._constraints(point)
            if np.linalg.norm(point) <= self.radius and np.max(vals) <= 0.:
                self._witness = point.copy()
                return {"status": "feasible", "phase_one_run": False,
                        "maximum_normalized_violation": float(np.max(vals)),
                        "ball_violation": float(np.linalg.norm(point) - self.radius),
                        "empty_set_dual_lower": None}
        start = self._witness.copy()
        norm = np.linalg.norm(start)
        if norm >= self.radius:
            start *= .99 * self.radius / norm
        vals, _ = self._constraints(start)
        initial = np.r_[start, np.max(vals) + 1e-3]

        def constraints(z):
            return z[-1] - self._constraints(z[:-1])[0]

        def jacobian(z):
            grads = self._constraints(z[:-1])[1]
            return np.c_[-grads, np.ones(len(grads))]

        fit = minimize(
            lambda z: float(z[-1]), initial,
            jac=lambda z: np.r_[np.zeros(self.dimension), 1.], method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": constraints, "jac": jacobian},
                {"type": "ineq", "fun": lambda z: self._ball(z[:-1]),
                 "jac": lambda z: np.r_[self._ball_jac(z[:-1]), 0.]},
            ], options={"ftol": 1e-12, "maxiter": self.maxiter})
        point = np.asarray(fit.x[:-1], dtype=np.float64)
        values, gradients = self._constraints(point)
        multipliers = np.maximum(np.asarray(getattr(fit, "multipliers", []))[:len(values)], 0.)
        dual_lower = -np.inf
        if len(multipliers) == len(values) and multipliers.sum() > 0:
            weights = multipliers / multipliers.sum()
            gradient = weights @ gradients
            value = float(weights @ values - gradient @ point - self.radius * np.linalg.norm(gradient))
            margin = self._margin(np.abs(weights * values).sum() + abs(gradient @ point)
                                  + self.radius * np.linalg.norm(gradient))
            dual_lower = value - margin
        violation = float(np.max(values))
        ball_violation = float(np.linalg.norm(point) - self.radius)
        if violation <= tolerance and ball_violation <= tolerance:
            status = "feasible"
            self._witness = point.copy()
        elif dual_lower > tolerance:
            status = "empty_certified"
        else:
            status = "feasibility_unresolved"
        return {"status": status, "phase_one_run": True,
                "solver_success": bool(fit.success), "solver_message": str(fit.message),
                "maximum_normalized_violation": violation, "ball_violation": ball_violation,
                "empty_set_dual_lower": float(dual_lower) if np.isfinite(dual_lower) else None}

    def contains(self, parameter, tolerance=0.):
        point = np.asarray(parameter, dtype=np.float64)
        if point.shape != (self.dimension,) or not np.isfinite(point).all():
            raise ValueError("parameter has invalid shape or values")
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite and nonnegative")
        vals, _ = self._constraints(point)
        return bool(np.linalg.norm(point) <= self.radius + tolerance
                    and (len(vals) == 0 or np.max(vals) <= tolerance))

    def snapshot(self):
        cp = self._checkpoints[-1] if self._checkpoints else None
        return {"status": self._status, "n": self.num_observations,
                "checkpoint_count": len(self._checkpoints),
                "checkpoint_sizes": [v.n for v in self._checkpoints],
                "dimension": self.dimension, "delta": self.delta,
                "regularization": self.regularization, "radius": self.radius,
                "outward_tolerance": self.outward_tolerance,
                "threshold": None if cp is None else cp.threshold,
                "threshold_raw": None if cp is None else cp.threshold_raw,
                "threshold_margin": None if cp is None else cp.threshold_margin,
                "map_estimate": self.map_estimate.tolist(),
                "map_success": None if cp is None else cp.map_success,
                "map_gradient_norm": None if cp is None else cp.map_gradient_norm,
                "prefix_sha256": None if cp is None else cp.prefix_sha256,
                "feasibility": dict(self._feasibility)}

    def _dual_bound(self, objective, point, multipliers):
        values, gradients = self._constraints(point)
        mu = np.maximum(multipliers, 0.)
        residual = objective + mu @ gradients
        lagrangian = float(objective @ point + mu @ values)
        raw = float(lagrangian - residual @ point - self.radius * np.linalg.norm(residual))
        scale = (abs(objective @ point) + np.abs(mu * values).sum()
                 + abs(residual @ point) + self.radius * np.linalg.norm(residual))
        margin = self._margin(scale)
        return raw - margin, margin, float(np.linalg.norm(residual))

    def _project_linear(self, objective):
        norm = float(np.linalg.norm(objective))
        ball_raw = -self.radius * norm
        ball_margin = 0. if norm == 0. else self._margin(abs(ball_raw))
        ball_bound = ball_raw - ball_margin
        if norm == 0. or not self._checkpoints:
            return {"lower_bound": ball_bound, "primal_value": ball_raw, "primal_dual_gap": ball_margin,
                    "solver_success": True, "fallback": False,
                    "source": "zero_objective" if norm == 0. else "known_ball",
                    "arithmetic_margin": ball_margin, "stationarity_residual": norm,
                    "maximum_normalized_violation": 0., "ball_violation": 0.}
        fit = minimize(
            lambda w: float(objective @ w), self._witness.copy(),
            jac=lambda w: objective, method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": lambda w: -self._constraints(w)[0],
                 "jac": lambda w: -self._constraints(w)[1]},
                {"type": "ineq", "fun": self._ball, "jac": self._ball_jac},
            ], options={"ftol": 1e-11, "maxiter": self.maxiter})
        point = np.asarray(fit.x, dtype=np.float64)
        if not np.isfinite(point).all():
            point = self._witness.copy()
        values, gradients = self._constraints(point)
        best, margin, residual, source = ball_bound, ball_margin, norm, "known_ball_fallback"
        candidates = []
        multipliers = np.asarray(getattr(fit, "multipliers", []))
        if len(multipliers) >= len(values) and np.isfinite(multipliers[:len(values)]).all():
            candidates.append(("slsqp_dual", np.maximum(multipliers[:len(values)], 0.)))
        active = np.flatnonzero(values >= -1e-5)
        if len(active):
            columns = [gradients[k] for k in active]
            if np.linalg.norm(point) >= self.radius - 1e-5:
                columns.append(2. * point / self.radius**2)
            try:
                weights, _ = nnls(np.column_stack(columns), -objective, maxiter=1000)
                mu = np.zeros(len(values))
                mu[active] = weights[:len(active)]
                candidates.append(("nnls_dual", mu))
            except (ValueError, RuntimeError):
                pass
        for name, mu in candidates:
            bound, pad, resid = self._dual_bound(objective, point, mu)
            if np.isfinite(bound) and bound > best:
                best, margin, residual, source = bound, pad, resid, name
        primal = float(objective @ point)
        gap = primal - best
        feasible = (float(np.max(values)) <= 100 * self.outward_tolerance
                    and np.linalg.norm(point) <= self.radius + 100 * self.outward_tolerance)
        if feasible and gap < -100 * self.outward_tolerance:
            best, margin, residual, source = ball_bound, ball_margin, norm, "numerical_inconsistency_ball_fallback"
            gap = primal - best
        if source == "known_ball_fallback" and feasible and gap <= 100 * self.outward_tolerance:
            source = "known_ball_active"
        return {"lower_bound": float(best), "primal_value": primal,
                "primal_dual_gap": float(gap), "primal_numerically_feasible": bool(feasible),
                "solver_success": bool(fit.success), "solver_message": str(fit.message),
                "fallback": "fallback" in source, "source": source,
                "arithmetic_margin": float(margin), "stationarity_residual": float(residual),
                "maximum_normalized_violation": float(np.max(values)),
                "ball_violation": float(np.linalg.norm(point) - self.radius)}

    def project(self, query_features):
        if self._status in {"empty_certified", "feasibility_unresolved"}:
            raise ConfidenceSetError("cannot project a failed evidence chain")
        queries = np.asarray(query_features, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries[None, :]
        if queries.ndim != 2 or queries.shape[1] != self.dimension or len(queries) == 0:
            raise ValueError("query_features must have shape (queries, dimension)")
        if not np.isfinite(queries).all():
            raise ValueError("query features must be finite")
        lower, upper, endpoints = [], [], []
        for query in queries:
            lo, hi = self._project_linear(query), self._project_linear(-query)
            lower.append(lo["lower_bound"])
            upper.append(-hi["lower_bound"])
            endpoints.append({"lower": lo, "upper": hi})
        lower_z, upper_z = np.asarray(lower), np.asarray(upper)
        key = _array_hash(queries)
        cached = self._projection_cache.get(key)
        cached_lower_reused, cached_upper_reused = 0, 0
        if cached is not None:
            cached_lower_reused = int(np.count_nonzero(cached[1] > lower_z))
            cached_upper_reused = int(np.count_nonzero(cached[2] < upper_z))
            lower_z = np.maximum(lower_z, cached[1])
            upper_z = np.minimum(upper_z, cached[2])
        if np.any(lower_z > upper_z):
            raise ConfidenceSetError("outward interval intersection is empty")
        self._projection_cache[key] = (len(self._checkpoints), lower_z.copy(), upper_z.copy())
        gaps = [side["primal_dual_gap"] for item in endpoints for side in item.values()]
        # One additional outward ULP after the monotone link. Endpoint branches
        # preserve the known probability range and avoid stepping below 0 or
        # above 1. This supplements, rather than replaces, the logit margin.
        lower_p, upper_p = expit(lower_z), expit(upper_z)
        lower_p = np.where(lower_p == 0., 0., np.nextafter(lower_p, -np.inf))
        upper_p = np.where(upper_p == 1., 1., np.nextafter(upper_p, np.inf))
        return {"lower": lower_p, "upper": upper_p,
                "logit_lower": lower_z, "logit_upper": upper_z,
                "diagnostics": {"status": self._status, "n": self.num_observations,
                    "checkpoint_count": len(self._checkpoints), "query_count": len(queries),
                    "endpoint_count": 2 * len(queries),
                    "fallback_count": int(sum(side["fallback"] for item in endpoints for side in item.values())),
                    "maximum_primal_dual_gap": float(max(gaps)), "minimum_primal_dual_gap": float(min(gaps)),
                    "nested_intersection_applied": cached is not None,
                    "cached_lower_reused": cached_lower_reused, "cached_upper_reused": cached_upper_reused,
                    "arithmetic": "float64 analytic dual bound with explicit margin; not interval arithmetic",
                    "endpoints": endpoints}}
