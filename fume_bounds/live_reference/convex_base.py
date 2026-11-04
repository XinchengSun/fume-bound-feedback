"""Data-compatible probability bounds with dual-feedback convex updates.

This implements the *fixed prediction-bank, shared convex mixture* prototype.
It is NOT end-to-end training of the encoders and does not certify arbitrary
real-world conditional probabilities. All bounds are relative to this family.

Numerical backend: SciPy SLSQP with analytic Jacobians. Each bound is checked
against a separate convex Lagrangian tangent lower bound, not merely an optimizer
success flag. Checks use floating point, not formal interval arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import minimize, lsq_linear


class OptimizationError(RuntimeError):
    """A numerical solve did not pass the required feasibility/gap checks."""


def _simplex(a: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    if a.ndim != 1 or not np.isfinite(a).all() or a.sum() <= 0:
        raise ValueError('Mixture weights must be a finite, nonzero 1-D vector.')
    return a / a.sum()


def validate_probabilities(p: np.ndarray, name: str = 'predictions') -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3 or min(p.shape) == 0:
        raise ValueError(f'{name} must have shape [samples, heads, classes].')
    if not np.isfinite(p).all() or p.min() < -1e-8 or p.max() > 1 + 1e-8:
        raise ValueError(f'{name} must contain finite probabilities in [0, 1].')
    return np.clip(p, 0, 1)


@dataclass(frozen=True)
class Event:
    modality: str   # 'image' or 'text'
    index: int
    class_index: int
    target: int = 1 # 0 means the event that this label is absent

    def coefficients(self, image: np.ndarray, text: np.ndarray) -> np.ndarray:
        if self.modality not in ('image', 'text') or self.target not in (0, 1):
            raise ValueError('Invalid event modality or binary target.')
        p = image if self.modality == 'image' else text
        if not 0 <= self.index < len(p) or not 0 <= self.class_index < p.shape[2]:
            raise IndexError('Event index or class_index is out of range.')
        v = np.asarray(p[self.index, :, self.class_index], dtype=np.float64)
        return v.copy() if self.target == 1 else 1.0 - v


def select_events(labels: np.ndarray, task: str, count: int = 8,
                  seed: int = 7) -> list[Event]:
    """Select fixed, observed-label events BEFORE feedback iterations.

    Uses only the risk-training split, never test labels. This is a design
    choice, not a new theorem. Do not change the event list during a run.
    """
    y = np.asarray(labels)
    if y.ndim != 2 or len(y) == 0 or count < 1:
        raise ValueError('Expected nonempty [N,K] labels and a positive count.')
    if task not in ('multiclass', 'multilabel'):
        raise ValueError('Unknown task.')
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    events = []
    for n in range(count):
        i = int(idx[(n // 2) % len(idx)])
        modality = 'image' if n % 2 == 0 else 'text'
        k = int(y[i].argmax()) if task == 'multiclass' else (n // 2) % y.shape[1]
        events.append(Event(modality, i, k, int(y[i, k])))
    return events


@dataclass
class RiskSystem:
    """R_j(a)=a^T G_j a - 2 h_j^T a + y2_j; j=(modality,class)."""
    gram: np.ndarray     # [J,M,M]
    linear: np.ndarray   # [J,M]
    constant: np.ndarray # [J]
    names: list[str]
    n_samples: int

    @classmethod
    def from_predictions(cls, image: np.ndarray, text: np.ndarray,
                         labels: np.ndarray) -> 'RiskSystem':
        image = validate_probabilities(image, 'image')
        text = validate_probabilities(text, 'text')
        if image.shape != text.shape:
            raise ValueError('Image/text probability banks must share [N,M,K].')
        n, m, k = image.shape
        y = np.asarray(labels, dtype=np.float64)
        if y.shape != (n, k) or not np.isin(y, [0.0, 1.0]).all():
            raise ValueError('Labels must be binary one-/multi-hot [N,K].')
        grams, linear, constant, names = [], [], [], []
        for modality, bank in [('image', image), ('text', text)]:
            for c in range(k):
                design = bank[:, :, c]
                grams.append(design.T @ design / n)
                linear.append(design.T @ y[:, c] / n)
                constant.append(float(np.mean(y[:, c] ** 2)))
                names.append(f'{modality}/class_{c}')
        return cls(np.array(grams), np.array(linear), np.array(constant), names, n)

    @property
    def n_heads(self) -> int:
        return self.linear.shape[1]

    @property
    def n_constraints(self) -> int:
        return len(self.constant)

    def risks(self, a: np.ndarray) -> np.ndarray:
        return (np.einsum('m,jmn,n->j', a, self.gram, a)
                - 2.0 * self.linear @ a + self.constant)

    def jacobian(self, a: np.ndarray) -> np.ndarray:
        return 2.0 * np.einsum('jmn,n->jm', self.gram, a) - 2.0 * self.linear

    def formal_kappa(self, delta: float = 0.05) -> float:
        """2*xi for a fixed bank independent of these iid risk samples.

        Binary labels ensure (p_m-y)(p_l-y) lies in [0,1]. A union bound over
        J*M^2 entries yields xi below. This is often conservative/vacuous.
        Truth inclusion additionally needs realizability in the mixture family.
        This function cannot verify independence or realizability for the user.
        """
        if not 0.0 < delta < 1.0:
            raise ValueError('delta must lie strictly between 0 and 1.')
        return float(2 * np.sqrt(np.log(2 * self.n_constraints * self.n_heads**2 / delta)
                                 / (2 * self.n_samples)))

    def save(self, path: str) -> None:
        np.savez_compressed(path, gram=self.gram, linear=self.linear,
                            constant=self.constant, names=np.array(self.names),
                            n_samples=self.n_samples)

    @classmethod
    def load(cls, path: str) -> 'RiskSystem':
        with np.load(path, allow_pickle=False) as d:
            return cls(d['gram'], d['linear'], d['constant'],
                       d['names'].tolist(), int(d['n_samples']))


@dataclass
class BoundResult:
    # min of the requested linear objective over the candidate set:
    primal: float       # an approximate feasible minimum, NOT itself a safe lower bound
    dual_lower: float   # convex tangent lower bound; no exact dual solve is assumed
    gap: float
    alpha: np.ndarray
    multipliers: np.ndarray
    max_violation: float
    message: str


class ConvexBounds:
    def __init__(self, risks: RiskSystem, *, feasibility_tol: float = 2e-8,
                 gap_tol: float = 2e-5, maxiter: int = 1200):
        self.risks = risks
        self.feasibility_tol = feasibility_tol
        self.gap_tol = gap_tol
        self.maxiter = maxiter

    def _constraints(self, thresholds: np.ndarray) -> list[dict]:
        m = self.risks.n_heads
        return [
            {'type': 'eq', 'fun': lambda a: a.sum() - 1.0,
             'jac': lambda a: np.ones(m)},
            {'type': 'ineq', 'fun': lambda a: thresholds - self.risks.risks(a),
             'jac': lambda a: -self.risks.jacobian(a)},
        ]

    def tangent_lower(self, c: np.ndarray, thresholds: np.ndarray,
                      a: np.ndarray, multipliers: np.ndarray) -> float:
        """Valid in exact arithmetic for ANY lambda>=0, not only optimal ones.

        F(a)=c*a+lambda*(R(a)-b) is convex. Over the simplex,
        min F >= F(a)+min(grad F)-grad F*a. Weak duality then lower-bounds
        min_{R<=b} c*a. A small floating point safety subtraction is included.
        """
        lam = np.maximum(np.asarray(multipliers, dtype=float), 0.0)
        f = float(c @ a + lam @ (self.risks.risks(a) - thresholds))
        grad = c + self.risks.jacobian(a).T @ lam
        value = float(f + grad.min() - grad @ a)
        safety = 2e-10 * (1.0 + abs(f) + np.linalg.norm(grad, 1))
        return value - safety

    def _recover_multipliers(self, c: np.ndarray, b: np.ndarray,
                             a: np.ndarray) -> np.ndarray:
        """Nonnegative least-squares KKT recovery, checked by the tangent gap."""
        active = np.flatnonzero(b - self.risks.risks(a) <= 2e-6)
        zeros = np.flatnonzero(a <= 2e-6)
        blocks = []
        if len(active):
            blocks.append(self.risks.jacobian(a)[active].T)
        if len(zeros):
            blocks.append(-np.eye(len(a))[:, zeros])
        blocks.append(np.ones((len(a), 1)))  # free equality multiplier
        matrix = np.concatenate(blocks, axis=1)
        lower = np.r_[np.zeros(len(active) + len(zeros)), -np.inf]
        upper = np.full(matrix.shape[1], np.inf)
        solution = lsq_linear(matrix, -c, bounds=(lower, upper),
                              tol=1e-13, max_iter=1500)
        lam = np.zeros(self.risks.n_constraints)
        lam[active] = np.maximum(solution.x[:len(active)], 0)
        return lam

    def minimize_linear(self, coefficients: np.ndarray, thresholds: np.ndarray,
                        witness: np.ndarray) -> BoundResult:
        # SLSQP expects a contiguous gradient buffer; embedding slices are often strided.
        c = np.array(coefficients, dtype=np.float64, copy=True, order='C')
        b = np.array(thresholds, dtype=np.float64, copy=True, order='C')
        a0 = _simplex(witness)
        if c.shape != (self.risks.n_heads,) or b.shape != (self.risks.n_constraints,):
            raise ValueError('Incorrect coefficient/threshold shape.')
        if not np.isfinite(c).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite bound inputs.')
        if np.max(self.risks.risks(a0) - b) > self.feasibility_tol:
            raise OptimizationError('Supplied witness is not feasible.')
        result = minimize(lambda a: float(c @ a), a0,
                          jac=lambda a: c, method='SLSQP',
                          bounds=[(0, 1)] * len(a0),
                          constraints=self._constraints(b),
                          options={'maxiter': self.maxiter, 'ftol': 1e-12})
        a = _simplex(result.x)
        violation = float(max(0, np.max(self.risks.risks(a) - b)))
        candidates = [np.zeros(self.risks.n_constraints)]
        if hasattr(result, 'multipliers'):
            candidates.append(np.maximum(np.asarray(result.multipliers[1:1+len(b)]), 0))
        candidates.append(self._recover_multipliers(c, b, a))
        scores = [self.tangent_lower(c, b, a, lam) for lam in candidates]
        best = int(np.argmax(scores))
        lower = float(scores[best])
        primal = float(c @ a)
        gap = primal - lower
        if (violation > self.feasibility_tol or gap > self.gap_tol
                or gap < -self.gap_tol or not np.isfinite(gap)):
            raise OptimizationError(
                f'Bound solve rejected: violation={violation:.3g}, gap={gap:.3g}; '
                f'{result.message}. Increase tolerance only after inspecting residuals.')
        return BoundResult(primal, lower, max(0.0, gap), a,
                           candidates[best], violation, str(result.message))


class BoundFeedback:
    """Core branch: update the shared mixture weights, not the fixed heads."""
    def __init__(self, risks: RiskSystem, kappa: float | np.ndarray,
                 alpha: np.ndarray | None = None, *, eta: float = 0.5,
                 tolerance_mode: str = 'empirical', feasibility_tol: float = 2e-8,
                 gap_tol: float = 2e-5):
        if eta <= 0 or tolerance_mode not in ('empirical', 'formal'):
            raise ValueError('eta>0 and an explicit tolerance_mode are required.')
        self.risk_system = risks
        self.kappa = np.broadcast_to(np.asarray(kappa, dtype=float),
                                     (risks.n_constraints,)).copy()
        if not np.isfinite(self.kappa).all() or np.any(self.kappa <= 0):
            raise ValueError('Every kappa must be positive and finite.')
        self.alpha = _simplex(np.ones(risks.n_heads) if alpha is None else alpha)
        if self.alpha.shape != (risks.n_heads,):
            raise ValueError('Incorrect number of mixture weights.')
        self.eta = eta
        self.tolerance_mode = tolerance_mode
        self.solver = ConvexBounds(risks, feasibility_tol=feasibility_tol,
                                   gap_tol=gap_tol)
        self.history: list[dict] = []

    @property
    def thresholds(self) -> np.ndarray:
        return self.risk_system.risks(self.alpha) + self.kappa

    def predict(self, probability_bank: np.ndarray) -> np.ndarray:
        bank = validate_probabilities(probability_bank)
        if bank.shape[1] != len(self.alpha):
            raise ValueError('The bank changed: construct a new candidate problem.')
        return np.einsum('nmk,m->nk', bank, self.alpha)

    def interval(self, coefficients: np.ndarray) -> dict:
        """Input coefficients are one [M] event probability vector.

        Lower/upper use dual-side bounds, not unchecked primal solver values.
        Endpoints are coordinatewise envelopes; all lower endpoints generally
        do not form one jointly attainable class probability vector.
        """
        c = np.asarray(coefficients, dtype=float)
        if c.shape != self.alpha.shape or np.any(c < 0) or np.any(c > 1):
            raise ValueError('interval expects an [M] probability vector.')
        lo = self.solver.minimize_linear(c, self.thresholds, self.alpha)
        neg_hi = self.solver.minimize_linear(-c, self.thresholds, self.alpha)
        return {
            'probability': float(c @ self.alpha),
            'lower': float(max(0.0, lo.dual_lower)),
            'upper': float(min(1.0, -neg_hi.dual_lower)),
            'lower_primal': lo.primal,
            'upper_primal': -neg_hi.primal,
            'max_solver_gap': max(lo.gap, neg_hi.gap),
            'max_violation': max(lo.max_violation, neg_hi.max_violation),
            'scope': 'fixed_candidate_family',
            'tolerance_mode': self.tolerance_mode,
        }

    def step(self, event_coefficients: np.ndarray,
             event_weights: np.ndarray | None = None) -> dict:
        """One checked lower-bound -> dual-feedback -> proximal update.

        Fix event_coefficients and event_weights across rounds. Calls can stop
        with zero improvement; no artificial minimum gain or kappa decay exists.
        """
        events = np.asarray(event_coefficients, dtype=float)
        if (events.ndim != 2 or events.shape[1] != len(self.alpha)
                or len(events) == 0 or not np.isfinite(events).all()
                or np.any(events < 0) or np.any(events > 1)):
            raise ValueError('event_coefficients must be finite [events,M] probabilities.')
        omega = _simplex(np.ones(len(events)) if event_weights is None else event_weights)
        if omega.shape != (len(events),):
            raise ValueError('Incorrect event_weights shape.')
        old = self.alpha.copy()
        old_risk = self.risk_system.risks(old)
        old_b = old_risk + self.kappa
        before = [self.solver.minimize_linear(c, old_b, old) for c in events]
        w = sum(float(o) * r.multipliers for o, r in zip(omega, before))
        old_primal = float(omega @ np.array([r.primal for r in before]))
        old_dual = float(omega @ np.array([r.dual_lower for r in before]))

        def fun(a: np.ndarray) -> float:
            return float(w @ self.risk_system.risks(a)
                         + np.sum((a - old)**2) / (2 * self.eta))

        def jac(a: np.ndarray) -> np.ndarray:
            return self.risk_system.jacobian(a).T @ w + (a - old) / self.eta

        if np.max(w) < 1e-10:
            proposal = old.copy()
            solve_message = 'No active lower-bound feedback; unchanged.'
        else:
            result = minimize(fun, old, jac=jac, method='SLSQP',
                              bounds=[(0, 1)] * len(old),
                              constraints=self.solver._constraints(old_risk),
                              options={'ftol': 1e-12, 'maxiter': 1200})
            proposal = _simplex(result.x)
            solve_message = str(result.message)

        # Accept only feasible improvement of the actual fixed objective.
        # Backtracking is for numerical safety, not a replacement for the QCQP.
        accepted = False
        new = old.copy()
        verification_tol = min(2e-9, self.solver.feasibility_tol)
        fraction = 1.0
        for _ in range(28):
            trial = _simplex(old + fraction * (proposal - old))
            if (np.max(self.risk_system.risks(trial) - old_risk) <= verification_tol
                    and fun(trial) <= fun(old) + verification_tol):
                new = trial
                accepted = True
                break
            fraction *= 0.5
        new_risk = self.risk_system.risks(new)
        new_b = new_risk + self.kappa
        after = [self.solver.minimize_linear(c, new_b, new) for c in events]
        new_primal = float(omega @ np.array([r.primal for r in after]))
        new_dual = float(omega @ np.array([r.dual_lower for r in after]))
        delta = old_risk - new_risk
        dual_gain = float(w @ delta)
        prox_floor = float(np.sum((new - old)**2) / (2 * self.eta))
        # With approximate dual solves the sensitivity statement has a gap term.
        numerical_budget = 4 * self.solver.gap_tol + 1e-8
        if (new_primal - old_primal + numerical_budget < dual_gain
                or dual_gain + numerical_budget < prox_floor
                or np.max(new_risk - old_risk) > verification_tol * 2):
            raise OptimizationError('Update failed the gain-chain or risk checks.')
        # Commit only after all new bound solves and checks passed.
        self.alpha = new
        record = {
            'round': len(self.history) + 1,
            'accepted': accepted,
            'changed': bool(np.linalg.norm(new-old) > 1e-9),
            'old_alpha': old.tolist(), 'new_alpha': new.tolist(),
            'old_risks': old_risk.tolist(), 'new_risks': new_risk.tolist(),
            'feedback_weights': w.tolist(),
            'mean_lower_primal_before': old_primal,
            'mean_lower_primal_after': new_primal,
            'mean_lower_dual_before': old_dual,
            'mean_lower_dual_after': new_dual,
            'actual_gain_estimate': new_primal - old_primal,
            'feedback_gain_estimate': dual_gain,
            'proximal_gain_floor': prox_floor,
            'old_bound_solver_gap': old_primal - old_dual,
            'new_bound_solver_gap': new_primal - new_dual,
            'gain_lower_with_old_solver_error': dual_gain - (old_primal-old_dual),
            'max_risk_increase': float(np.max(new_risk-old_risk)),
            'message': solve_message,
        }
        self.history.append(record)
        return record

    def save(self, path: str) -> None:
        r = self.risk_system
        np.savez_compressed(path, gram=r.gram, linear=r.linear, constant=r.constant,
                            names=np.array(r.names), n_samples=r.n_samples,
                            alpha=self.alpha, kappa=self.kappa, eta=self.eta,
                            tolerance_mode=self.tolerance_mode)

    @classmethod
    def load(cls, path: str) -> 'BoundFeedback':
        with np.load(path, allow_pickle=False) as d:
            risk = RiskSystem(d['gram'], d['linear'], d['constant'],
                              d['names'].tolist(), int(d['n_samples']))
            return cls(risk, d['kappa'], d['alpha'], eta=float(d['eta']),
                       tolerance_mode=str(d['tolerance_mode']))
