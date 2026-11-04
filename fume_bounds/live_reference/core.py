"""Current-encoder candidate family and its full envelope-gradient surrogate.

This changes the reference objective. It is NOT a drop-in compatibility claim
for an old epoch-80 prediction cache, nor a population-confidence guarantee.
Inner convex solves are detached; outer torch computations are differentiable.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Sequence
import numpy as np
import torch
from .convex_base import Event, RiskSystem, ConvexBounds, BoundResult


def _frozen(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


def array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _simplex(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'{name}: expected finite nonnegative 1-D weights')
    if not np.isclose(a.sum(), 1., rtol=0, atol=1e-10):
        raise ValueError(f'{name}: weights must already sum to one')
    return _frozen(a)


@dataclass(frozen=True)
class LiveSnapshot:
    results: tuple[BoundResult, ...]
    event_weights: np.ndarray
    anchor_risks: np.ndarray
    thresholds: np.ndarray
    witness_slack: np.ndarray
    worst_alpha: np.ndarray       # [E,M]
    dual_weights: np.ndarray      # [E,2K], raw, not normalized across constraints
    objective_lower: float
    objective_upper: float
    objective_id: str
    risk_id: str
    bank_hash: str
    max_gap: float

    def reference_solve(self):
        """Adapter for the previously delivered v3 controller (separate package)."""
        from fume_control import ReferenceSolve
        return ReferenceSolve(
            bound_results=self.results,
            event_weights=self.event_weights,
            risks=self.anchor_risks,
            objective_id=self.objective_id,
            risk_id=self.risk_id,
            feasibility_tol=2e-8,
            max_gap=self.max_gap,
            numerical_padding=1e-8,
        )


class LiveReference:
    """Fixed heads/risk rows/events, CURRENT encoder predictions for all candidates.

    p_theta(alpha) = sum_m alpha_m * head_m(encoder_theta(x)).
    The actual predictor MUST be the fixed anchor_alpha mixture. For a different
    deployed prediction rule, do not use this problem without redefining it.
    """
    def __init__(self, labels: np.ndarray, *, anchor_alpha: Sequence[float],
                 events: Sequence[Event], kappa: float | np.ndarray,
                 sample_ids: Sequence[str], heads_fingerprint: str,
                 event_weights: Sequence[float] | None = None,
                 task: str = 'multiclass', gap_tol: float = 2e-5):
        y = np.asarray(labels, dtype=float)
        if y.ndim != 2 or not all(y.shape) or not np.isin(y, [0., 1.]).all():
            raise ValueError('Expected nonempty hard binary labels [N,K]')
        if task not in ('multiclass', 'multilabel'):
            raise ValueError('Unknown task')
        if task == 'multiclass' and not np.all(y.sum(1) == 1):
            raise ValueError('Multiclass labels must be one-hot')
        if len(sample_ids) != len(y) or len(set(sample_ids)) != len(y):
            raise ValueError('Risk IDs must identify all distinct rows, in fixed order')
        if not heads_fingerprint:
            raise ValueError('An immutable heads/checkpoint fingerprint is required')
        self.labels = _frozen(y)
        self.anchor_alpha = _simplex(anchor_alpha, 'anchor_alpha')
        self.events = tuple(events)
        if not self.events:
            raise ValueError('Need a fixed nonempty event set')
        self.event_weights = _simplex(
            np.full(len(events), 1/len(events)) if event_weights is None else event_weights,
            'event_weights')
        if len(self.event_weights) != len(self.events):
            raise ValueError('Event weight length mismatch')
        self.kappa = _frozen(np.broadcast_to(np.asarray(kappa, dtype=float), (2*y.shape[1],)))
        if not np.isfinite(self.kappa).all() or np.min(self.kappa) <= 0:
            raise ValueError('Positive finite, fixed kappa required')
        if not np.isfinite(gap_tol) or gap_tol <= 0:
            raise ValueError('gap_tol must be positive')
        self.task, self.gap_tol = task, float(gap_tol)
        n, k = y.shape
        for e in self.events:
            if (e.modality not in ('image','text') or e.target not in (0,1)
                    or not 0 <= e.index < n or not 0 <= e.class_index < k):
                raise ValueError('Invalid fixed event')
        spec = dict(version='live_reference_v4', task=task, labels_hash=array_hash(y),
                    sample_ids=list(sample_ids), heads_fingerprint=heads_fingerprint,
                    anchor_alpha=self.anchor_alpha.tolist(), kappa=self.kappa.tolist(),
                    events=[vars(e) for e in events], weights=self.event_weights.tolist(),
                    risk='per_modality_per_class_mean_squared_probability_error',
                    candidate_rule='fixed_heads_on_current_encoder_convex_mixtures')
        blob = json.dumps(spec, sort_keys=True, allow_nan=False).encode()
        self.objective_id = hashlib.sha256(blob).hexdigest()
        self.risk_id = hashlib.sha256(json.dumps(
            {k:spec[k] for k in ('sample_ids','labels_hash','risk','anchor_alpha')},
            sort_keys=True).encode()).hexdigest()
        self.protocol = spec

    def _validate_banks(self, image, text):
        expected = (len(self.labels), len(self.anchor_alpha), self.labels.shape[1])
        for name, bank in [('image', image), ('text', text)]:
            bank = np.asarray(bank)
            if bank.shape != expected or not np.isfinite(bank).all():
                raise ValueError(f'{name}: expected finite bank of shape {expected}')
            if (bank < -1e-8).any() or (bank > 1+1e-8).any():
                raise ValueError('Probabilities outside [0,1]')
            if self.task == 'multiclass' and not np.allclose(bank.sum(2), 1, atol=2e-6):
                raise ValueError('Multiclass probabilities must sum to one')

    def solve(self, image: np.ndarray, text: np.ndarray, *,
              reported_current_risks: np.ndarray | None = None) -> LiveSnapshot:
        """Rebuild BOTH risks and event coefficients from the same current banks.

        Call under no_grad for extraction. Never combine epoch-80 risks/events
        with current thresholds. No state (including kappa) is changed here.
        """
        self._validate_banks(image, text)
        risks = RiskSystem.from_predictions(image, text, self.labels)
        anchor = risks.risks(self.anchor_alpha)
        if reported_current_risks is not None:
            rr = np.asarray(reported_current_risks)
            if rr.shape != anchor.shape or not np.allclose(rr, anchor, atol=1e-9, rtol=1e-6):
                raise ValueError('Current predictor/risk reduction differs from anchor mixture')
        b = anchor + self.kappa
        slack = b - risks.risks(self.anchor_alpha)
        if not np.allclose(slack, self.kappa, rtol=1e-10, atol=1e-12):
            raise RuntimeError('Anchor feasibility invariant failed')
        solver = ConvexBounds(risks, gap_tol=self.gap_tol)
        result = tuple(solver.minimize_linear(e.coefficients(image, text), b,
                                             self.anchor_alpha) for e in self.events)
        w = self.event_weights
        return LiveSnapshot(
            result, w, _frozen(anchor), _frozen(b), _frozen(slack),
            _frozen([r.alpha for r in result]), _frozen([r.multipliers for r in result]),
            float(w @ [r.dual_lower for r in result]), float(w @ [r.primal for r in result]),
            self.objective_id, self.risk_id, array_hash(image, text), self.gap_tol)

    def _torch_constants(self, snapshot, like):
        if snapshot.objective_id != self.objective_id:
            raise ValueError('Snapshot belongs to a different method/event/kappa protocol')
        kw = dict(device=like.device, dtype=like.dtype)
        return (torch.tensor(snapshot.worst_alpha.copy(), **kw).detach(),
                torch.tensor(snapshot.dual_weights.copy(), **kw).detach(),
                torch.tensor(self.anchor_alpha.copy(), **kw).detach(),
                torch.tensor(self.event_weights.copy(), **kw).detach())

    def loss(self, image: torch.Tensor, text: torch.Tensor, snapshot: LiveSnapshot,
             *, verify_snapshot_values: bool = True) -> torch.Tensor:
        """Negative full Lagrangian envelope surrogate, not the old w*R loss.

        At the solved point, under the envelope regularity conditions,
        grad B = grad c*a* + lambda*(grad R(a*) - grad R(a0)).
        a*, lambda are constants in autograd. No inner-solver differentiation.
        For fresh-gradient tests set verify_snapshot_values=True. A short segment
        may use a stale surrogate with False, but its new endpoint MUST be fully
        re-solved. No stale gradient is claimed to be exact at later steps.
        """
        expected = (len(self.labels), len(self.anchor_alpha), self.labels.shape[1])
        if image.shape != expected or text.shape != expected:
            raise ValueError('Differentiable loss needs ALL ordered risk rows')
        if image.device != text.device or image.dtype != text.dtype:
            raise ValueError('Paired tensors must share device/dtype')
        if not image.requires_grad and not text.requires_grad:
            raise ValueError('Detached probability banks cannot train the encoders')
        if verify_snapshot_values:
            i = image.detach().cpu().numpy(); t = text.detach().cpu().numpy()
            # solve converts only inside RiskSystem; snapshot hash is raw caller dtype.
            if array_hash(i,t) != snapshot.bank_hash:
                raise ValueError('Gradient point differs from solve point: recompute snapshot')
        # Match the float64 inner risk arithmetic; casts retain the encoder graph.
        image, text = image.to(torch.float64), text.to(torch.float64)
        a, lam, a0, w = self._torch_constants(snapshot, image)
        y = torch.tensor(self.labels.copy(), device=image.device, dtype=image.dtype)
        wc_risks, center_risks = [], []
        for bank in (image, text):
            qc = torch.einsum('nmk,m->nk', bank, a0)
            qw = torch.einsum('nmk,em->nek', bank, a)
            center_risks.append(((qc-y)**2).mean(0))
            wc_risks.append(((qw-y[:,None,:])**2).mean(0))
        center = torch.cat(center_risks)                       # [2K]
        worst = torch.cat(wc_risks, dim=1)                    # [E,2K]
        scores = []
        for e, aa in zip(self.events, a):
            bank = image if e.modality == 'image' else text
            score = bank[e.index,:,e.class_index] @ aa
            scores.append(score if e.target else 1-score)
        scores = torch.stack(scores)
        kappa = torch.tensor(self.kappa.copy(), device=image.device, dtype=image.dtype)
        lagrangian = scores + (lam*(worst-center[None,:]-kappa[None,:])).sum(1)
        return -(w*lagrangian).sum()

    def coefficients(self, image: torch.Tensor, text: torch.Tensor,
                     snapshot: LiveSnapshot) -> tuple[torch.Tensor, torch.Tensor]:
        """d(-B_surrogate)/d(bank) at the solved point, detached CPU/GPU tensors.

        These allow an EXACT first-order replay in microbatches: compute full
        probability-bank gradients, then backprop sum(p_chunk*g_chunk). Do NOT
        take an optimizer step until all rows have been accumulated. This avoids
        retaining a full large-encoder graph for 2700 rows.
        """
        i = image.detach().clone().requires_grad_(True)
        t = text.detach().clone().requires_grad_(True)
        value = self.loss(i,t,snapshot)
        gi,gt = torch.autograd.grad(value, (i,t))
        return gi.detach(),gt.detach()

    def risk_barriers(self, image: torch.Tensor, text: torch.Tensor):
        """Current anchor's per-modality/per-class errors (for audit, no filtering)."""
        a0 = torch.tensor(self.anchor_alpha.copy(), device=image.device, dtype=image.dtype)
        y = torch.tensor(self.labels.copy(), device=image.device, dtype=image.dtype)
        return torch.cat([((torch.einsum('nmk,m->nk',p,a0)-y)**2).mean(0)
                          for p in (image,text)])
