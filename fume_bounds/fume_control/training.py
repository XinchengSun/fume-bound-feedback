"""Replay a short segment with actual optimizer updates, not post-hoc interpolation."""
from __future__ import annotations
from typing import Callable, Sequence, Any
import torch
from .controller import AttemptSpec, ProposalInfo, InvalidCandidate
from .state import digest


class TorchSegment:
    """Materialize/reuse only a short segment of batches (e.g. 4, not 30+).

    loss_fn(batch) should call the existing encoder loss and feedback weighting;
    this wrapper neither invents a new loss nor changes kappa/risk budgets.
    Batches must be read-only. For random augmentation, materialize its outputs
    first, or explicitly snapshot its generators and recreate the same plan.
    A mixed-precision custom callback may be substituted; register its scaler
    in controller.stateful and report only actual optimizer steps.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 batches: Sequence[Any], loss_fn: Callable[[Any], torch.Tensor], *,
                 old_dual_hint: float | None = None,
                 set_training_mode: Callable[[], None] | None = None):
        if not batches:
            raise ValueError('A nonempty fixed batch sequence is required.')
        self.model, self.optimizer = model, optimizer
        self.batches, self.loss_fn = tuple(batches), loss_fn
        self.old_dual_hint = old_dual_hint
        self.set_training_mode = set_training_mode
        self.batch_digest = digest(self.batches)

    def __call__(self, spec: AttemptSpec) -> ProposalInfo:
        if len(self.batches) != spec.optimizer_steps:
            raise ValueError('Replay segment length differs from requested steps.')
        if digest(self.batches) != self.batch_digest:
            raise ValueError('Replay batches were modified between attempts.')
        if self.set_training_mode is None:
            self.model.train()
        else:
            self.set_training_mode()
        losses = []
        for batch in self.batches:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(batch)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise InvalidCandidate('Nonfinite or nonscalar training loss.')
            loss.backward()
            any_gradient = False
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        any_gradient = True
                        if not bool(torch.isfinite(p.grad).all()):
                            raise InvalidCandidate('Nonfinite gradient.')
            if not any_gradient:
                raise InvalidCandidate('Loss did not reach any optimizer parameter.')
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return ProposalInfo(self.old_dual_hint, {'losses': losses, 'steps': len(losses),
                                                'replay_batch_sha256': self.batch_digest})
