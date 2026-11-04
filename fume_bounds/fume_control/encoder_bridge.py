"""Adapter to the existing encoder loss and existing reference-bound solver.

This module does not redefine the family, risk budgets, kappa or objective.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, Any
import torch
from .acceptance import Evaluation, RiskGuard, bracket_from_bound_results
from .controller import TransactionalController, StepReport
from .training import TorchSegment


@dataclass
class ReferenceSolve:
    bound_results: Sequence[Any]  # each exposes primal, dual_lower, max_violation
    event_weights: Sequence[float]
    risks: Sequence[float]        # ALL fixed reference risk components in fixed order
    objective_id: str
    risk_id: str
    feasibility_tol: float = 2e-8
    max_gap: float = 2e-5
    numerical_padding: float = 1e-8

    def evaluation(self) -> Evaluation:
        bracket = bracket_from_bound_results(
            self.bound_results, self.event_weights, objective_id=self.objective_id,
            feasibility_tol=self.feasibility_tol, max_gap=self.max_gap,
            numerical_padding=self.numerical_padding)
        result = Evaluation(bracket, tuple(float(x) for x in self.risks), self.risk_id,
                            {'n_events':len(self.bound_results),
                             'scope':'numerical_reference_family_only'})
        result.validate()
        return result


def run_encoder_segment(controller: TransactionalController, *,
                        replay_batches: Sequence[Any],
                        current_loss: Callable[[Any], torch.Tensor],
                        solve_current_reference: Callable[[], ReferenceSolve],
                        risk_guard: RiskGuard,
                        round_id: str,
                        old_dual_hint: float | None = None,
                        set_training_mode: Callable[[], None] | None = None) -> StepReport:
    """Use in place of the old propose/interpolate/accept/rollback block.

    current_loss uses the current encoder and existing feedback loss. The
    reference callback must extract embeddings from the CURRENT encoder in
    eval mode and re-solve the same reference problem recipe, including old
    point baseline. Caches keyed only by epoch number are not sufficient.
    A solve that fails known feasibility/gap checks should raise InvalidCandidate.
    Programming errors should not be swallowed.
    """
    proposer = TorchSegment(controller.model, controller.optimizer,
                            replay_batches, current_loss, old_dual_hint=old_dual_hint,
                            set_training_mode=set_training_mode)
    def evaluate(): return solve_current_reference().evaluation()
    return controller.run_segment(proposer, evaluate, risk_guard,
                                  optimizer_steps=len(replay_batches), round_id=round_id)
