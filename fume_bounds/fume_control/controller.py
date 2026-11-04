"""A bounded, transactional retry controller for an existing encoder trainer.

No parameter interpolation. No second stale-certificate veto. No implicit
retraining of FUME. Both model and optimizer follow the same accepted trajectory.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np
import torch
from .acceptance import Evaluation, RiskGuard, Decision, decide
from .state import (Snapshot, RNGState, digest, fume_plain_tensors,
                    model_payload, parameter_stats)


class ControllerStopped(RuntimeError): pass
class ContractError(RuntimeError): pass
class InvalidCandidate(RuntimeError):
    """A recoverable candidate evaluation/forward failure, not a programming error."""


@dataclass(frozen=True)
class ControlConfig:
    max_attempts: int = 8
    shrink: float = 0.5
    min_lr: float = 1e-12
    min_gain: float = 1e-8
    parameter_atol: float = 1e-12
    parameter_rtol: float = 1e-10
    check_evaluator_purity: bool = True

    def validate(self):
        if self.max_attempts < 1 or not 0 < self.shrink < 1:
            raise ValueError('Require max_attempts>=1 and 0<shrink<1.')
        for x in (self.min_lr, self.min_gain, self.parameter_atol, self.parameter_rtol):
            if not math.isfinite(x) or x < 0:
                raise ValueError('Invalid control tolerance.')
        if self.min_lr == 0:
            raise ValueError('min_lr must be positive.')


@dataclass(frozen=True)
class AttemptSpec:
    index: int
    lr_scale: float
    learning_rates: tuple[float, ...]
    optimizer_steps: int
    round_id: str


@dataclass(frozen=True)
class ProposalInfo:
    # It is diagnostic ONLY. In particular a negative value is not a veto.
    old_dual_hint: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class StepReport:
    status: str
    round_id: str
    attempts: list[dict]
    attempted_optimizer_steps: int
    committed_optimizer_steps: int
    decision: Decision | None
    effective_delta_l2: float = 0.0
    stop_reason: str | None = None

    @property
    def accepted(self): return self.status == 'accepted'

    def as_dict(self): return asdict(self)


def _finite_tree(value):
    if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray): return bool(np.isfinite(value).all())
    if isinstance(value, dict): return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)): return all(_finite_tree(v) for v in value)
    if isinstance(value, float): return math.isfinite(value)
    return True


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x): return str(x)
    if isinstance(x, dict): return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [_json_safe(v) for v in x]
    if isinstance(x, np.generic): return _json_safe(x.item())
    return x


class TransactionalController:
    """One process, one optimizer, one root Module containing all trainable heads.

    evaluate() must recompute the full reference objective at the CURRENT model
    and return a checked lower/upper enclosure plus ordered risk components.
    It may not use test mAP. Its identity/recipe must remain fixed.

    propose(spec) must replay a fixed list of mini-batches from its beginning,
    call optimizer.step() exactly spec.optimizer_steps times, and leave the
    actual candidate model AND optimizer in place. It must NOT interpolate
    parameters afterwards or overwrite learning rates. On rejection everything
    is restored, then the new trial learning rate is set AFTER restore.

    Add scheduler/scaler/explicit generator state through stateful. Register
    all ordinary mutable Tensor attributes through plain_tensors (FUME.W is
    discovered automatically). Stateful DataLoader iterators are not supported:
    replay a materialized short segment or a fixed index plan, num_workers=0.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 config: ControlConfig | None = None, *,
                 stateful: Mapping[str, Any] | None = None,
                 plain_tensors: Mapping[str, torch.Tensor] | None = None,
                 log_path: str | Path | None = None):
        self.model, self.optimizer = model, optimizer
        self.config = config or ControlConfig(); self.config.validate()
        self.stateful = dict(stateful or {})
        self.plain = fume_plain_tensors(model)
        self.plain.update(plain_tensors or {})
        if any(not hasattr(v, 'state_dict') or not hasattr(v, 'load_state_dict')
               for v in self.stateful.values()):
            raise TypeError('stateful extras require state_dict/load_state_dict.')
        self.log_path = None if log_path is None else Path(log_path)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        model_params = {id(p) for p in model.parameters()}
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        opt_params = {id(p) for g in optimizer.param_groups for p in g['params']}
        if not trainable <= opt_params or not opt_params <= model_params:
            raise ContractError('The root model must include all optimizer parameters and trainable heads.')
        self.stopped = False
        self.stop_reason = None
        self.accepted_segments = 0
        self.total_attempted_optimizer_steps = 0
        self.total_committed_optimizer_steps = 0

    def state_dict(self):
        return {'format': 'fume_transactional_control_v3',
                'stopped': self.stopped, 'stop_reason': self.stop_reason,
                'accepted_segments': self.accepted_segments,
                'total_attempted_optimizer_steps': self.total_attempted_optimizer_steps,
                'total_committed_optimizer_steps': self.total_committed_optimizer_steps,
                'config': asdict(self.config)}

    def load_state_dict(self, state):
        if state.get('format') != 'fume_transactional_control_v3' or state['config'] != asdict(self.config):
            raise ValueError('Controller checkpoint/config mismatch.')
        for key in ('stopped', 'stop_reason', 'accepted_segments',
                    'total_attempted_optimizer_steps', 'total_committed_optimizer_steps'):
            setattr(self, key, state[key])

    def _log(self, row):
        if self.log_path is not None:
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()

    def _evaluate(self, evaluate: Callable[[], Evaluation]) -> Evaluation:
        rng = RNGState.capture()
        modes = {n: m.training for n, m in self.model.named_modules()}
        before = None
        if self.config.check_evaluator_purity:
            before = digest({'model': model_payload(self.model, self.plain),
                             'optimizer': self.optimizer.state_dict(),
                             'extras': {n: x.state_dict() for n, x in self.stateful.items()}})
        try:
            self.model.eval()
            with torch.no_grad():
                evaluation = evaluate()
        finally:
            for name, m in self.model.named_modules(): m.training = modes[name]
            rng.restore()
        if self.config.check_evaluator_purity:
            after = digest({'model': model_payload(self.model, self.plain),
                            'optimizer': self.optimizer.state_dict(),
                            'extras': {n: x.state_dict() for n, x in self.stateful.items()}})
            if before != after:
                raise ContractError('evaluate() changed model/optimizer/buffers/extra state.')
        if not isinstance(evaluation, Evaluation):
            raise TypeError('evaluate() must return Evaluation.')
        evaluation.validate()
        return evaluation

    def run_segment(self, propose: Callable[[AttemptSpec], ProposalInfo | None],
                    evaluate: Callable[[], Evaluation], guard: RiskGuard, *,
                    optimizer_steps: int = 4, round_id: str = 'segment') -> StepReport:
        if self.stopped:
            raise ControllerStopped('Previous segment failed. Do not repeat it unchanged: '
                                    + str(self.stop_reason))
        if optimizer_steps < 1:
            raise ValueError('optimizer_steps must be positive.')
        base = Snapshot.capture(self.model, self.optimizer, self.stateful, self.plain)
        base_lrs = tuple(float(g['lr']) for g in self.optimizer.param_groups)
        if not base_lrs or any(not math.isfinite(x) or x <= 0 for x in base_lrs):
            raise ValueError('Positive finite optimizer learning rates required.')
        attempts, total_steps = [], 0
        previous_candidate_hash = None
        committed = False
        stop_reason = 'retry_budget_exhausted'
        final_decision = None
        try:
            old = self._evaluate(evaluate)
            guard.arrays(len(old.risks))
            for retry in range(self.config.max_attempts):
                # Crucial order: restore FIRST; set the new learning rate AFTER.
                base.restore(self.model, self.optimizer, self.stateful, self.plain)
                scale = self.config.shrink ** retry
                trial_lrs = tuple(lr * scale for lr in base_lrs)
                if min(trial_lrs) < self.config.min_lr:
                    stop_reason = 'minimum_learning_rate_reached'; break
                for group, lr in zip(self.optimizer.param_groups, trial_lrs):
                    group['lr'] = lr
                spec = AttemptSpec(retry, scale, trial_lrs, optimizer_steps, str(round_id))
                steps = [0]
                last_step_model_hash = [None]
                last_step_optimizer_hash = [None]
                def pre_hook(opt, args, kwargs):
                    current = tuple(float(g['lr']) for g in opt.param_groups)
                    if current != trial_lrs:
                        raise ContractError('Trial learning rate overwritten inside propose(). '
                                            'Do not reload an optimizer or step a scheduler here.')
                def post_hook(opt, args, kwargs):
                    steps[0] += 1
                    if steps[0] == optimizer_steps:
                        # Detect the precise old failure: parameter interpolation
                        # after a real multi-step Adam trajectory was computed.
                        last_step_model_hash[0] = digest({
                            'parameters': dict(self.model.named_parameters()), 'plain': self.plain})
                        last_step_optimizer_hash[0] = digest(opt.state_dict())
                pre_handle = self.optimizer.register_step_pre_hook(pre_hook)
                post_handle = self.optimizer.register_step_post_hook(post_hook)
                row = {'round_id': str(round_id), 'attempt': retry,
                       'lr_scale': scale, 'learning_rates': list(trial_lrs)}
                recoverable = None
                try:
                    info = propose(spec) or ProposalInfo()
                except InvalidCandidate as exc:
                    info = ProposalInfo(); recoverable = str(exc)
                finally:
                    pre_handle.remove(); post_handle.remove()
                    total_steps += steps[0]
                    self.total_attempted_optimizer_steps += steps[0]
                row['attempted_optimizer_steps'] = steps[0]
                if recoverable is not None:
                    row.update(accepted=False, reasons=['invalid_candidate'], message=recoverable)
                    attempts.append(row); self._log(row)
                    continue
                if not isinstance(info, ProposalInfo):
                    raise ContractError('propose() must return ProposalInfo or None.')
                if steps[0] != optimizer_steps:
                    raise ContractError(f'Expected {optimizer_steps} actual optimizer steps, got {steps[0]}.')
                current_step_hash = digest({
                    'parameters': dict(self.model.named_parameters()), 'plain': self.plain})
                if (current_step_hash != last_step_model_hash[0]
                        or digest(self.optimizer.state_dict()) != last_step_optimizer_hash[0]):
                    raise ContractError('Model or optimizer changed after the last actual optimizer step. '
                                        'Post-hoc parameter interpolation / optimizer replacement is forbidden.')
                if tuple(id(p) for p in self.model.parameters()) != base.parameter_ids:
                    raise ContractError('Parameter identities changed during the proposal.')
                stats = parameter_stats(self.model, base)
                row.update(stats)
                row['old_dual_hint'] = info.old_dual_hint
                row['proposal_details'] = dict(info.details)
                row['candidate_hash'] = digest(model_payload(self.model, self.plain))
                if (not stats['finite'] or not _finite_tree(self.optimizer.state_dict())
                        or not _finite_tree(model_payload(self.model, self.plain))):
                    row.update(accepted=False, reasons=['nonfinite_candidate_or_optimizer'])
                    attempts.append(row); self._log(row)
                    continue
                effective_floor = self.config.parameter_atol + self.config.parameter_rtol * stats['base_l2']
                if (stats['changed_coordinates'] == 0 or stats['delta_l2'] <= effective_floor):
                    stop_reason = 'no_effective_parameter_update'
                    row.update(accepted=False, reasons=[stop_reason], effective_floor=effective_floor)
                    attempts.append(row); self._log(row); break
                if row['candidate_hash'] == previous_candidate_hash:
                    stop_reason = 'identical_candidate_despite_smaller_learning_rate'
                    row.update(accepted=False, reasons=[stop_reason])
                    attempts.append(row); self._log(row); break
                previous_candidate_hash = row['candidate_hash']
                try:
                    new = self._evaluate(evaluate)
                except InvalidCandidate as exc:
                    row.update(accepted=False, reasons=['reference_solve_failed'], message=str(exc))
                    attempts.append(row); self._log(row); continue
                decision = decide(old, new, guard, min_gain=self.config.min_gain,
                                  old_dual_hint=info.old_dual_hint)
                final_decision = decision
                row.update(asdict(decision))
                row['old_bound'] = asdict(old.bound); row['new_bound'] = asdict(new.bound)
                row['old_risks'] = list(old.risks); row['new_risks'] = list(new.risks)
                attempts.append(row); self._log(row)
                if decision.accepted:
                    # Keep the ACTUAL trajectory's parameters, buffers, moments,
                    # RNG, scaler and scheduler state. Nothing is interpolated.
                    self.accepted_segments += 1
                    self.total_committed_optimizer_steps += steps[0]
                    committed = True
                    return StepReport('accepted', str(round_id), attempts, total_steps,
                                      steps[0], decision, float(stats['delta_l2']))
            self.stopped = True
            self.stop_reason = stop_reason
            return StepReport('stalled', str(round_id), attempts, total_steps, 0,
                              final_decision, 0.0, stop_reason)
        except Exception:
            self.stopped = True
            self.stop_reason = 'contract_or_runtime_error'
            raise
        finally:
            if not committed:
                base.restore(self.model, self.optimizer, self.stateful, self.plain)
