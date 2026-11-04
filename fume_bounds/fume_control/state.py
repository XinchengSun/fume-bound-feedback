"""Transactional state: parameters, buffers, optimizer, explicit extras and RNG.

CPU snapshots avoid doubling GPU memory. No claim of cross-platform bitwise
reproducibility. Fixed replay batches and deterministic kernels are still needed.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import random
from typing import Any, Mapping
import numpy as np
import torch


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((k, cpu_copy(v)) for k, v in value.items())
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def digest(value: Any) -> str:
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            # Works also for bfloat16 and scalar tensors.
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=lambda k: str(k)):
                add(key); add(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x: add(item)
        else:
            h.update(repr(x).encode('utf-8'))
    add(value)
    return h.hexdigest()


@dataclass
class RNGState:
    python: Any
    numpy: Any
    torch_cpu: torch.Tensor
    cuda: list[torch.Tensor] | None

    @classmethod
    def capture(cls):
        return cls(random.getstate(), copy.deepcopy(np.random.get_state()),
                   torch.get_rng_state().clone(),
                   [s.clone() for s in torch.cuda.get_rng_state_all()]
                   if torch.cuda.is_initialized() else None)

    def restore(self):
        random.setstate(self.python); np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


class GeneratorState:
    """Register a non-global torch.Generator in stateful extras."""
    def __init__(self, generator: torch.Generator): self.generator = generator
    def state_dict(self): return {'state': self.generator.get_state().clone()}
    def load_state_dict(self, state): self.generator.set_state(state['state'])


def fume_plain_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Find the original FUME's ordinary W tensor (not in state_dict)."""
    result = {}
    registered = {id(x) for x in model.parameters()} | {id(x) for x in model.buffers()}
    for name, module in model.named_modules():
        w = getattr(module, 'W', None)
        if isinstance(w, torch.Tensor) and id(w) not in registered:
            result[f'{name}.W' if name else 'W'] = w
    return result


@dataclass
class Snapshot:
    model_state: Any
    buffers: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor | None]
    requires_grad: dict[str, bool]
    modes: dict[str, bool]
    optimizer: Any
    stateful: dict[str, Any]
    plain: dict[str, torch.Tensor]
    rng: RNGState
    parameter_ids: tuple[int, ...]

    @classmethod
    def capture(cls, model, optimizer, stateful, plain):
        return cls(cpu_copy(model.state_dict()),
                   {n: cpu_copy(b) for n, b in model.named_buffers()},
                   {n: None if p.grad is None else cpu_copy(p.grad) for n, p in model.named_parameters()},
                   {n: p.requires_grad for n, p in model.named_parameters()},
                   {n: m.training for n, m in model.named_modules()},
                   cpu_copy(optimizer.state_dict()),
                   {n: cpu_copy(obj.state_dict()) for n, obj in stateful.items()},
                   {n: cpu_copy(t) for n, t in plain.items()}, RNGState.capture(),
                   tuple(id(p) for p in model.parameters()))

    def restore(self, model, optimizer, stateful, plain):
        if tuple(id(p) for p in model.parameters()) != self.parameter_ids:
            raise RuntimeError('Proposal replaced Parameter objects; in-place rollback is unsafe.')
        model.load_state_dict(self.model_state, strict=True)
        with torch.no_grad():
            for n, b in model.named_buffers():
                b.copy_(self.buffers[n].to(device=b.device, dtype=b.dtype))
            for n, t in plain.items():
                t.copy_(self.plain[n].to(device=t.device, dtype=t.dtype))
        for n, p in model.named_parameters():
            p.requires_grad_(self.requires_grad[n])
            old_grad = self.gradients[n]
            p.grad = None if old_grad is None else old_grad.to(device=p.device, dtype=p.dtype).clone()
        for n, module in model.named_modules(): module.training = self.modes[n]
        # Schedulers/scalers must already be constructed before optimizer load.
        for n, obj in stateful.items(): obj.load_state_dict(copy.deepcopy(self.stateful[n]))
        optimizer.load_state_dict(copy.deepcopy(self.optimizer))
        # A candidate must not allocate/destroy devices mid-transaction.
        self.rng.restore()


def model_payload(model, plain):
    return {'state': model.state_dict(),
            'all_buffers': dict(model.named_buffers()), 'plain': plain}


def parameter_stats(model, before: Snapshot) -> dict[str, float | int]:
    delta2, base2, max_abs, changed, finite = 0.0, 0.0, 0.0, 0, True
    for name, p in model.named_parameters():
        if not before.requires_grad[name]:
            continue
        old = before.model_state[name].double()
        new = p.detach().cpu().double()
        finite = finite and bool(torch.isfinite(new).all())
        d = new - old
        delta2 += float((d*d).sum()); base2 += float((old*old).sum())
        if d.numel(): max_abs = max(max_abs, float(d.abs().max()))
        changed += int(torch.count_nonzero(d))
    return {'finite': finite, 'delta_l2': delta2**0.5, 'base_l2': base2**0.5,
            'max_abs_delta': max_abs, 'changed_coordinates': changed}
