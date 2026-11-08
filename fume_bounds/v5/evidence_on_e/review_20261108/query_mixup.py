"""Opt-in query-conditioned Mixup for the existing matching-evidence branch.

This module changes ONLY the virtual Mixup pair and its relationship target.
It does not alter the encoders, evidence head, +1 prior, E/Noise losses, weights,
KL setting, original-pair sampler, or retrieval protocol. It is NOT a guarantee
that virtual targets equal real-world conditional matching probabilities.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
from torch import Tensor


def _vector_indices(value: Tensor, length: int, name: str, upper: int,
                    device: torch.device) -> Tensor:
    if value.ndim != 1 or len(value) != length or value.dtype != torch.long:
        raise ValueError(f'{name} must be a length-{length} int64 vector')
    if value.device != device or ((value < 0) | (value >= upper)).any():
        raise ValueError(f'{name}: device or range mismatch')
    return value


def query_conditioned_mixup(
    image: Tensor,
    text: Tensor,
    labels: Tensor,
    image_indices: Tensor,
    text_indices: Tensor,
    beta: float = 10.0,
    *,
    permutation: Optional[Tensor] = None,
    coefficients: Optional[Tensor] = None,
    image_is_query: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep a query fixed and mix two candidates in the OTHER modality.

    image/text are original aligned feature matrices [B,D_I] and [B,D_T].
    labels is the original hard multi-hot [B,K], NOT a scalar pair label.
    image_indices/text_indices specify the already-sampled M original pairs.
    Alternating pairs use image queries and text queries by default. No pair is
    discarded. Supplied coefficients/permutations enable deterministic tests.

    For image query i: text = l*T_j+(1-l)*T_k; target=l*r_ij+(1-l)*r_ik.
    For text query j:  image=l*I_i+(1-l)*I_k; target=l*r_ij+(1-l)*r_kj.
    r_ab is 1 exactly when the corresponding original label sets overlap.

    This is a declared vicinal-supervision assumption, not a measured truth for
    a synthetic mixture. The original E and Noise branches remain unchanged.
    """
    if image.ndim != 2 or text.ndim != 2 or labels.ndim != 2:
        raise ValueError('image, text, labels must be matrices')
    n = image.shape[0]
    if n == 0 or len(text) != n or len(labels) != n or labels.shape[1] == 0:
        raise ValueError('Original aligned batches must be nonempty and same length')
    if not image.is_floating_point() or image.dtype != text.dtype:
        raise ValueError('image/text must have the same floating-point dtype')
    if text.device != image.device or labels.device != image.device:
        raise ValueError('All tensors must share a device')
    if not torch.isfinite(image).all() or not torch.isfinite(text).all():
        raise ValueError('Nonfinite input features')
    if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Use original hard multi-hot labels')
    if (labels.sum(1) == 0).any():
        raise ValueError('Empty labels cannot define positive relationships')
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    m = image_indices.numel()
    if m == 0:
        raise ValueError('At least one sampled pair is required')
    dev = image.device
    i = _vector_indices(image_indices, m, 'image_indices', n, dev)
    j = _vector_indices(text_indices, m, 'text_indices', n, dev)
    if permutation is None:
        permutation = torch.randperm(m, device=dev)
    p = _vector_indices(permutation, m, 'permutation', m, dev)
    if not torch.equal(torch.sort(p).values, torch.arange(m, device=dev)):
        raise ValueError('permutation must contain each pair index exactly once')
    if coefficients is None:
        coefficients = torch.distributions.Beta(beta, beta).sample((m, 1)).to(image)
    if coefficients.shape not in ((m,), (m, 1)):
        raise ValueError('coefficients must have shape [M] or [M,1]')
    lam = coefficients.reshape(m, 1).to(image)
    if not torch.isfinite(lam).all() or ((lam < 0) | (lam > 1)).any():
        raise ValueError('Mixing coefficients must be in [0,1]')
    if image_is_query is None:
        image_is_query = torch.arange(m, device=dev).remainder(2).eq(0)
    if image_is_query.shape != (m,) or image_is_query.dtype != torch.bool:
        raise ValueError('image_is_query must be a bool vector of length M')
    if image_is_query.device != dev:
        raise ValueError('image_is_query device mismatch')

    ip, jp = i[p], j[p]
    relation = lambda a, b: (labels[a].bool() & labels[b].bool()).any(-1, keepdim=True).to(image.dtype)
    original = relation(i, j)
    candidate_target = torch.where(image_is_query[:, None],
                                   relation(i, jp), relation(ip, j))
    mixed_image = torch.where(image_is_query[:, None], image[i],
                              lam * image[i] + (1 - lam) * image[ip])
    mixed_text = torch.where(image_is_query[:, None],
                             lam * text[j] + (1 - lam) * text[jp], text[j])
    target = lam * original + (1 - lam) * candidate_target
    return mixed_image, mixed_text, target
