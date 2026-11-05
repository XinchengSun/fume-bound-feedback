"""Full VLS (NeurIPS 2025) adaptation to binary cross-modal matching.

Both input views are mixed in their OWN original feature spaces, before the
FUME encoders. Labels are binary pair relevance, not 100-way multilabel vectors.
No query, example, or uncertain sample is dropped by an uncertainty threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VLSConfig:
    mix_beta: float = 10.0
    noise_beta_clean: float = 1.0
    noise_beta_random: float = 1.0
    noise_std: float = 1.0
    negatives_per_anchor: int = 1
    clean_weight: float = 0.0
    vicinal_weight: float = 1.0
    noise_weight: float = 1.0
    combined_weight: float = 1.0

    def validate(self) -> None:
        numeric = [self.mix_beta, self.noise_beta_clean, self.noise_beta_random,
                   self.noise_std, self.clean_weight, self.vicinal_weight,
                   self.noise_weight, self.combined_weight]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("Configuration values must be finite")
        if not isinstance(self.negatives_per_anchor, int):
            raise ValueError("negatives_per_anchor must be an integer")
        if min(self.mix_beta, self.noise_beta_clean, self.noise_beta_random) <= 0:
            raise ValueError("Beta distribution parameters must be > 0")
        if self.noise_std < 0 or self.negatives_per_anchor < 0:
            raise ValueError("Negative noise/std or negative sampling count")
        if min(self.clean_weight, self.vicinal_weight, self.noise_weight, self.combined_weight) < 0:
            raise ValueError("Loss coefficients must be nonnegative")
        if self.vicinal_weight + self.noise_weight + self.clean_weight <= 0:
            raise ValueError("Enable at least one VLS term")


def matching_pairs(labels: torch.Tensor, negatives_per_anchor: int = 1):
    """Keep every aligned index as a positive; sample supervised negatives.

    labels are multihot [B,K]. Negatives have NO observed shared labels.
    Returning both I->T and T->I negatives helps avoid directional sampling bias.
    A minibatch without negatives still retains every positive; track the count.
    """
    if labels.ndim != 2 or labels.shape[0] == 0:
        raise ValueError("labels must be nonempty [B,K]")
    if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Training labels must be finite hard 0/1 multilabel vectors")
    if (labels.sum(1) == 0).any():
        raise ValueError("Empty label row: cannot declare aligned pairs positive; fix annotation protocol")
    if not isinstance(negatives_per_anchor, int) or negatives_per_anchor < 0:
        raise ValueError("negatives_per_anchor must be a nonnegative integer")
    y = (labels > 0).float()
    overlap = (y @ y.T) > 0
    n = y.shape[0]
    device = y.device
    image = [torch.arange(n, device=device)]
    text = [torch.arange(n, device=device)]
    truth = [torch.ones(n, device=device)]
    for _ in range(negatives_per_anchor):
        ip, tp = [], []
        for i in range(n):
            js = (~overlap[i]).nonzero(as_tuple=False).flatten()
            js = js[js != i]
            if js.numel() > 0:
                j = js[torch.randint(js.numel(), (1,), device=device)]
                ip.append(torch.tensor(i, device=device)); tp.append(j.squeeze())
            js = (~overlap[:, i]).nonzero(as_tuple=False).flatten()
            js = js[js != i]
            if js.numel() > 0:
                j = js[torch.randint(js.numel(), (1,), device=device)]
                ip.append(j.squeeze()); tp.append(torch.tensor(i, device=device))
        if ip:
            im = torch.stack(ip).long(); tx = torch.stack(tp).long()
            image.append(im); text.append(tx)
            truth.append(torch.zeros(len(ip), device=device))
    return torch.cat(image), torch.cat(text), torch.cat(truth)


def sample_beta(a: float, b: float, n: int, device, dtype):
    return torch.distributions.Beta(a, b).sample((n,)).to(device=device, dtype=dtype)


def vicinal_pairs(image: torch.Tensor, text: torch.Tensor, y: torch.Tensor,
                  beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(y)
    partner = torch.randperm(n, device=y.device)
    lam = sample_beta(beta, beta, n, y.device, image.dtype)
    def mix(z):
        l = lam.reshape(-1, *([1] * (z.ndim - 1)))
        return l * z + (1 - l) * z[partner]
    return mix(image), mix(text), lam * y + (1 - lam) * y[partner]


def noise_pairs(image: torch.Tensor, text: torch.Tensor, y: torch.Tensor,
                beta_clean: float, beta_random: float, std: float):
    n = len(y)
    lam = sample_beta(beta_clean, beta_random, n, y.device, image.dtype)
    def pert(z):
        l = lam.reshape(-1, *([1] * (z.ndim - 1)))
        return l*z + (1-l)*(std*torch.randn_like(z))
    return pert(image), pert(text), lam*y + (1-lam)*0.5


def edl_expected_cross_entropy(logits: torch.Tensor, positive_target: torch.Tensor):
    """Paper Eq.(14) for binary categorical (match/nonmatch) soft targets."""
    if logits.shape != (len(positive_target), 2):
        raise ValueError("Evidence logits must be [N,2]")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite evidence logits")
    if (positive_target.ndim != 1 or not torch.isfinite(positive_target).all()
            or (positive_target < 0).any() or (positive_target > 1).any()):
        raise ValueError("Matching targets must be finite scalars in [0,1]")
    alpha = 1.0 + F.softplus(logits)
    t = torch.stack((positive_target, 1-positive_target), dim=-1).to(alpha.dtype)
    return (t*(torch.digamma(alpha.sum(dim=-1, keepdim=True))-
               torch.digamma(alpha))).sum(dim=-1).mean()


class PairEvidenceHead(nn.Module):
    """Black-box evidence predictor, no interpretable evidence assumptions."""
    def __init__(self, embedding_dim: int, hidden: int = 128):
        super().__init__()
        if embedding_dim < 1 or hidden < 1:
            raise ValueError("Positive embedding dimensions required")
        self.net = nn.Sequential(
            nn.Linear(4*embedding_dim+1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, zi: torch.Tensor, zt: torch.Tensor):
        if zi.shape != zt.shape or zi.ndim != 2:
            raise ValueError("Expected paired image/text embeddings [N,D]")
        return self.net(torch.cat((zi, zt, torch.abs(zi-zt), zi*zt,
                                   (zi*zt).sum(dim=-1, keepdim=True)), dim=-1))

    @staticmethod
    def mean_and_vacuity(logits: torch.Tensor):
        alpha = F.softplus(logits) + 1
        return alpha[:, 0] / alpha.sum(-1), 2 / alpha.sum(-1)


class FUMEVLS(nn.Module):
    """Trains both original FUME encoders through VLS; keeps FUME losses intact."""
    def __init__(self, fume_model: nn.Module, embedding_dim: int,
                 config: VLSConfig = VLSConfig(), hidden: int = 128):
        super().__init__()
        config.validate()
        self.fume = fume_model
        self.head = PairEvidenceHead(embedding_dim, hidden)
        self.cfg = config

    def forward_losses(self, images: torch.Tensor, texts: torch.Tensor,
                       labels: torch.Tensor, original_loss_fn, fuzzy_alpha: float):
        if len(images) != len(texts) or len(images) != len(labels):
            raise ValueError("Aligned original multimodal training rows required")
        ret = self.fume(images, texts)
        base = original_loss_fn(ret['view1_feature'], ret['view2_feature'],
                                ret['view1_membershipDegree'], ret['view2_membershipDegree'],
                                labels, labels, fuzzy_alpha)
        if self.cfg.combined_weight == 0:
            zero = base.new_zeros(())
            return base, dict(base=base.detach(), vls=zero, pairs=zero,
                              positives=zero, negatives=zero)
        ip, tp, relevance = matching_pairs(labels, self.cfg.negatives_per_anchor)
        xi = images[ip]
        xt = texts[tp]
        terms: Dict[str, torch.Tensor] = {}
        if self.cfg.vicinal_weight:
            xm, tm, target = vicinal_pairs(xi, xt, relevance, self.cfg.mix_beta)
            logits = self.head(self.fume.img_net(xm), self.fume.text_net(tm))
            terms['vicinal'] = edl_expected_cross_entropy(logits, target)
        if self.cfg.noise_weight:
            xn, tn, target = noise_pairs(xi, xt, relevance,
                                         self.cfg.noise_beta_clean,
                                         self.cfg.noise_beta_random,
                                         self.cfg.noise_std)
            logits = self.head(self.fume.img_net(xn), self.fume.text_net(tn))
            terms['noise'] = edl_expected_cross_entropy(logits, target)
        if self.cfg.clean_weight:
            logits = self.head(self.fume.img_net(xi), self.fume.text_net(xt))
            terms['clean'] = edl_expected_cross_entropy(logits, relevance)
        vls = (self.cfg.vicinal_weight * terms.get('vicinal', base.new_zeros(()))
               + self.cfg.noise_weight * terms.get('noise', base.new_zeros(()))
               + self.cfg.clean_weight * terms.get('clean', base.new_zeros(())))
        total = base + self.cfg.combined_weight*vls
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite combined FUME+VLS objective")
        logs = {'base': base.detach(), 'vls': vls.detach(),
                'pairs': torch.tensor(len(relevance), device=images.device),
                'positives': relevance.sum().detach(),
                'negatives': (1-relevance).sum().detach(),
                **{k: val.detach() for k, val in terms.items()}}
        return total, logs