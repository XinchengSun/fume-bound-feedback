"""Multihot supervision, VLS feature augmentation, and FUME contrastive loss."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    classification: float = 1.0
    vicinal: float = 1.0
    noise: float = 1.0
    contrastive: float = 0.05
    mix_beta: float = 10.0
    noise_beta: float = 1.0
    temperature: float = 1.0
    noise_multiplier: float = 1.0

    def validate(self):
        for key, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if min(self.mix_beta, self.noise_beta, self.temperature) <= 0:
            raise ValueError('Beta shapes and temperature must be positive')
        if self.classification + self.vicinal + self.noise + self.contrastive == 0:
            raise ValueError('At least one training term must be enabled')


def validate_targets(y, hard=False):
    if y.ndim != 2 or not y.numel() or not torch.isfinite(y).all():
        raise ValueError('Expected nonempty finite [B,K] targets')
    if (y < 0).any() or (y > 1).any():
        raise ValueError('Targets must lie in [0,1], with no cross-class normalization')
    if hard and not ((y == 0) | (y == 1)).all():
        raise ValueError('Original labels must be hard multihot vectors')


def classification_loss(output, y):
    validate_targets(y)
    if output['predictive_logits'].shape != y.shape:
        raise ValueError('Classification target shape mismatch')
    # sigmoid(log(alpha)-log(beta)) is the FINAL mean, not the evidence direction.
    return F.binary_cross_entropy_with_logits(output['predictive_logits'], y.float())


def expected_ce(output, target):
    validate_targets(target)
    a, b = output['alpha'].double(), output['beta'].double()
    if a.shape != target.shape or b.shape != a.shape:
        raise ValueError('Beta/target shapes differ')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a < 1).any() or (b < 1).any():
        raise ValueError('v5 uses finite evidence+1 concentrations')
    t = target.double()
    s = a + b  # Only within EACH label. Never sum over the K classes.
    return (t * (torch.digamma(s) - torch.digamma(a))
            + (1-t) * (torch.digamma(s) - torch.digamma(b))).mean()


def fume_contrastive(zi, zt, temperature=1.0):
    """Exactly FUME's two symmetric terms; stable log-softmax implementation.

    Includes intra-modal negatives, excludes self, paired opposite modality is
    the sole positive. Same-label nonpaired rows are NOT excluded (upstream rule).
    """
    if zi.ndim != 2 or zi.shape != zt.shape or len(zi) == 0:
        raise ValueError('Nonempty equal [B,D] embeddings required')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive')
    n = len(zi)
    z = torch.cat((zi, zt), dim=0)
    logits = (z @ z.T) / temperature
    logits = logits.masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), -torch.inf)
    positive = (torch.arange(2*n, device=z.device) + n) % (2*n)
    return 2 * F.cross_entropy(logits, positive)


def _beta(shape, n, device, dtype):
    return torch.distributions.Beta(shape, shape).sample((n, 1)).to(device=device, dtype=dtype)


def mix_batch(image, text, labels, beta=10.0):
    permutation = torch.randperm(len(labels), device=labels.device)
    lam = _beta(beta, len(labels), image.device, image.dtype)
    return (lam*image+(1-lam)*image[permutation],
            lam*text+(1-lam)*text[permutation],
            lam*labels+(1-lam)*labels[permutation])


def noise_batch(image, text, labels, scales, beta=1.0, multiplier=1.0):
    lam = _beta(beta, len(labels), image.device, image.dtype)
    xi = lam*image+(1-lam)*torch.randn_like(image)*scales[0]*multiplier
    xt = lam*text+(1-lam)*torch.randn_like(text)*scales[1]*multiplier
    target = lam*labels+(1-lam)*0.5  # per-label Bernoulli uniform target
    return xi, xt, target


def objective(model, batch, config, scales, backward=False):
    """One batch. Optional staged backward, with NO intervening optimizer step.

    Staging frees each view's graph before the next virtual batch. When
    backward=False, returns a differentiable sum for independent testing.
    """
    config.validate()
    xi, xt, y = batch
    validate_targets(y, hard=True)
    if len(xi) != len(y) or len(xt) != len(y):
        raise ValueError('Paired rows/labels differ')
    terms, values = [], {}

    def add(name, value, weight):
        if not torch.isfinite(value):
            raise FloatingPointError(f'Nonfinite {name} loss')
        values[name] = float(value.detach())
        term = weight*value
        if backward:
            term.backward()
            terms.append(term.detach())
        else:
            terms.append(term)

    if config.classification or config.contrastive:
        oi, ot = model(xi, xt)
        clean = xi.new_zeros(())
        if config.classification:
            cls = (classification_loss(oi, y)+classification_loss(ot, y))/2
            values['classification'] = float(cls.detach())
            clean = clean+config.classification*cls
        if config.contrastive:
            cl = fume_contrastive(oi['z'], ot['z'], config.temperature)
            values['contrastive'] = float(cl.detach())
            clean = clean+config.contrastive*cl
        values['evidence_image'] = float(oi['evidence_total'].detach().mean())
        values['evidence_text'] = float(ot['evidence_total'].detach().mean())
        values['h_norm_image'] = float(oi['h'].detach().norm(dim=-1).mean())
        values['h_norm_text'] = float(ot['h'].detach().norm(dim=-1).mean())
        add('clean_total', clean, 1.0)
        del oi, ot, clean
    if config.vicinal:
        mi, mt, target = mix_batch(xi, xt, y, config.mix_beta)
        oi, ot = model(mi, mt)
        add('vicinal', (expected_ce(oi, target)+expected_ce(ot, target))/2, config.vicinal)
        del oi, ot, mi, mt, target
    if config.noise:
        ni, nt, target = noise_batch(xi, xt, y, scales, config.noise_beta, config.noise_multiplier)
        oi, ot = model(ni, nt)
        add('noise', (expected_ce(oi, target)+expected_ce(ot, target))/2, config.noise)
    total = sum(terms)
    values['total'] = float(total.detach())
    return total, values
