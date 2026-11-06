"""Optional v5 experiments. The original v5 implementation stays unchanged.

TCL-inspired wrong-evidence KL is adapted to HARD multilabel targets only.
Belief consistency is an optional aligned-modality adaptation, NOT TCL's two
complete retrieval models or alternating optimizer. All priors remain +1.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
if __package__:
    from .model import ModelConfig, MultiLabelV5
    from .losses import (LossConfig, validate_targets, classification_loss,
                         expected_ce, fume_contrastive, mix_batch, noise_batch)
else:
    from model import ModelConfig, MultiLabelV5
    from losses import (LossConfig, validate_targets, classification_loss,
                        expected_ce, fume_contrastive, mix_batch, noise_batch)


class ExperimentModel(MultiLabelV5):
    def __init__(self, config=ModelConfig(), evidence_mode='shared'):
        if evidence_mode not in ('shared', 'per_label'):
            raise ValueError('evidence_mode must be shared or per_label')
        super().__init__(config)
        self.evidence_mode = evidence_mode
        if evidence_mode == 'per_label':
            # Preserve subsequent RNG: this ablation must not change data ordering.
            with torch.random.fork_rng(devices=[]):
                self.img_evidence = nn.Linear(config.hidden, config.classes)
                self.text_evidence = nn.Linear(config.hidden, config.classes)
            b = config.initial_evidence + math.log(-math.expm1(-config.initial_evidence))
            for layer in (self.img_evidence, self.text_evidence):
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, b)
        # Inherited view() broadcasts [B,1] or uses [B,K]; no change to +1 or mean.


@dataclass(frozen=True)
class ExperimentLoss(LossConfig):
    error_kl: float = 0.0
    kl_anneal_epochs: int = 10
    consistency: float = 0.0
    image_noise_multiplier: float = 1.0
    text_noise_multiplier: float = 1.0

    def validate(self):
        super().validate()
        if type(self.kl_anneal_epochs) is not int or self.kl_anneal_epochs < 0:
            raise ValueError('kl_anneal_epochs must be a nonnegative integer')

    def kl_coefficient(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError('epoch is 1-based')
        if self.kl_anneal_epochs <= 1:
            return self.error_kl
        return self.error_kl * min(1.0, (epoch-1)/(self.kl_anneal_epochs-1))


def wrong_evidence_kl(output, target):
    """Mean KL(Beta(masked alpha, masked beta) || Beta(1,1)).

    Positive label masks alpha to 1 and penalizes beta; negative label does the
    reverse. This analytic special case equals log(q)-1+1/q, q >= 1, avoiding
    cancellation in lgamma/digamma. Soft targets are intentionally rejected.
    """
    validate_targets(target, hard=True)
    a, b = output['alpha'].double(), output['beta'].double()
    if a.shape != target.shape or b.shape != a.shape:
        raise ValueError('KL target/shape mismatch')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a < 1).any() or (b < 1).any():
        raise ValueError('Wrong-evidence KL requires evidence+1')
    q = torch.where(target.bool(), b, a)
    delta = q-1
    return (torch.log1p(delta)-delta/q).mean()


def belief_consistency(image, text):
    """Symmetric L1 of positive/negative belief masses; clean aligned rows only."""
    def belief(o):
        a, b = o['alpha'], o['beta']
        return torch.stack(((a-1)/(a+b), (b-1)/(a+b)), dim=-1)
    if image['alpha'].shape != text['alpha'].shape:
        raise ValueError('Belief comparison requires aligned modalities and labels')
    return F.l1_loss(belief(image), belief(text))


def experiment_objective(model, batch, config, scales, epoch=1, backward=False):
    """Reuses all four v5 losses; optional terms reuse the SAME clean forward.

    One optimizer step per batch remains the caller's responsibility. Staged
    backward changes memory use, not the objective. No filtering or relabeling.
    """
    config.validate()
    xi, xt, y = batch
    validate_targets(y, hard=True)
    if len(xi) != len(y) or len(xt) != len(y) or len(scales) != 2:
        raise ValueError('Aligned input rows and two noise scales required')
    if not all(math.isfinite(float(s)) and s >= 0 for s in scales):
        raise ValueError('Noise scales must be finite and nonnegative')
    kw = config.kl_coefficient(epoch)
    terms, logs = [], {'kl_coefficient': kw}

    def finish(value):
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite experiment loss')
        if backward:
            value.backward()
            terms.append(value.detach())
        else:
            terms.append(value)

    if config.classification or config.contrastive or config.error_kl or config.consistency:
        oi, ot = model(xi, xt)
        clean = xi.new_zeros(())
        components = []
        if config.classification:
            components.append(('classification', (classification_loss(oi,y)+classification_loss(ot,y))/2,
                               config.classification))
        if config.contrastive:
            components.append(('contrastive', fume_contrastive(oi['z'],ot['z'],config.temperature),
                               config.contrastive))
        if config.error_kl:
            components.append(('error_kl', (wrong_evidence_kl(oi,y)+wrong_evidence_kl(ot,y))/2, kw))
        if config.consistency:
            components.append(('consistency', belief_consistency(oi,ot), config.consistency))
        for name, value, weight in components:
            logs[name] = float(value.detach())
            logs['weighted_'+name] = float((weight*value).detach())
            clean = clean+weight*value
        for name, out in (('image',oi),('text',ot)):
            budget = out['evidence_total'].detach()
            logs['evidence_'+name] = float(budget.mean())
            logs['max_evidence_'+name] = float(budget.max())
            logs['h_norm_'+name] = float(out['h'].detach().norm(dim=-1).mean())
        finish(clean)
        # Release ALL references to this graph before constructing the next view.
        del components, clean, oi, ot, out, value
    if config.vicinal:
        mi, mt, target = mix_batch(xi,xt,y,config.mix_beta)
        oi, ot = model(mi,mt)
        v = (expected_ce(oi,target)+expected_ce(ot,target))/2
        logs['vicinal'] = float(v.detach())
        logs['weighted_vicinal'] = float((config.vicinal*v).detach())
        finish(config.vicinal*v)
        del mi, mt, target, oi, ot, v
    if config.noise:
        ns = (scales[0]*config.image_noise_multiplier, scales[1]*config.text_noise_multiplier)
        ni, nt, target = noise_batch(xi,xt,y,ns,config.noise_beta,config.noise_multiplier)
        oi, ot = model(ni,nt)
        v = (expected_ce(oi,target)+expected_ce(ot,target))/2
        logs['noise'] = float(v.detach())
        logs['weighted_noise'] = float((config.noise*v).detach())
        finish(config.noise*v)
    total = sum(terms)
    logs['total'] = float(total.detach())
    return total, logs
