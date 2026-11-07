"""Explicit alternatives: category evidence OR image-text matching evidence.

Existing v5 files/defaults are unchanged. Matching retains ordinary multihot BCE
and FUME contrastive learning; it does NOT instantiate category evidence layers.
No fuzzy c/necessity, uncertainty-based filtering, or test-dependent training.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
if __package__:
    from .model import ModelConfig, MultiLabelV5
    from .experiment import ExperimentModel, ExperimentLoss, experiment_objective, wrong_evidence_kl
    from .losses import validate_targets, expected_ce, fume_contrastive, mix_batch, noise_batch
else:
    from model import ModelConfig, MultiLabelV5
    from experiment import ExperimentModel, ExperimentLoss, experiment_objective, wrong_evidence_kl
    from losses import validate_targets, expected_ce, fume_contrastive, mix_batch, noise_batch


class PairEvidenceHead(nn.Module):
    """Same 4d+1 -> hidden -> 2 structure as the historical vls_transfer head."""
    def __init__(self, dimension, hidden=128):
        super().__init__()
        if type(hidden) is not int or hidden < 1:
            raise ValueError('pair_hidden must be a positive integer')
        self.net = nn.Sequential(nn.Linear(4*dimension+1, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, image, text):
        if image.ndim != 2 or image.shape != text.shape:
            raise ValueError('Paired embeddings must have identical [M,D] shapes')
        product = image*text
        logits = self.net(torch.cat((image, text, (image-text).abs(), product,
                                    product.sum(-1, keepdim=True)), -1))
        concentrations = 1 + F.softplus(logits)
        if not torch.isfinite(concentrations).all():
            raise FloatingPointError('Nonfinite pair evidence')
        alpha, beta = concentrations[:, :1], concentrations[:, 1:]
        return dict(alpha=alpha, beta=beta, probability=alpha/(alpha+beta),
                    predictive_logits=alpha.log()-beta.log(),
                    evidence_total=alpha+beta-2)


class MatchingModel(MultiLabelV5):
    def __init__(self, config=ModelConfig(), pair_hidden=128):
        # Same backbone and W initialization as the category arm for the same seed.
        super().__init__(config)
        del self.img_evidence
        del self.text_evidence
        # Do not let head size change the RNG state used to generate training data.
        with torch.random.fork_rng(devices=[]):
            self.pair_head = PairEvidenceHead(config.classes, pair_hidden)
        self.evidence_task = 'matching'

    def view(self, x, modality):
        if modality not in ('image', 'text'):
            raise ValueError('modality must be image or text')
        tower = self.img_net if modality == 'image' else self.text_net
        hidden, h, z = tower.features(x)
        logits = h @ self.W
        return dict(hidden=hidden, h=h, z=z, logits=logits,
                    probability=logits.sigmoid())  # ordinary multihot classifier, not Beta UQ

    def matching(self, image, text):
        return self.pair_head(self.img_net(image), self.text_net(text))


@dataclass(frozen=True)
class PairConfig:
    negatives_per_anchor: int = 1
    positive_mode: str = 'overlap'
    chunk: int = 128
    clean_weight: float = 0.0

    def validate(self):
        if type(self.negatives_per_anchor) is not int or self.negatives_per_anchor < 0:
            raise ValueError('negatives_per_anchor must be a nonnegative integer')
        if self.positive_mode not in ('aligned', 'overlap'):
            raise ValueError('positive_mode must be aligned or overlap')
        if type(self.chunk) is not int or self.chunk < 1:
            raise ValueError('pair chunk must be positive')
        if not math.isfinite(self.clean_weight) or self.clean_weight < 0:
            raise ValueError('pair clean weight must be finite and nonnegative')


def sample_training_pairs(labels, config=PairConfig()):
    """One positive per anchor plus sampled disjoint-label negatives.

    A uniformly chosen orientation anchors an image or a text, without claiming
    independent pairs. Overlap-positive sampling includes nonaligned positives.
    Sampling may repeat a pair. No-negative anchors keep their positive; all
    original records are still used in the clean BCE/contrastive forward.
    """
    config.validate(); validate_targets(labels, hard=True)
    if (labels.sum(1) == 0).any():
        raise ValueError('Empty labels cannot define positive pairs')
    y = labels.float()
    overlap = (y @ y.T) > 0
    n, device = len(y), y.device
    anchor = torch.arange(n, device=device)
    positive = (anchor if config.positive_mode == 'aligned' else
                torch.multinomial(overlap.float(), 1).squeeze(1))
    reverse = torch.rand(n, device=device) < .5
    images = [torch.where(reverse, positive, anchor)]
    texts = [torch.where(reverse, anchor, positive)]
    targets = [torch.ones(n, 1, device=device)]
    available = (~overlap).any(1)
    candidates = (~overlap).float()
    candidates[~available] = 0
    candidates[~available, anchor[~available]] = 1  # placeholder never returned as a negative
    for _ in range(config.negatives_per_anchor):
        negative = torch.multinomial(candidates, 1).squeeze(1)
        reverse = torch.rand(n, device=device) < .5
        images.append(torch.where(reverse, negative, anchor)[available])
        texts.append(torch.where(reverse, anchor, negative)[available])
        targets.append(torch.zeros(int(available.sum()), 1, device=device))
    i, j, target = torch.cat(images), torch.cat(texts), torch.cat(targets)
    if not torch.equal(overlap[i, j], target[:, 0].bool()):
        raise RuntimeError('Pair construction violated the label-overlap rule')
    return i, j, target, dict(pairs=len(i), positives=n, negatives=len(i)-n,
        nonaligned_positives=int((images[0] != texts[0]).sum()),
        anchors_without_negative=int((~available).sum()))


def build_model(config, task='category', evidence_mode='shared', pair_hidden=128):
    if task == 'category':
        model = ExperimentModel(config, evidence_mode)
        model.evidence_task = 'category'
        return model
    if task == 'matching':
        if evidence_mode != 'shared':
            raise ValueError('per_label evidence mode applies only to category evidence')
        return MatchingModel(config, pair_hidden)
    raise ValueError('task must be category or matching')


def ab_objective(model, batch, loss, scales, *, epoch=1, backward=False,
                 classification_source='direction', pair_config=PairConfig()):
    """Mutually exclusive evidence objectives; a single optimizer step by caller.

    category+posterior delegates exactly to the old v5 experiment objective.
    The explicit A/B script uses common direction-BCE in both arms. This is a
    declared comparison control, NOT a silent change to legacy v5's default.
    """
    loss.validate(); pair_config.validate()
    task = model.evidence_task
    if classification_source not in ('direction', 'posterior'):
        raise ValueError('Invalid classification source')
    if task == 'category' and classification_source == 'posterior':
        if pair_config.clean_weight:
            raise ValueError('Pair-only clean loss cannot be enabled for categories')
        return experiment_objective(model, batch, loss, scales, epoch, backward)
    if task == 'matching' and classification_source != 'direction':
        raise ValueError('Matching has no category posterior; use direction BCE')
    if task == 'matching' and loss.consistency:
        raise ValueError('Category belief consistency has no pair counterpart; keep it disabled')
    if task == 'category' and pair_config.clean_weight:
        raise ValueError('Pair-only clean loss cannot be enabled for categories')
    xi, xt, y = batch
    validate_targets(y, hard=True)
    if len(xi) != len(xt) or len(xi) != len(y) or len(scales) != 2:
        raise ValueError('Aligned inputs and two noise scales required')
    if not all(math.isfinite(float(s)) and s >= 0 for s in scales):
        raise ValueError('Noise scales must be finite and nonnegative')
    terms, logs = [], {'kl_coefficient': loss.kl_coefficient(epoch)}

    def finish(name, value, weight):
        if not torch.isfinite(value):
            raise FloatingPointError(f'Nonfinite {name}')
        logs[name] = logs.get(name, 0.) + float(value.detach())
        logs['weighted_'+name] = logs.get('weighted_'+name, 0.) + float((weight*value).detach())
        if backward:
            (weight*value).backward()
            terms.append((weight*value).detach())
        else:
            terms.append(weight*value)

    if task == 'matching':
        i, j, targets, counts = sample_training_pairs(y, pair_config)
        logs.update({'pair_'+k: float(v) for k, v in counts.items()})
    # The original complete multihot labels are always available to this path.
    if (loss.classification or loss.contrastive or loss.error_kl or loss.consistency
            or (task == 'matching' and pair_config.clean_weight)):
        oi, ot = model(xi, xt)
        clean = xi.new_zeros(())
        if loss.classification:
            cls = (F.binary_cross_entropy_with_logits(oi['logits'], y.float())+
                   F.binary_cross_entropy_with_logits(ot['logits'], y.float()))/2
            logs['classification'] = float(cls.detach())
            logs['weighted_classification'] = loss.classification*logs['classification']
            clean = clean + loss.classification*cls
        if loss.contrastive:
            cl = fume_contrastive(oi['z'], ot['z'], loss.temperature)
            logs['contrastive'] = float(cl.detach())
            logs['weighted_contrastive'] = loss.contrastive*logs['contrastive']
            clean = clean + loss.contrastive*cl
        if task == 'category':
            if loss.error_kl:
                kl = (wrong_evidence_kl(oi, y)+wrong_evidence_kl(ot, y))/2
                logs['error_kl'] = float(kl.detach())
                logs['weighted_error_kl'] = logs['kl_coefficient']*logs['error_kl']
                clean = clean+logs['kl_coefficient']*kl
            if loss.consistency:
                if __package__:
                    from .experiment import belief_consistency
                else:
                    from experiment import belief_consistency
                c = belief_consistency(oi, ot)
                logs['consistency'] = float(c.detach())
                logs['weighted_consistency'] = loss.consistency*logs['consistency']
                clean = clean+loss.consistency*c
        if task == 'matching' and (loss.error_kl or pair_config.clean_weight):
            pair_out = model.pair_head(oi['z'][i], ot['z'][j])
            if loss.error_kl:
                kl = wrong_evidence_kl(pair_out, targets)
                logs['error_kl'] = float(kl.detach())
                logs['weighted_error_kl'] = logs['kl_coefficient']*logs['error_kl']
                clean = clean+logs['kl_coefficient']*kl
            if pair_config.clean_weight:
                pcl = expected_ce(pair_out, targets)
                logs['pair_clean'] = float(pcl.detach())
                logs['weighted_pair_clean'] = pair_config.clean_weight*logs['pair_clean']
                clean = clean+pair_config.clean_weight*pcl
            del pair_out
        for name, out in (('image', oi), ('text', ot)):
            logs['h_norm_'+name] = float(out['h'].detach().norm(dim=-1).mean())
            if task == 'category':
                logs['evidence_'+name] = float(out['evidence_total'].detach().mean())
        finish('clean_total', clean, 1.)
        del oi, ot, out, clean

    if task == 'matching':
        inputs = (xi[i], xt[j], targets)
        # Mathematically average over ALL sampled pairs, not average chunk means.
        def pair_loss(name, data, weight, fn):
            m = len(data[2])
            for start in range(0, m, pair_config.chunk):
                end = min(m, start+pair_config.chunk)
                pred = model.matching(data[0][start:end], data[1][start:end])
                value = fn(pred, data[2][start:end]) * ((end-start)/m)
                finish(name, value, weight)
    else:
        inputs = (xi, xt, y)

    ns = (float(scales[0])*loss.image_noise_multiplier,
          float(scales[1])*loss.text_noise_multiplier)
    for name, weight in (('vicinal', loss.vicinal), ('noise', loss.noise)):
        if not weight:
            continue
        data = (mix_batch(*inputs, beta=loss.mix_beta) if name == 'vicinal' else
                noise_batch(*inputs, ns, beta=loss.noise_beta, multiplier=loss.noise_multiplier))
        if task == 'matching':
            pair_loss(name, data, weight, expected_ce)
        else:
            oi, ot = model(data[0], data[1])
            finish(name, (expected_ce(oi, data[2])+expected_ce(ot, data[2]))/2, weight)
            del oi, ot
        del data
    if not terms:
        raise ValueError('No active objective')
    total = sum(terms)
    logs['total'] = float(total.detach())
    return total, logs
