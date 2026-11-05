"""Dirichlet metrics and binary evaluation; VLS Appendix C, natural logs.

These are model-implied quantities, NOT ground-truth AU/EU measurements.
All scores are oriented so that a larger value means more uncertainty.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F

SCORES = ('total_entropy', 'aleatoric_entropy', 'epistemic_mi',
          'dirichlet_entropy', 'vacuity')


def dirichlet_statistics(alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    a = torch.as_tensor(alpha).to(torch.float64)
    if a.ndim < 2 or a.shape[-1] < 2 or not a.numel():
        raise ValueError('alpha must be nonempty [..., K], K >= 2')
    if not torch.isfinite(a).all() or (a <= 0).any():
        raise ValueError('Dirichlet concentrations must be finite and positive')
    s = a.sum(-1)
    p = a / s.unsqueeze(-1)
    total = -(p * p.log()).sum(-1)
    aleatoric = torch.digamma(s + 1) - (p * torch.digamma(a + 1)).sum(-1)
    mi = total - aleatoric
    # Only round-off may be clamped; large inconsistencies are errors.
    if (mi < -1e-8).any() or (aleatoric < -1e-8).any():
        raise FloatingPointError('Entropy decomposition failed numerically')
    mi = mi.clamp_min(0)
    result = dict(mean_probability=p, concentration=s,
                  total_entropy=total, aleatoric_entropy=total-mi,
                  epistemic_mi=mi,
                  dirichlet_entropy=(torch.lgamma(a).sum(-1)-torch.lgamma(s)
                                     +(s-a.shape[-1])*torch.digamma(s)
                                     -((a-1)*torch.digamma(a)).sum(-1)),
                  vacuity=a.shape[-1]/s)
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite uncertainty statistics')
    return result


def statistics_from_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite evidence logits')
    return dirichlet_statistics(1 + F.softplus(logits.to(torch.float64)))


def _vector(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a nonempty finite vector')
    return a


def _binary(y) -> np.ndarray:
    a = _vector(y, 'labels')
    if not np.isin(a, [0, 1]).all():
        raise ValueError('Evaluation labels must be binary')
    return a


def detection_metrics(y, score) -> dict:
    """Positive class = detected event (error/OOD/relevance as specified by caller).

    ROC/PR include tied thresholds as groups. Degenerate labels -> null metrics.
    FPR95 is the smallest empirical FPR at TPR >= .95 (no interpolation).
    """
    y, score = _binary(y), _vector(score, 'score')
    if y.shape != score.shape:
        raise ValueError('Label/score shapes differ')
    pos, neg = y.sum(), len(y)-y.sum()
    if pos == 0 or neg == 0:
        return dict(auroc=None, average_precision=None, fpr_at_tpr95=None,
                    reason='Only one class is present')
    order = np.argsort(-score, kind='stable')
    ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    tp = np.cumsum(ys)[ends]
    fp = ends+1-tp
    tpr, fpr = np.r_[0., tp/pos], np.r_[0., fp/neg]
    ap = np.sum(np.diff(tpr) * tp/(ends+1))
    return dict(auroc=float(np.trapezoid(tpr, fpr)),
                average_precision=float(ap),
                fpr_at_tpr95=float(fpr[tpr >= .95].min()), reason=None)


def risk_coverage(errors, uncertainty) -> dict:
    """0/1 error risk after keeping the least uncertain items.

    Exact ties use the expected risk of random ordering WITHIN the tied group,
    so positive-first pair enumeration cannot manufacture a good AURC.
    AURC = mean risk at coverages 1/N,...,1, including ties' expected risks.
    """
    errors, score = _binary(errors), _vector(uncertainty, 'uncertainty')
    if errors.shape != score.shape:
        raise ValueError('Error/score shapes differ')
    order = np.argsort(score, kind='stable')
    e, s = errors[order], score[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:])+1, len(s)]
    starts = np.r_[0, ends[:-1]]
    sums = np.add.reduceat(e, starts)
    group_size = ends-starts
    average = np.repeat(sums/group_size, group_size)
    k = np.arange(1, len(e)+1, dtype=float)
    risk = np.cumsum(average)/k
    oracle = np.cumsum(np.sort(errors))/k
    return dict(coverage=k/len(e), risk=risk, aurc=float(risk.mean()),
                oracle_aurc=float(oracle.mean()),
                excess_aurc=float(max(0., risk.mean()-oracle.mean())))


def binary_report(y, p, scores: dict, bins: int = 15):
    y, p = _binary(y), _vector(p, 'probabilities')
    if y.shape != p.shape or (p < 0).any() or (p > 1).any() or bins < 1:
        raise ValueError('Invalid probabilities, dimensions or bin count')
    errors = ((p >= .5) != y).astype(float)
    edges = np.minimum((p*bins).astype(int), bins-1)
    calibration, ece = [], 0.
    for b in range(bins):
        mask = edges == b
        n = int(mask.sum())
        pred = float(p[mask].mean()) if n else None
        freq = float(y[mask].mean()) if n else None
        if n:
            ece += n/len(y)*abs(pred-freq)
        calibration.append(dict(bin=b, count=n, mean_match_probability=pred,
                                observed_match_frequency=freq))
    clipped = np.clip(p, 1e-12, 1-1e-12)
    result = dict(n_pairs=len(y), n_positive=int(y.sum()), n_negative=int((1-y).sum()),
                  accuracy=float(1-errors.mean()),
                  brier_binary=float(np.mean((p-y)**2)),
                  brier_two_class_sum=float(2*np.mean((p-y)**2)),
                  nll=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
                  ece_match_probability=float(ece), calibration_bins=calibration,
                  relevance_detection=detection_metrics(y, p), uncertainty={})
    curves = {}
    for name, value in scores.items():
        value = _vector(value, name)
        curve = risk_coverage(errors, value)
        result['uncertainty'][name] = dict(
            mean=float(value.mean()), min=float(value.min()), max=float(value.max()),
            error_detection=detection_metrics(errors, value),
            **{k:curve[k] for k in ('aurc','oracle_aurc','excess_aurc')})
        curves[name] = curve
    return result, curves


def ood_report(id_scores: dict, ood_scores: dict) -> dict:
    report = {}
    for name in SCORES:
        ids, ods = _vector(id_scores[name], name), _vector(ood_scores[name], name)
        report[name] = detection_metrics(np.r_[np.zeros(len(ids)), np.ones(len(ods))],
                                         np.r_[ids, ods])
    return dict(n_id=len(ids), n_ood=len(ods), positive_class='OOD', scores=report)
