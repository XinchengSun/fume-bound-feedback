"""Unfiltered multi-label retrieval and model-implied Beta uncertainty.

No ground-truth AU/EU identification and no independence of N*K labels is claimed.
"""
from pathlib import Path
import json
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

SCORES = ('total_entropy', 'aleatoric_entropy', 'epistemic_mi', 'beta_entropy', 'vacuity')


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def beta_statistics(alpha, beta):
    a, b = alpha.double(), beta.double()
    if a.shape != b.shape or not a.numel() or (a <= 0).any() or (b <= 0).any():
        raise ValueError('Equal, nonempty, positive Beta parameters required')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite Beta parameters')
    s = a+b
    p = a/s
    total = -torch.xlogy(p, p)-torch.xlogy(1-p, 1-p)
    conditional = torch.digamma(s+1)-p*torch.digamma(a+1)-(1-p)*torch.digamma(b+1)
    mi = total-conditional
    if (mi < -1e-8).any() or (conditional < -1e-8).any():
        raise FloatingPointError('Beta entropy decomposition inconsistent')
    mi = mi.clamp_min(0)
    out = dict(probability=p, concentration=s, total_entropy=total,
               aleatoric_entropy=total-mi, epistemic_mi=mi,
               beta_entropy=torch.lgamma(a)+torch.lgamma(b)-torch.lgamma(s)
               -(a-1)*torch.digamma(a)-(b-1)*torch.digamma(b)+(s-2)*torch.digamma(s),
               vacuity=2/s)
    if not all(torch.isfinite(x).all() for x in out.values()):
        raise FloatingPointError('Nonfinite Beta statistics')
    return out


def detection(y, score):
    y, score = np.asarray(y, dtype=float).ravel(), np.asarray(score, dtype=float).ravel()
    if y.shape != score.shape or not len(y) or not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite equal-sized binary labels and scores required')
    positive, negative = y.sum(), len(y)-y.sum()
    if not positive or not negative:
        return dict(auroc=None, ap=None, fpr95=None, reason='only_one_class')
    order = np.argsort(-score, kind='stable')
    ys, scores = y[order], score[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(y)-1]
    tp = np.cumsum(ys)[ends]
    fp = ends+1-tp
    tpr, fpr = np.r_[0., tp/positive], np.r_[0., fp/negative]
    return dict(auroc=float(np.trapezoid(tpr, fpr)),
                ap=float(np.sum(np.diff(tpr)*tp/(ends+1))),
                fpr95=float(fpr[tpr >= .95].min()), reason=None)


def risk_coverage(error, score):
    e, s = np.asarray(error, dtype=float).ravel(), np.asarray(score, dtype=float).ravel()
    if e.shape != s.shape or not len(e) or not np.isfinite(s).all() or not np.isin(e, [0, 1]).all():
        raise ValueError('Binary errors and finite equal-sized uncertainty required')
    order = np.argsort(s, kind='stable')
    e, s = e[order], s[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1])+1, len(s)]
    starts = np.r_[0, ends[:-1]]
    sizes = ends-starts
    # Expected random ordering within exact ties: never exploit input order.
    averaged = np.repeat(np.add.reduceat(e, starts)/sizes, sizes)
    ranks = np.arange(1, len(e)+1, dtype=float)
    risks = averaged.cumsum()/ranks
    oracle = np.sort(e).cumsum()/ranks
    return dict(coverage=ranks/len(e), risk=risks, aurc=float(risks.mean()),
                oracle_aurc=float(oracle.mean()), excess_aurc=float(max(0, risks.mean()-oracle.mean())))


@torch.no_grad()
def retrieval_map(image, text, labels, chunk=128):
    if chunk < 1 or len(image) != len(text) or len(image) != len(labels) or not len(image):
        raise ValueError('Invalid retrieval inputs')
    image, text = F.normalize(image.float(), dim=-1), F.normalize(text.float(), dim=-1)
    y = labels.float()
    def direction(query, gallery):
        aps, zero = [], 0
        for start in range(0, len(query), chunk):
            score = query[start:start+chunk] @ gallery.T
            rel = (y[start:start+chunk] @ y.T) > 0
            order = torch.argsort(score, dim=1, descending=True, stable=True)
            ranked = rel.gather(1, order).float()
            ranks = torch.arange(1, len(gallery)+1, device=ranked.device)
            nrel = rel.sum(1)
            zero += int((nrel == 0).sum())
            ap = (ranked.cumsum(1)/ranks*ranked).sum(1)/nrel.clamp_min(1)
            aps.append(ap.cpu())
        return float(torch.cat(aps).mean()), zero
    a, za = direction(image, text)
    b, zb = direction(text, image)
    return dict(i2t_map_all=a, t2i_map_all=b, mean_map=(a+b)/2, queries=len(image),
                candidates=len(text), zero_relevance_i2t=za, zero_relevance_t2i=zb,
                paired_items_included=True, confidence_filter=False)


def classification_report(labels, stats):
    y = np.asarray(labels, dtype=float)
    p = np.asarray(stats['probability'], dtype=float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid multilabel evaluation shapes/labels')
    clipped = np.clip(p, 1e-12, 1-1e-12)
    error = ((p >= .5) != y).astype(float)
    bins = np.minimum((p.ravel()*15).astype(int), 14)
    ece = sum(float(np.mean(bins == k))*abs(float(p.ravel()[bins == k].mean()
                    -y.ravel()[bins == k].mean())) for k in range(15) if np.any(bins == k))
    per_class = [detection(y[:, k], p[:, k]) for k in range(y.shape[1])]
    ap = [v['ap'] for v in per_class if v['ap'] is not None]
    out = dict(n_samples=len(y), n_labels=y.shape[1], brier=float(((p-y)**2).mean()),
               nll=float((-y*np.log(clipped)-(1-y)*np.log1p(-clipped)).mean()),
               label_accuracy=float(1-error.mean()), exact_match=float((error.sum(1) == 0).mean()),
               ece_15bins=float(ece), macro_ap=float(np.mean(ap)) if ap else None,
               macro_ap_valid_classes=len(ap), per_class=per_class, uncertainty={})
    curves = {}
    for name in SCORES:
        values = stats[name]
        curve = risk_coverage(error, values)
        out['uncertainty'][name] = dict(mean=float(values.mean()), min=float(values.min()),
            max=float(values.max()), error_detection=detection(error, values),
            **{k: curve[k] for k in ('aurc', 'oracle_aurc', 'excess_aurc')})
        curves[name] = curve
    return out, curves


@torch.no_grad()
def evaluate(model, dataset, output=None, batch_size=128, chunk=128):
    old_mode = model.training
    model.eval()
    device = next(model.parameters()).device
    labels, views = [], {'image': [], 'text': []}
    try:
        for xi, xt, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False):
            oi, ot = model(xi.to(device), xt.to(device))
            labels.append(y)
            for name, data in (('image', oi), ('text', ot)):
                values = {k: v.cpu() for k, v in beta_statistics(data['alpha'], data['beta']).items()}
                values.update(z=data['z'].cpu(), alpha=data['alpha'].cpu(), beta=data['beta'].cpu())
                views[name].append(values)
        if not labels:
            raise ValueError('Empty evaluation dataset')
        y = torch.cat(labels)
        merged = {v: {k: torch.cat([b[k] for b in views[v]]) for k in views[v][0]} for v in views}
        report = dict(retrieval=retrieval_map(merged['image']['z'], merged['text']['z'], y, chunk),
                      protocol=dict(labels='multihot, no argmax', uncertain_quantities='model-implied, not ground truth',
                      uq_units='label decisions, statistically dependent', predictive_threshold=.5,
                      log_unit='nats', no_filtering=True), classification={})
        if output is not None:
            output = Path(output)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('Use a new/empty evaluation directory')
            output.mkdir(parents=True, exist_ok=True)
        for name in views:
            arrays = {k: v.numpy() for k, v in merged[name].items()}
            summary, curves = classification_report(y.numpy(), arrays)
            report['classification'][name] = summary
            if output is not None:
                np.savez_compressed(output/f'{name}_predictions.npz', labels=y.numpy(), **arrays)
                for score, curve in curves.items():
                    ix = np.unique(np.linspace(0, len(curve['risk'])-1, min(1000, len(curve['risk'])), dtype=int))
                    with (output/f'{name}_{score}_risk_coverage.csv').open('w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(('coverage', 'label_error_risk'))
                        writer.writerows(zip(curve['coverage'][ix], curve['risk'][ix]))
        if output is not None:
            save_json(output/'metrics.json', report)
        return report
    finally:
        model.train(old_mode)
