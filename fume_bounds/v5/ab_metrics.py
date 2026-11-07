"""Same unfiltered retrieval protocol; distinct category and pair UQ reports.

Category uncertainty is NOT converted into matching uncertainty by an untested
formula. Native category-label and pair-label metrics must not be directly ranked.
"""
from pathlib import Path
import hashlib
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
if __package__:
    from .metrics import beta_statistics, classification_report, retrieval_map, detection, save_json
else:
    from metrics import beta_statistics, classification_report, retrieval_map, detection, save_json


def multilabel_probability_report(labels, probability):
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if (y.ndim != 2 or not y.size or p.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()):
        raise ValueError('Finite [N,K] probabilities and hard multihot labels required')
    c = np.clip(p, 1e-12, 1-1e-12)
    error = (p >= .5) != y
    per_class = [detection(y[:, k], p[:, k]) for k in range(y.shape[1])]
    valid = [r['ap'] for r in per_class if r['ap'] is not None]
    bins = np.minimum((p.ravel()*15).astype(int), 14)
    ece = 0.
    for k in range(15):
        mask = bins == k
        if mask.any():
            ece += mask.mean()*abs(p.ravel()[mask].mean()-y.ravel()[mask].mean())
    return dict(n_samples=len(y), n_labels=y.shape[1], brier=float(((p-y)**2).mean()),
        nll=float((-y*np.log(c)-(1-y)*np.log1p(-c)).mean()),
        label_accuracy=float(1-error.mean()), exact_match=float((~error).all(1).mean()),
        ece_15bins=float(ece), macro_ap=float(np.mean(valid)) if valid else None,
        macro_ap_valid_classes=len(valid), per_class=per_class)


def uniform_eval_pairs(n, budget, seed):
    if type(n) is not int or n < 1 or type(budget) is not int or budget < 0:
        raise ValueError('n must be positive and pair budget nonnegative')
    total = n*n
    flat = (np.arange(total, dtype=np.int64) if budget == 0 or budget >= total else
            np.random.default_rng(seed).choice(total, budget, replace=False))
    flat.sort()
    return flat//n, flat % n


def write_curves(output, curves, unit):
    for score, curve in curves.items():
        take = np.unique(np.linspace(0, len(curve['risk'])-1, min(1000, len(curve['risk'])), dtype=int))
        with (output/f'{score}_risk_coverage.csv').open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['coverage', unit+'_error_risk'])
            writer.writerows(zip(curve['coverage'][take], curve['risk'][take]))


@torch.no_grad()
def evaluate_ab(model, dataset, output=None, *, batch_size=128, chunk=128,
                pair_budget=20000, pair_seed=20261107, pair_chunk=1024, uq_enabled=True):
    if min(batch_size, chunk, pair_chunk) < 1 or pair_budget < 0:
        raise ValueError('Positive chunks and nonnegative pair budget required')
    old = model.training; model.eval()
    device = next(model.parameters()).device
    labels, collected = [], {'image': [], 'text': []}
    path = None if output is None else Path(output)
    if path is not None and path.exists() and any(path.iterdir()):
        model.train(old)
        raise FileExistsError('Use a new evaluation directory')
    try:
        for xi, xt, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False):
            oi, ot = model(xi.to(device), xt.to(device)); labels.append(y.cpu())
            for name, o in (('image', oi), ('text', ot)):
                values = dict(z=o['z'].cpu(), class_probability=o['logits'].sigmoid().cpu())
                if model.evidence_task == 'category' and uq_enabled:
                    values.update(alpha=o['alpha'].cpu(), beta=o['beta'].cpu())
                collected[name].append(values)
        if not labels:
            raise ValueError('Empty evaluation data')
        y = torch.cat(labels)
        merged = {name: {k: torch.cat([v[k] for v in rows]) for k in rows[0]}
                  for name, rows in collected.items()}
        report = dict(evidence_task=model.evidence_task,
            retrieval=retrieval_map(merged['image']['z'], merged['text']['z'], y, chunk),
            common_classification={}, native_uncertainty={},
            protocol=dict(classifier='sigmoid(h @ W) in BOTH arms', no_confidence_filter=True,
                labels='complete multihot, no argmax', log_unit='nats',
                uq_status='evaluated' if uq_enabled else 'not_run_evidence_branch_not_supervised',
                scope='model-implied AU/EU; not ground-truth sources',
                warning='Category-label UQ and pair UQ are different targets; no cross-target scalar comparison'))
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
        for name, values in merged.items():
            report['common_classification'][name] = multilabel_probability_report(y.numpy(), values['class_probability'].numpy())
            if path is not None:
                np.savez_compressed(path/f'{name}_common_predictions.npz', labels=y.numpy(),
                                    z=values['z'].numpy(), probability=values['class_probability'].numpy())
            if model.evidence_task == 'category' and uq_enabled:
                stats = {k: v.numpy() for k, v in beta_statistics(values['alpha'], values['beta']).items()}
                summary, curves = classification_report(y.numpy(), stats)
                summary['unit'] = 'semantic_label_decision'
                report['native_uncertainty'][name] = summary
                if path is not None:
                    target = path/(name+'_category_uq'); target.mkdir()
                    np.savez_compressed(target/'predictions.npz', labels=y.numpy(),
                                        alpha=values['alpha'].numpy(), beta=values['beta'].numpy(), **stats)
                    write_curves(target, curves, 'label')
        if model.evidence_task == 'matching' and uq_enabled:
            ii, jj = uniform_eval_pairs(len(y), pair_budget, pair_seed)
            alphas, betas = [], []
            for start in range(0, len(ii), pair_chunk):
                i = torch.as_tensor(ii[start:start+pair_chunk]); j = torch.as_tensor(jj[start:start+pair_chunk])
                o = model.pair_head(merged['image']['z'][i].to(device), merged['text']['z'][j].to(device))
                alphas.append(o['alpha'].cpu()); betas.append(o['beta'].cpu())
            a, b = torch.cat(alphas), torch.cat(betas)
            truth = ((y[ii] > 0) & (y[jj] > 0)).any(1).numpy().astype(np.float32)[:, None]
            stats = {k: v.numpy() for k, v in beta_statistics(a, b).items()}
            summary, curves = classification_report(truth, stats)
            summary.update(unit='image_text_pair', n_pairs=len(ii), positive_pairs=int(truth.sum()),
                negative_pairs=int(len(ii)-truth.sum()), sampling='uniform_cartesian_without_replacement',
                pair_budget=pair_budget, pair_seed=pair_seed,
                pair_ids_sha256=hashlib.sha256(np.column_stack([ii, jj]).astype('<i8').tobytes()).hexdigest(),
                independent_pairs=False, labels_used_for_prediction=False)
            report['native_uncertainty']['matching'] = summary
            if path is not None:
                target = path/'matching_uq'; target.mkdir()
                np.savez_compressed(target/'predictions.npz', image_index=ii, text_index=jj,
                                    labels=truth, alpha=a.numpy(), beta=b.numpy(), **stats)
                write_curves(target, curves, 'pair')
        if path is not None:
            save_json(path/'metrics.json', report)
        return report
    finally:
        model.train(old)


def report_summary(report):
    out = dict(report['retrieval'])
    out['evidence_task'] = report['evidence_task']
    for key in ('nll', 'brier', 'ece_15bins', 'macro_ap'):
        values = [report['common_classification'][v][key] for v in ('image', 'text')]
        valid = [x for x in values if x is not None]
        out['common_'+key] = sum(valid)/len(valid) if valid else None
    # Native metrics are named by their task; NEVER merge label and matching NLL.
    for name, r in report['native_uncertainty'].items():
        prefix = 'pair' if name == 'matching' else name+'_category'
        out[prefix+'_nll'] = r['nll']; out[prefix+'_brier'] = r['brier']
        for score in ('epistemic_mi', 'aleatoric_entropy', 'vacuity'):
            out[prefix+'_'+score+'_aurc'] = r['uncertainty'][score]['aurc']
            out[prefix+'_'+score+'_auroc'] = r['uncertainty'][score]['error_detection']['auroc']
    return out
