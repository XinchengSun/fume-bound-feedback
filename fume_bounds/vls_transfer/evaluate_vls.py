"""Evaluate a trained binary-matching VLS head without retraining.

Retrieval: all queries/candidates, no uncertainty filtering.
UQ: uniform fixed-seed pairs from the Cartesian test pool (not class balanced).
Optional OOD inputs must be actual user-supplied out-of-distribution pairs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from uncertainty import SCORES, binary_report, dirichlet_statistics, ood_report


def _digest(array) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding='utf-8')


def sample_pairs(n: int, budget: int, seed: int):
    if n < 1 or budget < 0:
        raise ValueError('n > 0 and pair budget >= 0 required')
    total = n*n
    flat = (np.arange(total, dtype=np.int64) if budget == 0 or budget >= total else
            np.random.default_rng(seed).choice(total, budget, replace=False))
    flat.sort()
    return flat//n, flat % n


@torch.no_grad()
def _head_scores(head, image, text, i, j, device, chunk):
    alpha = []
    for start in range(0, len(i), chunk):
        zi = image[torch.as_tensor(i[start:start+chunk])].to(device)
        zt = text[torch.as_tensor(j[start:start+chunk])].to(device)
        logits = head(zi, zt)
        alpha.append((1+torch.nn.functional.softplus(logits.double())).cpu())
    alpha = torch.cat(alpha)
    stats = {k:v.numpy() for k,v in dirichlet_statistics(alpha).items()}
    stats['alpha'] = alpha.numpy()
    return stats


@torch.no_grad()
def collect_embeddings(model, loader, device):
    data = DataLoader(loader.dataset, batch_size=loader.batch_size or 128,
                      shuffle=False, drop_last=False, num_workers=0)
    vi, vt, ys = [], [], []
    for img, txt, labels in data:
        out = model.fume(img.to(device).float(), txt.to(device).float())
        vi.append(out['view1_feature'].cpu())
        vt.append(out['view2_feature'].cpu())
        ys.append(labels.cpu())
    if not vi:
        raise ValueError('Empty evaluation dataset')
    vi, vt, ys = torch.cat(vi), torch.cat(vt), torch.cat(ys)
    if not all(torch.isfinite(t).all() for t in (vi,vt,ys)):
        raise ValueError('Nonfinite evaluation data or embeddings')
    if not torch.isin(ys, torch.tensor([0,1],dtype=ys.dtype)).all():
        raise ValueError('Expected hard multilabel/one-hot evaluation labels')
    return vi,vt,ys


def save_assessment(head, image, text, labels, output, device='cpu',
                    pair_budget=20000, seed=20261105, chunk=1024):
    if chunk < 1:
        raise ValueError('Pair chunk must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    i, j = sample_pairs(len(labels), pair_budget, seed)
    stats = _head_scores(head,image,text,i,j,device,chunk)
    truth = ((labels[i] > 0) & (labels[j] > 0)).any(1).numpy().astype(np.int8)
    report, curves = binary_report(truth,stats['mean_probability'][:,0],
                                    {k:stats[k] for k in SCORES})
    np.savez_compressed(output/'pair_predictions.npz', image_index=i,text_index=j,
                        relevance=truth, **stats)
    for name, curve in curves.items():
        # All pairs contribute to exact AURC; CSV is a <=1000 point display grid.
        ix = np.unique(np.linspace(0,len(i)-1,min(len(i),1000),dtype=int))
        with (output/f'risk_coverage_{name}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['coverage','classification_error_risk'])
            w.writerows(zip(curve['coverage'][ix],curve['risk'][ix]))
    report['protocol'] = dict(
        version='vls_uq_v2', target='binary image-text relevance', class_order=['match','nonmatch'],
        positive_rule='shared observed label', sampling='uniform_without_replacement',
        total_available_pairs=len(labels)**2, evaluated_pairs=len(i),
        full_cartesian_pairs=len(i)==len(labels)**2, pair_budget=pair_budget, seed=seed,
        pair_ids_sha256=_digest(np.column_stack([i,j])), labels_sha256=_digest(labels.numpy()),
        log_unit='nats', match_threshold=.5, nll_clip=1e-12,
        ties='risk-coverage averages random order inside exact uncertainty ties',
        filtering='No confidence filtering. UQ sampling does not change retrieval mAP.',
        interpretation='Model-implied AU/EU, not ground-truth decomposition; no iid-pair CI claimed.')
    _save_json(output/'uncertainty_report.json',report)
    return report, stats


def evaluate_model(model, loader, device, output, *, pair_budget=20000,
                   seed=20261105, chunk=1024, retrieval_chunk=128, ood_pairs=None):
    from train_fume_vls import map_all_queries
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names=['uncertainty_report.json','pair_predictions.npz','test_metrics.json']
    if any((output/n).exists() for n in names):
        raise FileExistsError(f'Evaluation output already exists: {output}')
    training = model.training
    model.eval()
    try:
        image,text,labels=collect_embeddings(model,loader,device)
        metrics=map_all_queries(image,text,labels,chunk=retrieval_chunk)
        _save_json(output/'test_metrics.json',metrics)
        report,id_stats=save_assessment(model.head,image,text,labels,output,device,
                                        pair_budget,seed,chunk)
        ood=None
        if ood_pairs is not None:
            # NPZ rows are actual pairs, not an artificial Cartesian pairing.
            with np.load(ood_pairs,allow_pickle=False) as d:
                xi,xt=d['image_features'],d['text_features']
                if xi.ndim!=2 or xt.ndim!=2 or len(xi)!=len(xt) or len(xi)==0:
                    raise ValueError('OOD requires nonempty paired feature matrices')
                if not np.isfinite(xi).all() or not np.isfinite(xt).all():
                    raise ValueError('OOD features must be finite')
                vi,vt=[],[]
                with torch.no_grad():
                    for a in range(0,len(xi),chunk):
                        vi.append(model.fume.img_net(torch.as_tensor(xi[a:a+chunk],device=device).float()).cpu())
                        vt.append(model.fume.text_net(torch.as_tensor(xt[a:a+chunk],device=device).float()).cpu())
                ii=np.arange(len(xi))
                od=_head_scores(model.head,torch.cat(vi),torch.cat(vt),ii,ii,device,chunk)
                np.savez_compressed(output/'ood_predictions.npz',**od)
                ood=ood_report(id_stats,od)
                ood['protocol']=dict(source=str(ood_pairs), image_sha256=_digest(xi),
                                     text_sha256=_digest(xt),
                                     warning='User declares genuine OOD. Pair sampling must be comparable to ID; no synthetic-noise OOD claim.')
                _save_json(output/'ood_report.json',ood)
        result=dict(retrieval=metrics, uncertainty=report,
                    ood_status='evaluated_user_supplied_pairs' if ood else 'not_run_no_ood_data',
                    scope='No claim of ground-truth AU/EU identification')
        _save_json(output/'evaluation_summary.json',result)
        return result
    finally:
        model.train(training)


def main(argv=None):
    from train_fume_vls import build_repo, load_encoder_or_run
    from vls_transfer import FUMEVLS, VLSConfig
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fume-dir',type=Path,required=True)
    p.add_argument('--dataset',default='INRIA')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--uq-pairs',type=int,default=20000,help='0 = all Cartesian pairs; can be expensive')
    p.add_argument('--uq-seed',type=int,default=20261105)
    p.add_argument('--uq-chunk',type=int,default=1024)
    p.add_argument('--eval-chunk',type=int,default=128)
    p.add_argument('--ood-pairs',type=Path,default=None)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args(argv)
    if a.threads<1 or a.uq_pairs<0 or min(a.uq_chunk,a.eval_chunk)<1:
        p.error('Positive chunks/threads and nonnegative UQ pair count required')
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new evaluation output directory')
    torch.set_num_threads(a.threads)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if 'evidence_head' not in saved:
        raise ValueError('Encoder-only checkpoint has NO trained VLS head; cannot evaluate it as VLS')
    if saved.get('evidence_head_trained') is False:
        raise ValueError('This checkpoint records that the evidence head was not trained')
    if saved.get('dataset',a.dataset)!=a.dataset:
        raise ValueError('Checkpoint dataset differs from requested ID dataset')
    base,_,cfg,loaders=build_repo(a.fume_dir,a.dataset,a.device)
    load_encoder_or_run(base,a.checkpoint,a.device)
    hidden=saved['evidence_head']['net.0.weight'].shape[0]
    model=FUMEVLS(base,cfg['class_number'],VLSConfig(**saved['vls_config']),hidden).to(a.device)
    model.head.load_state_dict(saved['evidence_head'],strict=True)
    result=evaluate_model(model,loaders['test'],a.device,a.output,pair_budget=a.uq_pairs,
                           seed=a.uq_seed,chunk=a.uq_chunk,retrieval_chunk=a.eval_chunk,
                           ood_pairs=a.ood_pairs)
    _save_json(a.output/'checkpoint_manifest.json',dict(
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        checkpoint=str(a.checkpoint),dataset=a.dataset,
        caveat='Old v1 checkpoints lack an explicit evidence-head-trained flag' if 'evidence_head_trained' not in saved else None))
    print(json.dumps(dict(output=str(a.output),retrieval=result['retrieval'],
                          pairs=result['uncertainty']['n_pairs']),indent=2))


if __name__=='__main__':
    main()
