"""End-to-end FUME + complete VLS transfer (without editing upstream files).

Example:
  python train_fume_vls.py --fume-dir /path/to/FUME/2025-CVPR-FUME \\
    --dataset INRIA --epochs 20 --checkpoint /path/to/encoder.pt \\
    --output /path/to/run_vls
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from vls_transfer import FUMEVLS, VLSConfig


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_repo(fume_dir: Path, dataset: str, device):
    fume_dir = fume_dir.resolve()
    if not all((fume_dir/n).exists() for n in ('model.py', 'options.py', 'processor.py')):
        raise FileNotFoundError('Expected original FUME model.py/options.py/processor.py in --fume-dir')
    sys.path.insert(0, str(fume_dir))
    old = os.getcwd()
    try:
        os.chdir(fume_dir)
        fmod = importlib.import_module('model')
        proc = importlib.import_module('processor')
        config, loaders = importlib.import_module('options').get_dataloader(dataset)
    finally:
        os.chdir(old)
    fume = fmod.FUME(device=device, img_input_dim=config['input_dim_I'],
                     text_input_dim=config['input_dim_T'], output_dim=config['class_number'],
                     num_class=config['class_number'], layer_num=config['layer_num'])
    return fume, proc.calc_loss, config, loaders


def load_encoder_or_run(model, checkpoint: Path, device):
    obj = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError('Expected a dictionary checkpoint')
    if 'fume_state' in obj:
        model.load_state_dict(obj['fume_state'], strict=True)
    elif 'state_dict' in obj:
        model.load_state_dict(obj['state_dict'], strict=True)
    else:
        raise ValueError('Unsupported checkpoint; no FUME encoder weights')
    if 'fixed_W' not in obj:
        raise ValueError('Checkpoint missing original FUME.W; cannot silently regenerate')
    model.W = obj['fixed_W'].to(device)


def map_all_queries(image: torch.Tensor, text: torch.Tensor,
                    labels: torch.Tensor, chunk: int = 128):
    """All unfiltered queries; no relevant candidate -> AP=0.

    Uses observed multilabel overlap, and includes paired gallery entries.
    mAP uses cosine of the same embeddings actually trained by FUME+VLS.
    """
    image = torch.nn.functional.normalize(image.float(), dim=1)
    text = torch.nn.functional.normalize(text.float(), dim=1)
    labels = (labels > 0).float()
    relevance = labels @ labels.T > 0
    def ap_direction(q, gallery):
        aps = []
        n = len(q)
        for a in range(0, n, chunk):
            s = q[a:a+chunk] @ gallery.T
            ind = torch.argsort(s, dim=1, descending=True, stable=True)
            rel = relevance[a:a+chunk].gather(1, ind).float()
            ranks = torch.arange(1, n+1, device=rel.device, dtype=rel.dtype)
            precision = rel.cumsum(1) / ranks
            total = relevance[a:a+chunk].sum(1).clamp(min=1)
            aps.append(((precision*rel).sum(1) / total).cpu())
        return torch.cat(aps).mean().item()
    return {'i2t_map_all': ap_direction(image, text),
            't2i_map_all': ap_direction(text, image)}


@torch.no_grad()
def evaluate_full(model, loader, device, eval_chunk):
    model.eval()
    vi, vt, yl = [], [], []
    ordered = DataLoader(loader.dataset,
                         batch_size=loader.batch_size or 128,
                         shuffle=False, drop_last=False, num_workers=0)
    for img, txt, labels in ordered:
        out = model.fume(img.to(device).float(), txt.to(device).float())
        vi.append(out['view1_feature'].detach().cpu())
        vt.append(out['view2_feature'].detach().cpu())
        yl.append(labels.detach().cpu())
    image, text, labels = torch.cat(vi), torch.cat(vt), torch.cat(yl)
    return map_all_queries(image, text, labels, chunk=eval_chunk)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fume-dir', type=Path, required=True)
    p.add_argument('--dataset', default='INRIA',
                   choices=['pascal','wiki','nus_deep','INRIA','xmedianet_deep'])
    p.add_argument('--checkpoint', type=Path, default=None, help='Existing encoder.pt with fixed_W')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--alpha-fume', type=float, default=None)
    p.add_argument('--head-hidden', type=int, default=128)
    p.add_argument('--mix-beta', type=float, default=10)
    p.add_argument('--noise-beta-clean', type=float, default=1)
    p.add_argument('--noise-beta-random', type=float, default=1)
    p.add_argument('--noise-std', type=float, default=1)
    p.add_argument('--negatives-per-anchor', type=int, default=1)
    p.add_argument('--clean-weight', type=float, default=0)
    p.add_argument('--vicinal-weight', type=float, default=1)
    p.add_argument('--noise-weight', type=float, default=1)
    p.add_argument('--vls-weight', type=float, default=1)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--eval-chunk', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--no-eval', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        raise ValueError('epochs >=1, batch-size>=2 and positive lr are required')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output directory must be new/empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    base, base_loss, upstream_config, loaders = build_repo(args.fume_dir, args.dataset, args.device)
    if args.checkpoint:
        load_encoder_or_run(base, args.checkpoint, args.device)
    vls_conf = VLSConfig(mix_beta=args.mix_beta,
                         noise_beta_clean=args.noise_beta_clean,
                         noise_beta_random=args.noise_beta_random,
                         noise_std=args.noise_std,
                         negatives_per_anchor=args.negatives_per_anchor,
                         clean_weight=args.clean_weight,
                         vicinal_weight=args.vicinal_weight,
                         noise_weight=args.noise_weight,
                         combined_weight=args.vls_weight)
    vls_conf.validate()
    model = FUMEVLS(base, embedding_dim=upstream_config['class_number'],
                    config=vls_conf, hidden=args.head_hidden).to(args.device)
    fuzzy_alpha = upstream_config['alpha'] if args.alpha_fume is None else args.alpha_fume
    train_loader = DataLoader(loaders['train'].dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    history = args.output / 'history.jsonl'
    with history.open('w', encoding='utf-8') as writer:
        for epoch in range(args.epochs):
            model.train()
            aggregate, count = {}, 0
            for img, txt, label in train_loader:
                img = img.to(args.device).float()
                txt = txt.to(args.device).float()
                label = label.to(args.device)
                opt.zero_grad(set_to_none=True)
                loss, detail = model.forward_losses(img, txt, label, base_loss, fuzzy_alpha)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                    raise FloatingPointError('Nonfinite FUME/VLS gradients')
                opt.step()
                count += len(label)
                for k, v in {**detail, 'total':loss.detach()}.items():
                    aggregate[k] = aggregate.get(k, 0.) + float(v) * len(label)
            if count != len(train_loader.dataset):
                raise RuntimeError('Some train rows were not consumed')
            record = {'epoch':epoch+1, 'training_samples': count,
                      **{k:v/count for k,v in aggregate.items()}}
            writer.write(json.dumps(record)+'\n'); writer.flush()
            print(json.dumps(record), flush=True)
    saved = dict(format='fume_vls_transfer_v1',fume_state={k:v.cpu() for k,v in model.fume.state_dict().items()},
                 fixed_W=model.fume.W.detach().cpu(),
                 evidence_head={k:v.cpu() for k,v in model.head.state_dict().items()},
                 vls_config=asdict(vls_conf),dataset=args.dataset,
                 classes=upstream_config['class_number'], seed=args.seed,
                 full_data_no_uncertainty_filter=True)
    torch.save(saved, args.output / 'vls_final.pt')
    metrics = None
    if not args.no_eval:
        metrics = evaluate_full(model, loaders['test'], args.device, args.eval_chunk)
        (args.output / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    cfg = {**vars(args), 'fume_dir':str(args.fume_dir),
           'checkpoint':str(args.checkpoint) if args.checkpoint else None,
           'output':str(args.output), 'vls_config':asdict(vls_conf),
           'dataset_config':{k: (v if isinstance(v,(str,int,float,bool)) else str(v))
                             for k,v in upstream_config.items()},
           'tested_once_after_training':not args.no_eval}
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({'saved':str(args.output/'vls_final.pt'),'full_unfiltered_map':metrics}, indent=2))


if __name__ == '__main__':
    main()