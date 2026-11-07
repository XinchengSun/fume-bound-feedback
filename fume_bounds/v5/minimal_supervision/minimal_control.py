"""Read-only-to-old-files diagnostic: original FUME membership supervision minus necessity.

No BCE, evidence head, augmentation, confidence filtering, or checkpoint selection.
The encoder and fixed W are instantiated from the archived original model.py.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
ARCHIVE = V5 / 'diagnostics' / '20261107_ab150'
FORMAT = 'fume_minimal_no_necessity_v1'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def backbone_hash(model):
    # Same name/order/bytes convention as the archived run_evidence_ab.py.
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name == 'W' or name.startswith(('img_net.', 'text_net.')):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def order_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


def original_contrastive(zi, zt):
    """Original processor.py arithmetic, tau=1; no temperature/negative redesign."""
    batch = len(zi)
    all_fea = torch.cat([zi, zt])
    sim = all_fea.mm(all_fea.t()).exp()
    sim = sim - sim.diag().diag()
    s1 = sim[:, :batch] + sim[:, batch:]
    d1 = torch.cat([s1[:batch].diag(), s1[batch:].diag()])
    s2 = sim[:batch] + sim[batch:]
    d2 = torch.cat([s2[:, :batch].diag(), s2[:, batch:].diag()])
    return -(d1 / sim.sum(1)).log().mean() - (d2 / sim.sum(1)).log().mean()


def minimal_objective(model, image, text, labels):
    """Calls the original towers, NOT FUME.forward (which also computes c)."""
    if labels.ndim != 2 or not torch.isfinite(labels).all():
        raise ValueError('Finite [B,K] hard labels required')
    if not (((labels == 0) | (labels == 1)).all() and (labels.sum(1) == 1).all()):
        raise ValueError('This diagnostic is limited to single-positive-label INRIA; do not normalize multihot labels')
    zi, zt = model.img_net(image), model.text_net(text)
    w = model.W / torch.norm(model.W, p=2, dim=1, keepdim=True)
    mi, mt = torch.relu(zi.mm(w)), torch.relu(zt.mm(w))
    if mi.shape != labels.shape or mt.shape != labels.shape:
        raise ValueError('Label/membership shape mismatch')
    # Same Euclidean norm values as sqrt(sum(square)); vector_norm uses a finite
    # zero subgradient at exact equality, rather than sqrt(0)'s NaN backward.
    supervised = (torch.linalg.vector_norm(mi-labels.float(), dim=1).mean()
                  + torch.linalg.vector_norm(mt-labels.float(), dim=1).mean())
    contrastive = original_contrastive(zi, zt)
    total = supervised + 0.05 * contrastive
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite loss/encoder output; no optimizer step applied')
    return total, supervised, contrastive


def output_layer_probe(model, supervised, contrastive):
    """First-batch output-layer gradients only; no optimizer step and no full-tower claim."""
    weights = (model.img_net.layers[-1].weight, model.text_net.layers[-1].weight)
    ga = torch.autograd.grad(supervised, weights, retain_graph=True)
    gb = torch.autograd.grad(0.05 * contrastive, weights, retain_graph=True)
    out = {}
    for side, a, b in zip(('image', 'text'), ga, gb):
        na, nb = a.norm(), b.norm()
        cosine = float((a*b).sum()/(na*nb)) if na > 0 and nb > 0 else None
        out[side] = dict(supervision_grad_norm=float(na), weighted_cl_grad_norm=float(nb),
                         cosine=cosine, parameter='layers[-1].weight')
    return out


@torch.no_grad()
def evaluate(model, dataset, batch_size=128, chunk=128):
    """Full-query/full-gallery cosine AP; preserve RNG and mode during validation."""
    state, old_mode = torch.get_rng_state(), model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        images, texts, labels = [], [], []
        for xi, xt, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            images.append(model.img_net(xi.to(device)).cpu())
            texts.append(model.text_net(xt.to(device)).cpu())
            labels.append(y)
        image, text, y = torch.cat(images), torch.cat(texts), torch.cat(labels).float()
        if not torch.isfinite(image).all() or not torch.isfinite(text).all():
            raise FloatingPointError('Nonfinite evaluation embedding')
        image, text = F.normalize(image.float(), dim=-1), F.normalize(text.float(), dim=-1)
        def direction(q, g):
            aps = []
            for start in range(0, len(q), chunk):
                scores = q[start:start+chunk] @ g.T
                rel = (y[start:start+chunk] @ y.T) > 0
                order = torch.argsort(scores, descending=True, dim=1, stable=True)
                ranked = rel.gather(1, order).float()
                ranks = torch.arange(1, len(g)+1)
                aps.append(((ranked.cumsum(1)/ranks*ranked).sum(1)/rel.sum(1).clamp_min(1)))
            return float(torch.cat(aps).mean())
        i2t, t2i = direction(image, text), direction(text, image)
        return dict(i2t_map_all=i2t, t2i_map_all=t2i, mean_map=(i2t+t2i)/2,
                    queries=len(y), candidates=len(y), paired_items_included=True,
                    confidence_filter=False)
    finally:
        model.train(old_mode)
        torch.set_rng_state(state)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def save_checkpoint(path, model, optimizer, epoch, generator, config, receipt):
    temp = path.with_suffix('.tmp')
    torch.save(dict(format=FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch, loader_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    config=config, data_receipt=receipt), temp)
    os.replace(temp, path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--classes', type=int, default=100)
    p.add_argument('--label-base', type=int, choices=(0,1))
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--require-ab150', action='store_true', help='Require original data/init and every epoch order hash')
    p.add_argument('--resume', type=Path, help='Own last.pt only; all training settings must match')
    p.add_argument('--final-test', action='store_true', help='Evaluate test once at requested final epoch, never select best')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.classes, args.epochs, args.batch_size, args.layers, args.threads, args.save_every) < 1:
        raise ValueError('Positive dimensions/counts required')
    if args.classes < 2 or args.batch_size < 2 or not 0 <= args.seed < 2**32:
        raise ValueError('classes/batch >=2 and valid NumPy seed required')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Finite positive lr required')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('New/empty output directory required; no old run may be overwritten')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    source = ARCHIVE/'training_source'/'model.py'
    legacy = load_module(source, '_minimal_archived_fume')
    data_api = load_module(V5/'data.py', '_minimal_existing_data')
    datasets, receipt = data_api.load_data(args.data, args.classes, args.label_base)
    if any(not (ds.tensors[2].sum(1) == 1).all() for ds in datasets.values()):
        raise ValueError('Single-positive-label diagnostic only; multihot data are not silently changed')
    model = legacy.FUME('cpu', receipt['image_dim'], receipt['text_dim'],
                        receipt['classes'], receipt['classes'], args.layers)
    w = model.W; del model.W; model.register_buffer('W', w)
    model = model.to(args.device)
    initial_hash = backbone_hash(model)
    config = dict(classes=args.classes, layers=args.layers, batch_size=args.batch_size, lr=args.lr,
                  seed=args.seed, label_base=args.label_base, require_ab150=args.require_ab150)
    reference = None
    if args.require_ab150:
        reference = json.loads((ARCHIVE/'acceptance150.json').read_text())
        if (args.seed, args.layers, args.batch_size, args.lr) != (2,4,300,5e-5):
            raise ValueError('AB150 comparison requires seed2/layers4/batch300/lr5e-5')
        if receipt['file_sha256'] != reference['data']['file_sha256']:
            raise ValueError('AB150 data hash mismatch')
        if initial_hash != reference['initial_backbone_and_W_sha256']:
            raise ValueError('AB150 initialization hash mismatch; do not call this matched initialization')
        if sha256(source) != reference['legacy_source']['model.py']:
            raise ValueError('Archived original model source changed')
        reference_orders = [json.loads(s) for s in (ARCHIVE/'runs/A_fume/order.jsonl').read_text().splitlines()]
        if args.epochs > len(reference_orders): raise ValueError('Reference order records exhausted')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.5,.999), weight_decay=0)
    generator = torch.Generator().manual_seed(args.seed+1)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved.get('format') != FORMAT or saved['config'] != config:
            raise ValueError('Resume requires this format and identical declared training settings')
        if saved['data_receipt']['file_sha256'] != receipt['file_sha256']:
            raise ValueError('Resume data changed')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['loader_rng']); torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng']:
            if not torch.cuda.is_available() or len(saved['cuda_rng']) != torch.cuda.device_count():
                raise ValueError('CUDA resume requires same device count')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start = saved['epoch']
    if start >= args.epochs: raise ValueError('Requested total epochs must exceed checkpoint epoch')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'provenance.json', dict(format=FORMAT, config=config, data=receipt,
        initial_backbone_and_W_sha256=initial_hash, archive_commit='6de422a322eb5b47db7b5f5be2b81c40b55c1d16',
        source_sha256={str(p.relative_to(V5)): sha256(p) for p in (source,V5/'data.py',Path(__file__))},
        torch=str(torch.__version__), numpy=str(np.__version__), device=args.device,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        original_fume_forward_called=False, necessity=False, fuzzy_c=False, evidence=False,
        mixup=False, noise=False, fixed_cl_weight=.05, fixed_temperature=1.,
        final_test_requested=args.final_test, resume=str(args.resume) if args.resume else None))
    write_json(args.output/'initial_validation.json', evaluate(model,datasets['val']))
    current_epoch = start
    class LoggedSampler(RandomSampler):
        def __iter__(self):
            ids = list(super().__iter__())
            row = dict(epoch=current_epoch, order=order_hash(ids), first_batch=order_hash(ids[:args.batch_size]))
            if reference is not None and row != reference_orders[current_epoch-1]:
                raise ValueError(f'AB150 sample order mismatch at epoch {current_epoch}')
            with (args.output/'order.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            return iter(ids)
    loader = DataLoader(datasets['train'], batch_size=args.batch_size,
        sampler=LoggedSampler(datasets['train'], generator=generator), generator=generator,
        drop_last=False, num_workers=0)
    with (args.output/'history.jsonl').open('w') as hist, (args.output/'validation.jsonl').open('w') as val:
        for current_epoch in range(start+1,args.epochs+1):
            model.train(); count=0; sums=np.zeros(3); clock=time.perf_counter()
            for step,(xi,xt,y) in enumerate(loader):
                xi,xt,y=(t.to(args.device) for t in (xi,xt,y))
                optimizer.zero_grad(set_to_none=True)
                total, sup, cl = minimal_objective(model,xi,xt,y)
                if step == 0:
                    probe=output_layer_probe(model,sup,cl)
                    with (args.output/'gradient_probe.jsonl').open('a') as f:
                        f.write(json.dumps(dict(epoch=current_epoch,batch=1,output_layer_only=True,**probe),allow_nan=False)+'\n')
                total.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient; optimizer step not applied')
                optimizer.step(); n=len(y);count+=n
                sums+=n*np.array([float(total.detach()),float(sup.detach()),float(cl.detach())])
            if count != len(datasets['train']): raise RuntimeError('Training rows dropped')
            if str(args.device).startswith('cuda'): torch.cuda.synchronize()
            means=sums/count
            row=dict(epoch=current_epoch,train_rows=count,total=means[0],supervision=means[1],
                contrastive=means[2],weighted_contrastive=.05*means[2],lr=optimizer.param_groups[0]['lr'],
                train_seconds=time.perf_counter()-clock)
            hist.write(json.dumps(row,allow_nan=False)+'\n');hist.flush()
            report=dict(epoch=current_epoch,**evaluate(model,datasets['val']))
            val.write(json.dumps(report,allow_nan=False)+'\n');val.flush()
            print(json.dumps(dict(**row,val_mean_map=report['mean_map']),allow_nan=False),flush=True)
            if current_epoch%args.save_every == 0 or current_epoch == args.epochs:
                save_checkpoint(args.output/'last.pt',model,optimizer,current_epoch,generator,config,receipt)
    if args.final_test: write_json(args.output/'test_metrics.json',evaluate(model,datasets['test']))
    write_json(args.output/'completion.json',dict(completed_epochs=args.epochs,
        total_updates=math.ceil(len(datasets['train'])/args.batch_size)*args.epochs,
        fixed_final_epoch=True,test_evaluated=args.final_test,no_performance_claim=True))


if __name__ == '__main__':
    main()
