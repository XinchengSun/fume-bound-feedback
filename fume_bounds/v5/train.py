"""Train v5 from original INRIA MAT or canonical NPZ. See README.md."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
if __package__:
    from .model import ModelConfig, MultiLabelV5, warmstart_fume
    from .losses import LossConfig, objective
    from .data import load_data, sha256_file
    from .metrics import evaluate, save_json
else:
    from model import ModelConfig, MultiLabelV5, warmstart_fume
    from losses import LossConfig, objective
    from data import load_data, sha256_file
    from metrics import evaluate, save_json

UPSTREAM = 'siyuancncd/FUME@5dd7a306aa505d8f60265b0353ff98de19ad9815'
FORMAT = 'fume_v5_multilabel_unit_prior_v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    n = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=(n[0], torch.from_numpy(n[1].astype(np.int64)), n[2], n[3], n[4]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed; exact RNG resume is unavailable')
        torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda']])


def save_checkpoint(path, model, optimizer, epoch, config, receipt, scales):
    state = dict(format=FORMAT, model_config=model.configuration(),
                 model_state=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                 run_config=config, data_receipt=receipt, noise_scales=list(scales), rng=rng_state())
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--classes', type=int)
    p.add_argument('--label-base', type=int, choices=(0, 1))
    p.add_argument('--epochs', type=int, default=100, help='Total epoch count, including resumed epochs')
    p.add_argument('--batch-size', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--hidden', type=int, default=4096)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--initial-evidence', type=float, default=2.0)
    p.add_argument('--cls-weight', type=float, default=1.0)
    p.add_argument('--vicinal-weight', type=float, default=1.0)
    p.add_argument('--noise-weight', type=float, default=1.0)
    p.add_argument('--contrastive-weight', type=float, default=0.05)
    p.add_argument('--mix-beta', type=float, default=10.0)
    p.add_argument('--noise-beta', type=float, default=1.0)
    p.add_argument('--noise-multiplier', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--clip-grad', type=float, default=0.0, help='0 disables clipping (upstream default)')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--eval-chunk', type=int, default=128)
    p.add_argument('--no-test', action='store_true', help='Save checkpoint without final test evaluation')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--resume', type=Path, help='v5 checkpoint; restore optimizer/RNG and saved hyperparameters')
    group.add_argument('--warmstart', type=Path, help='Explicit FUME encoder warm start, NOT a from-scratch baseline')
    return p


def main(argv=None):
    import sys
    actual = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(actual)
    if min(args.epochs, args.threads, args.eval_chunk, args.eval_batch_size) < 1:
        raise ValueError('Positive epochs/threads/evaluation sizes required')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output directory must be new/empty; never overwrite a run')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=True)
        if resumed.get('format') != FORMAT:
            raise ValueError('Not a v5 checkpoint; use --warmstart only for explicit FUME transfer')
        fixed = {'batch-size', 'lr', 'hidden', 'layers', 'initial-evidence', 'cls-weight',
                 'vicinal-weight', 'noise-weight', 'contrastive-weight', 'mix-beta', 'noise-beta',
                 'noise-multiplier', 'temperature', 'clip-grad', 'seed'}
        if any(a.split('=')[0] in {'--'+k for k in fixed} for a in actual):
            raise ValueError('Resume restores all saved training settings; do not pass training overrides')
        cfg = resumed['run_config']
        classes = resumed['model_config']['classes']
        label_base = cfg['label_base']
    else:
        cfg = dict(batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, layers=args.layers,
                   initial_evidence=args.initial_evidence, seed=args.seed, clip_grad=args.clip_grad,
                   label_base=args.label_base, loss=asdict(LossConfig(args.cls_weight, args.vicinal_weight,
                   args.noise_weight, args.contrastive_weight, args.mix_beta, args.noise_beta,
                   args.temperature, args.noise_multiplier)))
        classes, label_base = args.classes, args.label_base
    if cfg['batch_size'] < 2 or not np.isfinite(cfg['lr']) or cfg['lr'] <= 0:
        raise ValueError('batch-size >=2 and finite positive learning rate required')
    if not np.isfinite(cfg['clip_grad']) or cfg['clip_grad'] < 0:
        raise ValueError('clip-grad must be finite and nonnegative')
    loss_config = LossConfig(**cfg['loss'])
    loss_config.validate()
    datasets, receipt = load_data(args.data, classes, label_base)
    config = ModelConfig(receipt['image_dim'], receipt['text_dim'], receipt['classes'],
                         cfg['hidden'], cfg['layers'], cfg['initial_evidence'])
    model = MultiLabelV5(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.5, 0.999), weight_decay=0)
    start = 0
    scales = receipt['train_noise_rms']
    if resumed:
        if resumed['data_receipt']['file_sha256'] != receipt['file_sha256']:
            raise ValueError('Resume data SHA differs; no silent data change')
        if resumed['model_config'] != model.configuration():
            raise ValueError('Resume architecture mismatch')
        model.load_state_dict(resumed['model_state'], strict=True)
        optimizer.load_state_dict(resumed['optimizer'])
        scales = resumed['noise_scales']
        start = resumed['epoch']
        restore_rng(resumed['rng'])
    elif args.warmstart:
        warmstart_fume(model, torch.load(args.warmstart, map_location='cpu', weights_only=True))
    if start >= args.epochs:
        raise ValueError('--epochs must exceed the checkpoint epoch')
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = ('model.py', 'losses.py', 'data.py', 'metrics.py', 'train.py')
    provenance = dict(format=FORMAT, upstream=UPSTREAM, model_config=model.configuration(),
        run_config=cfg, data=receipt, parameters=sum(p.numel() for p in model.parameters()),
        source_sha256={n: sha256_file(Path(__file__).parent/n) for n in source_files},
        torch=str(torch.__version__), numpy=str(np.__version__), device=args.device,
        resume=str(args.resume) if args.resume else None,
        warmstart=str(args.warmstart) if args.warmstart else None,
        warmstart_sha256=sha256_file(args.warmstart) if args.warmstart else None,
        checkpoint_policy='fixed_last_epoch', fuzzy_branch=False, paired_evidence_head=False,
        shared_evidence_budget=True, unit_prior=True, confidence_filter=False,
        caveat='Reused defaults are starting points, not retuned v5 hyperparameters')
    save_json(args.output/'provenance.json', provenance)
    loader = DataLoader(datasets['train'], batch_size=cfg['batch_size'], shuffle=True,
                        drop_last=False, num_workers=0)
    with (args.output/'history.jsonl').open('w', encoding='utf8') as history:
        for epoch in range(start, args.epochs):
            model.train()
            count, sums, t0 = 0, {}, time.perf_counter()
            for batch in loader:
                batch = tuple(t.to(args.device) for t in batch)
                optimizer.zero_grad(set_to_none=True)
                _, logs = objective(model, batch, loss_config, scales, backward=True)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient; optimizer step NOT applied')
                if cfg['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'], error_if_nonfinite=True)
                optimizer.step()
                n = len(batch[0]); count += n
                for k, value in logs.items():
                    sums[k] = sums.get(k, 0.0)+n*value
            if count != len(datasets['train']):
                raise RuntimeError('Training rows were skipped')
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            row = dict(epoch=epoch+1, train_rows=count, seconds=time.perf_counter()-t0,
                       **{k: v/count for k, v in sums.items()})
            history.write(json.dumps(row, allow_nan=False)+'\n'); history.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
            save_checkpoint(args.output/'last.pt', model, optimizer, epoch+1, cfg, receipt, scales)
    # No validation/test-set checkpoint selection. Test is evaluated once at the end.
    if not args.no_test:
        report = evaluate(model, datasets['test'], args.output/'test', args.eval_batch_size, args.eval_chunk)
        print(json.dumps(report['retrieval'], allow_nan=False), flush=True)
    save_json(args.output/'completion.json', dict(completed_epochs=args.epochs,
        checkpoint=str(args.output/'last.pt'), test_evaluated=not args.no_test,
        no_claim_of_real_data_uq_identification=True))


if __name__ == '__main__':
    main()
