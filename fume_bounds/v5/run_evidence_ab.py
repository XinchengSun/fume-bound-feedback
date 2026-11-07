"""Category OR matching evidence comparisons, without changing legacy entrypoints.

Use compare_evidence.sh for the declared common-classifier A/B protocol.
Category+posterior reproduces the existing v5 objective; category+direction is an
explicit controlled ablation. Matching keeps ordinary multi-hot classification.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
if __package__:
    from .model import ModelConfig, warmstart_fume
    from .experiment import ExperimentLoss
    from .evidence_ab import PairConfig, build_model, ab_objective
    from .ab_metrics import evaluate_ab, report_summary
    from .data import load_data, sha256_file
    from .metrics import save_json
else:
    from model import ModelConfig, warmstart_fume
    from experiment import ExperimentLoss
    from evidence_ab import PairConfig, build_model, ab_objective
    from ab_metrics import evaluate_ab, report_summary
    from data import load_data, sha256_file
    from metrics import save_json

FORMAT = 'fume_v5_evidence_ab_v1'
DEFAULTS = dict(evidence_task='category', classification_source='posterior', evidence_mode='shared',
    pair_hidden=128, pair_negatives=1, pair_positive_mode='overlap', pair_chunk=128, pair_clean_weight=0.,
    batch_size=300, lr=5e-5, hidden=4096, layers=4, initial_evidence=2., seed=2,
    weight_decay=0., clip_grad=0., cls_weight=1., contrastive_weight=.05,
    vicinal_weight=1., noise_weight=1., kl_weight=.01, kl_anneal_epochs=10,
    consistency_weight=0., mix_beta=10., noise_beta=1., noise_multiplier=1.,
    image_noise_multiplier=1., text_noise_multiplier=1., temperature=1.,
    uq_pairs=20000, uq_seed=20261107)
LOSS_MAP = dict(cls_weight='classification', contrastive_weight='contrastive',
    vicinal_weight='vicinal', noise_weight='noise', kl_weight='error_kl',
    kl_anneal_epochs='kl_anneal_epochs', consistency_weight='consistency', mix_beta='mix_beta',
    noise_beta='noise_beta', noise_multiplier='noise_multiplier',
    image_noise_multiplier='image_noise_multiplier', text_noise_multiplier='text_noise_multiplier',
    temperature='temperature')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--classes', type=int); p.add_argument('--label-base', type=int, choices=(0, 1))
    p.add_argument('--epochs', type=int, default=30, help='Total epochs including resumed epochs')
    choices = dict(evidence_task=('category', 'matching'), classification_source=('posterior', 'direction'),
                   evidence_mode=('shared', 'per_label'), pair_positive_mode=('overlap', 'aligned'))
    for key, default in DEFAULTS.items():
        p.add_argument('--'+key.replace('_', '-'), type=type(default), choices=choices.get(key))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--val-every', type=int, default=1); p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--eval-batch-size', type=int, default=128); p.add_argument('--eval-chunk', type=int, default=128)
    p.add_argument('--eval-pair-chunk', type=int, default=1024)
    g=p.add_mutually_exclusive_group(); g.add_argument('--resume', type=Path); g.add_argument('--warmstart', type=Path)
    p.add_argument('--evaluate', choices=('val', 'test')); p.add_argument('--checkpoint', type=Path)
    return p


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def get_rng():
    n=np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
        numpy=(n[0],torch.from_numpy(n[1].astype(np.int64)),n[2],n[3],n[4]),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch'].cpu())
    n=state['numpy'];np.random.set_state((n[0],n[1].cpu().numpy().astype(np.uint32),n[2],n[3],n[4]))
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(path, state):
    temp=path.with_suffix('.tmp'); torch.save(state,temp);os.replace(temp,path)


def backbone_hash(model):
    h=hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name == 'W' or name.startswith(('img_net.', 'text_net.')):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def configurations(c):
    lc = ExperimentLoss(**{dest:c[src] for src,dest in LOSS_MAP.items()}); lc.validate()
    pc = PairConfig(c['pair_negatives'],c['pair_positive_mode'],c['pair_chunk'],c['pair_clean_weight']);pc.validate()
    for key in ('hidden','layers','batch_size','pair_hidden'):
        if type(c[key]) is not int or c[key] < 1: raise ValueError(f'{key} must be positive')
    if c['batch_size'] < 2 or not 0 <= c['seed'] < 2**32: raise ValueError('Invalid batch size or seed')
    for key in ('lr','initial_evidence','weight_decay','clip_grad'):
        if not math.isfinite(c[key]) or c[key] < 0: raise ValueError(f'Invalid {key}')
    if min(c['lr'],c['initial_evidence']) <= 0 or c['uq_pairs'] < 0 or c['uq_seed'] < 0:
        raise ValueError('Invalid learning rate, evidence or evaluation pair settings')
    if c['evidence_task']=='matching':
        if c['classification_source']!='direction' or c['evidence_mode']!='shared' or c['consistency_weight']:
            raise ValueError('Matching requires direction BCE, shared mode marker and consistency-weight=0')
    elif c['pair_clean_weight']:
        raise ValueError('Pair clean loss only applies to matching evidence')
    return lc, pc


def main(argv=None):
    args=parser().parse_args(argv)
    if min(args.epochs,args.threads,args.val_every,args.save_every,args.eval_batch_size,args.eval_chunk,args.eval_pair_chunk)<1:
        raise ValueError('Positive epochs, cadences and runtime sizes required')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output must be new/empty; no run will be overwritten')
    if bool(args.evaluate)!=bool(args.checkpoint) or (args.evaluate and (args.resume or args.warmstart)):
        raise ValueError('Evaluation needs checkpoint and is not resume/warmstart')
    torch.set_num_threads(args.threads)
    loaded=None
    if args.resume or args.checkpoint:
        loaded=torch.load(args.resume or args.checkpoint,map_location='cpu',weights_only=True)
        if loaded.get('format')!=FORMAT: raise ValueError('Not an evidence A/B checkpoint; legacy entrypoints are unchanged')
        if args.resume and loaded['kind']!='resume': raise ValueError('Resume requires last.pt with optimizer/RNG')
        if any(getattr(args,key) is not None for key in DEFAULTS):
            raise ValueError('Restore uses saved settings; no task/loss/training overrides allowed')
        c=loaded['run_config']
        if args.classes is not None and args.classes!=loaded['model_config']['classes']: raise ValueError('Class count differs')
        if args.label_base is not None and args.label_base!=c['label_base']: raise ValueError('Label encoding differs')
        classes=loaded['model_config']['classes']
    else:
        c={k:(getattr(args,k) if getattr(args,k) is not None else v) for k,v in DEFAULTS.items()}
        if c['evidence_task']=='matching' and args.classification_source is None:
            c['classification_source']='direction'
        c['label_base']=args.label_base;classes=args.classes
    lc,pc=configurations(c);seed_all(c['seed'])
    datasets,receipt=load_data(args.data,classes,c['label_base'])
    if (receipt['val']['image_sha256']==receipt['test']['image_sha256'] and
        receipt['val']['text_sha256']==receipt['test']['text_sha256']):
        raise ValueError('Validation equals test; no test-set tuning')
    mc=ModelConfig(receipt['image_dim'],receipt['text_dim'],receipt['classes'],c['hidden'],c['layers'],c['initial_evidence'])
    model=build_model(mc,c['evidence_task'],c['evidence_mode'],c['pair_hidden']).to(args.device)
    if loaded:
        if receipt['file_sha256']!=loaded['data_receipt']['file_sha256']: raise ValueError('Data SHA mismatch')
        if model.configuration()!=loaded['model_config']:raise ValueError('Model configuration mismatch')
        model.load_state_dict(loaded['model_state'],strict=True)
    elif args.warmstart:
        warmstart_fume(model,torch.load(args.warmstart,map_location='cpu',weights_only=True))
    args.output.mkdir(parents=True,exist_ok=True)

    def assess(split,output=None,epoch=1):
        state=get_rng()
        uq=bool(c['vicinal_weight'] or c['noise_weight'] or
                (c['evidence_task']=='category' and c['cls_weight'] and c['classification_source']=='posterior') or
                (c['evidence_task']=='matching' and c['pair_clean_weight']) or lc.kl_coefficient(epoch)>0)
        try:
            return evaluate_ab(model,datasets[split],output,batch_size=args.eval_batch_size,chunk=args.eval_chunk,
                pair_budget=c['uq_pairs'],pair_seed=c['uq_seed'],pair_chunk=args.eval_pair_chunk,uq_enabled=uq)
        finally:set_rng(state)

    if args.evaluate:
        report=assess(args.evaluate,args.output,loaded['epoch'])
        save_json(args.output/'evaluation_manifest.json',dict(format=FORMAT,split=args.evaluate,
            checkpoint_sha256=sha256_file(args.checkpoint),epoch=loaded['epoch'],run_config=c,
            data_sha256=receipt['file_sha256'],no_confidence_filter=True))
        print(json.dumps(report_summary(report),allow_nan=False));return
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=(.5,.999),weight_decay=c['weight_decay'])
    # Task-specific augmentation draws cannot change original-example minibatches.
    generator=torch.Generator().manual_seed(c['seed']+1)
    start=0;best={'map':None,'nll':None}
    if loaded:
        optimizer.load_state_dict(loaded['optimizer']);set_rng(loaded['rng'])
        generator.set_state(loaded['loader_rng'].cpu());start=loaded['epoch'];best=loaded['best']
    if start>=args.epochs:raise ValueError('Total epochs must exceed completed checkpoint epoch')
    source_names=('model.py','losses.py','experiment.py','data.py','metrics.py',
                  'evidence_ab.py','ab_metrics.py','run_evidence_ab.py')
    hashes={n:sha256_file(Path(__file__).parent/n) for n in source_names}
    save_json(args.output/'provenance.json',dict(format=FORMAT,model_config=model.configuration(),run_config=c,
        parameters=sum(p.numel() for p in model.parameters()),backbone_sha256=backbone_hash(model),
        source_sha256=hashes,data=receipt,torch=str(torch.__version__),device=args.device,
        fuzzy=False,category_evidence=c['evidence_task']=='category',matching_evidence=c['evidence_task']=='matching',
        input_paths=['clean','mixup','noise'],model_selection_split='val',test_used_for_tuning=False,
        warmstart=str(args.warmstart) if args.warmstart else None,
        classification_control='direction BCE matches both arms; posterior is the legacy category objective',
        runtime={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    loader=DataLoader(datasets['train'],batch_size=c['batch_size'],shuffle=True,drop_last=False,
                      generator=generator,num_workers=0)
    def checkpoint(epoch,kind,selected_by=None):
        state=dict(format=FORMAT,kind=kind,model_config=model.configuration(),model_state=model.state_dict(),
            run_config=c,data_receipt=receipt,epoch=epoch,best=best,selected_by=selected_by,source_sha256=hashes)
        if kind=='resume':state.update(optimizer=optimizer.state_dict(),rng=get_rng(),loader_rng=generator.get_state())
        return state
    save_json(args.output/'initial_validation.json',report_summary(assess('val',epoch=max(1,start))))
    with (args.output/'history.jsonl').open('w') as hf,(args.output/'validation.jsonl').open('w') as vf:
        for epoch in range(start+1,args.epochs+1):
            model.train();count=0;sums={};clock=time.perf_counter()
            for batch in loader:
                batch=tuple(t.to(args.device) for t in batch);optimizer.zero_grad(set_to_none=True)
                _,logs=ab_objective(model,batch,lc,receipt['train_noise_rms'],epoch=epoch,backward=True,
                    classification_source=c['classification_source'],pair_config=pc)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient; no optimizer update')
                if c['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_grad'],error_if_nonfinite=True)
                optimizer.step();n=len(batch[0]);count+=n
                for k,v in logs.items():sums[k]=sums.get(k,0.)+n*v
            if count!=len(datasets['train']):raise RuntimeError('Original training rows were dropped')
            if str(args.device).startswith('cuda'):torch.cuda.synchronize()
            row=dict(epoch=epoch,train_rows=count,seconds=time.perf_counter()-clock,
                     **{k:v/count for k,v in sums.items()})
            hf.write(json.dumps(row,allow_nan=False)+'\n');hf.flush();print(json.dumps(row),flush=True)
            if epoch%args.val_every==0 or epoch==args.epochs:
                summary=report_summary(assess('val',epoch=epoch));summary['epoch']=epoch
                vf.write(json.dumps(summary,allow_nan=False)+'\n');vf.flush()
                for key,value in (('map',summary['mean_map']),('nll',summary['common_nll'])):
                    better=best[key] is None or (value>best[key]['value'] if key=='map' else value<best[key]['value'])
                    if better:
                        best[key]={'value':value,'epoch':epoch}
                        atomic_save(args.output/f'best_{key}.pt',checkpoint(epoch,'inference',key))
            if epoch%args.save_every==0 or epoch==args.epochs:
                atomic_save(args.output/'last.pt',checkpoint(epoch,'resume'))
    save_json(args.output/'summary.json',dict(completed_epochs=args.epochs,best=best,run_config=c,
        test_evaluated=False,backbone_final_sha256=backbone_hash(model),
        warning='Native category and matching UQ have different targets; compare common metrics separately'))


if __name__=='__main__':main()
