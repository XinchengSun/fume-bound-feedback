"""Validation-first v5 experiments; legacy train.py and checkpoints are untouched.

Train: python run_experiment.py --data DATA --output NEW --preset coarse
Test:  python run_experiment.py --data DATA --output NEW --evaluate test --checkpoint BEST
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
if __package__:
    from .experiment import ExperimentLoss, ExperimentModel, experiment_objective
    from .model import ModelConfig, warmstart_fume
    from .data import load_data, sha256_file
    from .metrics import evaluate, save_json
else:
    from experiment import ExperimentLoss, ExperimentModel, experiment_objective
    from model import ModelConfig, warmstart_fume
    from data import load_data, sha256_file
    from metrics import evaluate, save_json

FORMAT = 'fume_v5_experiment_v1'
PRESETS = {'control': dict(vicinal=0., noise=0., error_kl=0.),
           'v5': dict(error_kl=0.), 'coarse': dict(error_kl=.01)}
# CLI defaults below are explicit experimental starting points, NOT retuned optima.
TRAIN_DEFAULTS = dict(batch_size=300, lr=5e-5, hidden=4096, layers=4,
    initial_evidence=2., evidence_mode='shared', weight_decay=0., clip_grad=0.,
    seed=2, scheduler='none', schedule_epochs=100, min_lr_ratio=.1,
    selection='map', patience=0, min_delta=0.)
LOSS_FLAGS = {'cls_weight':'classification', 'vicinal_weight':'vicinal',
    'noise_weight':'noise', 'contrastive_weight':'contrastive', 'kl_weight':'error_kl',
    'kl_anneal_epochs':'kl_anneal_epochs', 'consistency_weight':'consistency',
    'mix_beta':'mix_beta', 'noise_beta':'noise_beta', 'temperature':'temperature',
    'noise_multiplier':'noise_multiplier', 'image_noise_multiplier':'image_noise_multiplier',
    'text_noise_multiplier':'text_noise_multiplier'}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    n = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
        numpy=(n[0],torch.from_numpy(n[1].astype(np.int64)),n[2],n[3],n[4]),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    torch.set_rng_state(s['torch'].cpu()); random.setstate(s['python'])
    n=s['numpy']; np.random.set_state((n[0],n[1].cpu().numpy().astype(np.uint32),n[2],n[3],n[4]))
    if s['cuda']:
        if not torch.cuda.is_available() or len(s['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact RNG resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all([t.cpu() for t in s['cuda']])


def isolated_evaluate(model, dataset, output=None, batch_size=128, chunk=128):
    """Evaluation's DataLoader must not advance the training random stream."""
    state=rng_state()
    try: return evaluate(model,dataset,output,batch_size,chunk)
    finally: restore_rng(state)


def report_summary(report):
    c=report['classification']
    out=dict(report['retrieval'])
    for key in ('brier','nll','ece_15bins','label_accuracy','macro_ap'):
        vals=[c[v][key] for v in ('image','text') if c[v][key] is not None]
        out['mean_'+key]=sum(vals)/len(vals) if vals else None
    for v in ('image','text'):
        out[v+'_nll']=c[v]['nll']; out[v+'_brier']=c[v]['brier']
        for name in ('epistemic_mi','aleatoric_entropy','vacuity'):
            q=c[v]['uncertainty'][name]
            out[v+'_'+name+'_aurc']=q['aurc']
            out[v+'_'+name+'_auroc']=q['error_detection']['auroc']
    return out


def learning_rate(config, epoch):
    if config['scheduler']=='none': return config['lr']
    horizon=config['schedule_epochs']
    fraction=min(max(epoch-1,0),horizon-1)/(horizon-1)
    return config['lr']*(config['min_lr_ratio']+(1-config['min_lr_ratio'])*.5*(1+math.cos(math.pi*fraction)))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--classes',type=int);p.add_argument('--label-base',type=int,choices=(0,1))
    p.add_argument('--preset',choices=tuple(PRESETS))
    p.add_argument('--epochs',type=int,default=30,help='Total desired epochs, including resume')
    for name in ('batch_size','hidden','layers','seed','schedule_epochs','patience'):
        p.add_argument('--'+name.replace('_','-'),type=int)
    for name in ('lr','initial_evidence','weight_decay','clip_grad','min_lr_ratio','min_delta'):
        p.add_argument('--'+name.replace('_','-'),type=float)
    p.add_argument('--evidence-mode',choices=('shared','per_label'))
    p.add_argument('--scheduler',choices=('none','cosine'))
    p.add_argument('--selection',choices=('map','nll'),help='Early-stop/selection metric; both best files are tracked')
    for name in LOSS_FLAGS:
        p.add_argument('--'+name.replace('_','-'),type=int if name=='kl_anneal_epochs' else float)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--val-every',type=int,default=1)
    p.add_argument('--save-every',type=int,default=5,help='Full optimizer checkpoint cadence; final always saved')
    p.add_argument('--eval-batch-size',type=int,default=128)
    p.add_argument('--eval-chunk',type=int,default=128)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--resume',type=Path)
    g.add_argument('--warmstart',type=Path,help='Explicit FUME encoder transfer, not a from-scratch experiment')
    p.add_argument('--evaluate',choices=('val','test'))
    p.add_argument('--checkpoint',type=Path,help='Required with --evaluate; experiment checkpoints only')
    return p


def resolved_config(args):
    cfg={k:(getattr(args,k) if getattr(args,k) is not None else v) for k,v in TRAIN_DEFAULTS.items()}
    cfg['preset']=args.preset or 'coarse'
    lc=asdict(ExperimentLoss()); lc.update(PRESETS[cfg['preset']])
    for flag,key in LOSS_FLAGS.items():
        if getattr(args,flag) is not None: lc[key]=getattr(args,flag)
    cfg.update(loss=lc,label_base=args.label_base)
    return cfg


def validate_config(cfg):
    for k in ('batch_size','hidden','layers','schedule_epochs'):
        if type(cfg[k]) is not int or cfg[k]<1: raise ValueError(f'Positive integer {k} required')
    if cfg['batch_size']<2 or cfg['schedule_epochs']<2: raise ValueError('batch-size and schedule-epochs >=2 required')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32: raise ValueError('seed outside NumPy range')
    if type(cfg['patience']) is not int or cfg['patience']<0: raise ValueError('patience must be nonnegative')
    for k in ('lr','initial_evidence','weight_decay','clip_grad','min_lr_ratio','min_delta'):
        if not math.isfinite(cfg[k]) or cfg[k]<0: raise ValueError(f'Invalid {k}')
    if min(cfg['lr'],cfg['initial_evidence'])<=0 or cfg['min_lr_ratio']>1:
        raise ValueError('lr/evidence must be positive and min-lr-ratio in [0,1]')
    if cfg['scheduler'] not in ('none','cosine') or cfg['selection'] not in ('map','nll'):
        raise ValueError('Invalid scheduler/selection')
    ExperimentLoss(**cfg['loss']).validate()


def atomic_checkpoint(path, state):
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);os.replace(tmp,path)


def main(argv=None):
    args=parser().parse_args(argv)
    if min(args.epochs,args.threads,args.val_every,args.save_every,args.eval_batch_size,args.eval_chunk)<1:
        raise ValueError('Positive epoch/cadence/runtime sizes required')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output must be new/empty; never overwrite an experiment')
    if bool(args.evaluate) != bool(args.checkpoint): raise ValueError('--evaluate requires --checkpoint and vice versa')
    if args.evaluate and (args.resume or args.warmstart): raise ValueError('Evaluation is not resume/warmstart')
    torch.set_num_threads(args.threads)
    loaded=None
    if args.resume or args.evaluate:
        loaded=torch.load(args.resume or args.checkpoint,map_location='cpu',weights_only=True)
        if loaded.get('format')!=FORMAT: raise ValueError('Not an experiment checkpoint; legacy v5 uses evaluate.py')
        if args.resume and loaded.get('kind')!='resume': raise ValueError('Resume needs last.pt with optimizer/RNG')
        overrides=[k for k in list(TRAIN_DEFAULTS)+list(LOSS_FLAGS) if getattr(args,k) is not None]
        if overrides or args.preset:
            raise ValueError('Resume/evaluate uses saved settings; remove overrides: '+','.join(overrides))
        cfg=loaded['run_config']
        if args.classes is not None and args.classes!=loaded['model_config']['classes']:
            raise ValueError('Checkpoint class count differs')
        if args.label_base is not None and args.label_base!=cfg['label_base']:
            raise ValueError('Checkpoint label encoding differs')
        classes=loaded['model_config']['classes']
    else:
        cfg=resolved_config(args);classes=args.classes
    validate_config(cfg);seed_all(cfg['seed'])
    datasets,receipt=load_data(args.data,classes,cfg['label_base'])
    if receipt['val']['image_sha256']==receipt['test']['image_sha256'] and receipt['val']['text_sha256']==receipt['test']['text_sha256']:
        raise ValueError('Validation equals test; do not tune on the test set')
    mc=ModelConfig(receipt['image_dim'],receipt['text_dim'],receipt['classes'],cfg['hidden'],cfg['layers'],cfg['initial_evidence'])
    model=ExperimentModel(mc,cfg['evidence_mode']).to(args.device)
    if loaded:
        if loaded['data_receipt']['file_sha256']!=receipt['file_sha256']:
            raise ValueError('Checkpoint data SHA differs; no silent data change')
        if loaded['model_config']!=model.configuration(): raise ValueError('Model configuration differs')
        model.load_state_dict(loaded['model_state'],strict=True)
    elif args.warmstart:
        warmstart_fume(model,torch.load(args.warmstart,map_location='cpu',weights_only=True))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.evaluate:
        report=isolated_evaluate(model,datasets[args.evaluate],args.output,args.eval_batch_size,args.eval_chunk)
        save_json(args.output/'evaluation_manifest.json',dict(split=args.evaluate,checkpoint=str(args.checkpoint),
            checkpoint_sha256=sha256_file(args.checkpoint),data_sha256=receipt['file_sha256'],
            checkpoint_epoch=loaded['epoch'],selected_by=loaded.get('selected_by'),no_confidence_filter=True))
        print(json.dumps(report_summary(report),allow_nan=False));return
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.5,.999),weight_decay=cfg['weight_decay'])
    start=0;best={'map':None,'nll':None};stale=0;stop_reference=None
    if args.resume:
        optimizer.load_state_dict(loaded['optimizer']);start=loaded['epoch']
        best=loaded['best'];stale=loaded['stale'];stop_reference=loaded['stop_reference']
        restore_rng(loaded['rng'])
    if start>=args.epochs: raise ValueError('--epochs must exceed completed checkpoint epochs')
    if cfg['scheduler']=='cosine' and args.epochs>cfg['schedule_epochs']:
        raise ValueError('Cosine schedule exhausted; declare sufficient schedule-epochs at the start')
    lc=ExperimentLoss(**cfg['loss']);scales=receipt['train_noise_rms']
    sources={p:sha256_file(Path(__file__).parent/p) for p in ('model.py','losses.py','data.py','metrics.py','experiment.py','run_experiment.py')}
    save_json(args.output/'provenance.json',dict(format=FORMAT,run_config=cfg,model_config=model.configuration(),data=receipt,
        source_sha256=sources,torch=str(torch.__version__),numpy=str(np.__version__),device=args.device,
        parameters=sum(p.numel() for p in model.parameters()),fuzzy=False,unit_prior=True,confidence_filter=False,
        test_used_for_tuning=False,resume=str(args.resume) if args.resume else None,
        warmstart=str(args.warmstart) if args.warmstart else None,
        warmstart_sha256=sha256_file(args.warmstart) if args.warmstart else None,
        optimizer='Adam',adam_betas=[.5,.999],model_selection_split='val',runtime={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}))
    initial=isolated_evaluate(model,datasets['val'],None,args.eval_batch_size,args.eval_chunk)
    save_json(args.output/'initial_validation.json',report_summary(initial))
    loader=DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,drop_last=False,num_workers=0)
    stopped=False;done=start

    def state(epoch,kind,selected_by=None):
        value=dict(format=FORMAT,kind=kind,model_config=model.configuration(),model_state=model.state_dict(),
            run_config=cfg,data_receipt=receipt,epoch=epoch,best=best,stale=stale,stop_reference=stop_reference,
            selected_by=selected_by,source_sha256=sources)
        if kind=='resume': value.update(optimizer=optimizer.state_dict(),rng=rng_state())
        return value

    with (args.output/'history.jsonl').open('w',encoding='utf8') as history, (args.output/'validation.jsonl').open('w',encoding='utf8') as vh:
        for epoch in range(start+1,args.epochs+1):
            model.train();lr=learning_rate(cfg,epoch)
            for group in optimizer.param_groups: group['lr']=lr
            count=0;sums={};clock=time.perf_counter()
            for batch in loader:
                batch=tuple(t.to(args.device) for t in batch)
                optimizer.zero_grad(set_to_none=True)
                _,logs=experiment_objective(model,batch,lc,scales,epoch=epoch,backward=True)
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Nonfinite gradient; optimizer step not applied')
                if cfg['clip_grad']:
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'],error_if_nonfinite=True)
                    logs['preclip_gradient_norm']=float(norm)
                optimizer.step();n=len(batch[0]);count+=n
                for k,v in logs.items():
                    sums[k]=max(sums.get(k,0.),v) if k.startswith('max_') else sums.get(k,0.)+n*v
            if count!=len(datasets['train']): raise RuntimeError('Training rows dropped')
            if str(args.device).startswith('cuda'): torch.cuda.synchronize()
            row=dict(epoch=epoch,lr=lr,train_rows=count,train_seconds=time.perf_counter()-clock,
                **{k:(v if k.startswith('max_') else v/count) for k,v in sums.items()})
            done=epoch
            if epoch%args.val_every==0 or epoch==args.epochs:
                summary=report_summary(isolated_evaluate(model,datasets['val'],None,args.eval_batch_size,args.eval_chunk))
                values={'map':summary['mean_map'],'nll':summary['mean_nll']}
                for key,value in values.items():
                    better=best[key] is None or (value>best[key]['value'] if key=='map' else value<best[key]['value'])
                    if better:
                        path=args.output/f'best_{key}.pt'
                        best[key]=dict(value=value,epoch=epoch,path=str(path.resolve()),split='val')
                        atomic_checkpoint(path,state(epoch,'evaluation','val_'+key))
                score=values[cfg['selection']]
                improved=stop_reference is None or (score>stop_reference+cfg['min_delta'] if cfg['selection']=='map' else score<stop_reference-cfg['min_delta'])
                if improved: stale=0;stop_reference=score
                else: stale+=1
                row['validation']=summary
                vh.write(json.dumps(dict(epoch=epoch,**summary),allow_nan=False)+'\n');vh.flush()
                stopped=bool(cfg['patience'] and stale>=cfg['patience'])
            history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
            print(json.dumps(row,allow_nan=False),flush=True)
            if epoch%args.save_every==0 or epoch==args.epochs or stopped:
                atomic_checkpoint(args.output/'last.pt',state(epoch,'resume'))
            if stopped: break
    save_json(args.output/'summary.json',dict(completed_epochs=done,early_stopped=stopped,best=best,
        selection=cfg['selection'],test_evaluated=False,source_sha256=sources,run_config=cfg))


if __name__=='__main__':
    main()
