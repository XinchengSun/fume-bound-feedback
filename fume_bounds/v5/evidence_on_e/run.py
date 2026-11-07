"""Train original E objective with optional category OR matching evidence.

Original E loss, normalization, ReLU, orthogonal W, and FUME contrastive are
unchanged. Independent output dirs; no BCE, necessity, c, or confidence filtering.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler
from core import EvidenceOnE,AuxConfig,objective,base,V5,LEGACY_PATH
from metrics import evidence_report

FORMAT='fume_evidence_on_e_v1'
DEFAULTS=dict(task='matching',classes=100,label_base=None,layers=4,seed=2,
              batch_size=300,lr=5e-5,cl_weight=.05,initial_evidence=2.,pair_hidden=128,
              vicinal=1.,noise=1.,error_kl=.01,kl_anneal_epochs=10,
              mix_beta=10.,noise_beta=1.,noise_multiplier=1.,
              pair_negatives=1,pair_positive_mode='overlap',pair_chunk=128,
              pair_clean_weight=0.,uq_pairs=20000,uq_seed=20261107,
              require_ab150=False)
AUX=tuple(AuxConfig.__dataclass_fields__)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    for name,value in DEFAULTS.items():
        if name=='require_ab150':
            p.add_argument('--require-ab150',action='store_true',default=None)
            continue
        choices={'task':('none','category','matching'),'label_base':(0,1),
                 'pair_positive_mode':('overlap','aligned')}.get(name)
        p.add_argument('--'+name.replace('_','-'),
                       type=int if name=='label_base' else type(value),choices=choices)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save-every',type=int,default=5)
    p.add_argument('--uq-every',type=int,default=10,help='0 means final validation only')
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--resume',type=Path)
    return p


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def rng():
    n=np.random.get_state()
    return dict(python=random.getstate(),torch=torch.get_rng_state(),
                numpy=(n[0],torch.from_numpy(n[1].astype(np.int64)),n[2],n[3],n[4]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore(s):
    random.setstate(s['python'])
    torch.set_rng_state(s['torch'].cpu())
    n=s['numpy']
    np.random.set_state((n[0],n[1].cpu().numpy().astype(np.uint32),n[2],n[3],n[4]))
    if s['cuda']:
        if len(s['cuda'])!=torch.cuda.device_count():
            raise ValueError('CUDA device count differs on resume')
        torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def atomic_save(path,state):
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def main(argv=None):
    args=parser().parse_args(argv)
    if min(args.epochs,args.threads,args.save_every)<1 or args.uq_every<0:
        raise ValueError('Invalid runtime counts')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new/empty output directory')
    saved=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved.get('format')!=FORMAT:
            raise ValueError('Only evidence_on_e checkpoints can resume')
        overrides=[k for k in DEFAULTS if getattr(args,k) is not None]
        if overrides:
            raise ValueError('Resume restores saved settings; remove overrides: '+','.join(overrides))
        cfg=saved['config']
    else:
        cfg={k:(getattr(args,k) if getattr(args,k) is not None else v) for k,v in DEFAULTS.items()}
        if cfg['task']=='none':
            for k in ('vicinal','noise','error_kl','pair_clean_weight'):
                cfg[k]=0.
    if min(cfg['classes'],cfg['batch_size'])<2 or cfg['layers']<1 or not 0<=cfg['seed']<2**32:
        raise ValueError('Invalid dimensions/seed')
    if not math.isfinite(cfg['lr']) or cfg['lr']<=0 or cfg['uq_pairs']<1 or not math.isfinite(cfg['cl_weight']) or cfg['cl_weight']<0:
        raise ValueError('Invalid lr/UQ count')
    aux=AuxConfig(**{k:cfg[k] for k in AUX})
    aux.validate()
    torch.set_num_threads(args.threads)
    seed_all(cfg['seed'])
    data_api=base.load_module(V5/'data.py','_e_evidence_data')
    datasets,receipt=data_api.load_data(args.data,cfg['classes'],cfg['label_base'])
    if any(not (d.tensors[2].sum(1)==1).all() for d in datasets.values()):
        raise ValueError('This E comparison is single-positive-label only; no normalization/argmax conversion')
    model=EvidenceOnE(receipt['image_dim'],receipt['text_dim'],receipt['classes'],cfg['layers'],
                      cfg['task'],cfg['initial_evidence'],cfg['pair_hidden']).to(args.device)
    initial=base.backbone_hash(model)
    reference=None
    if cfg['require_ab150']:
        ref=json.loads((base.ARCHIVE/'acceptance150.json').read_text())
        if (cfg['seed'],cfg['batch_size'],cfg['layers'],cfg['lr'])!=(2,300,4,5e-5):
            raise ValueError('AB150 needs seed2/batch300/layers4/lr5e-5')
        if receipt['file_sha256']!=ref['data']['file_sha256'] or initial!=ref['initial_backbone_and_W_sha256']:
            raise ValueError('AB150 data or initialization mismatch')
        if base.sha256(LEGACY_PATH)!=ref['legacy_source']['model.py']:
            raise ValueError('Archived source changed')
        reference=[json.loads(x) for x in (base.ARCHIVE/'runs/A_fume/order.jsonl').read_text().splitlines()]
        if args.epochs>len(reference):
            raise ValueError('Reference order records exhausted')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.5,.999),weight_decay=0)
    generator=torch.Generator().manual_seed(cfg['seed']+1)
    source_paths=[Path(__file__),Path(__file__).with_name('core.py'),Path(__file__).with_name('metrics.py'),
                  V5/'minimal_supervision/minimal_control.py',V5/'data.py',LEGACY_PATH]
    sources={str(p.relative_to(V5)):base.sha256(p) for p in source_paths}
    start=0
    if saved:
        if saved['data_receipt']['file_sha256']!=receipt['file_sha256']:
            raise ValueError('Resume data changed')
        if saved['source_sha256']!=sources:
            raise ValueError('Training source changed; not an exact resume')
        if saved['model_config']!=model.config:
            raise ValueError('Resume model configuration changed')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['loader_rng'].cpu())
        restore(saved['rng'])
        start=saved['epoch']
    if start>=args.epochs:
        raise ValueError('epochs must exceed checkpoint epoch')
    args.output.mkdir(parents=True,exist_ok=True)
    write=base.write_json
    write(args.output/'provenance.json',dict(format=FORMAT,config=cfg,model_config=model.config,
          data=receipt,source_sha256=sources,initial_backbone_and_W_sha256=initial,
          parameters=sum(p.numel() for p in model.parameters()),torch=str(torch.__version__),
          numpy=str(np.__version__),device=args.device,
          fixed_E_membership_supervision=True,
          fixed_E_clean_objective=bool(cfg['cl_weight']==.05),
          contrastive_weight=cfg['cl_weight'],
          bce=False,necessity=False,confidence_filter=False,unit_prior=True,
          test_used_for_selection=False,weight_decay=0,adam_betas=[.5,.999],
          resume=str(args.resume) if args.resume else None,final_test_requested=args.final_test))
    write(args.output/'initial_validation.json',base.evaluate(model,datasets['val']))
    epoch=start
    class LoggedSampler(RandomSampler):
        def __iter__(self):
            ids=list(super().__iter__())
            row=dict(epoch=epoch,order=base.order_hash(ids),
                     first_batch=base.order_hash(ids[:cfg['batch_size']]))
            if reference is not None and row!=reference[epoch-1]:
                raise ValueError('AB150 order mismatch')
            with (args.output/'order.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            return iter(ids)
    loader=DataLoader(datasets['train'],batch_size=cfg['batch_size'],
                      sampler=LoggedSampler(datasets['train'],generator=generator),
                      generator=generator,drop_last=False,num_workers=0)
    with (args.output/'history.jsonl').open('w') as hist,(args.output/'validation.jsonl').open('w') as val:
        for epoch in range(start+1,args.epochs+1):
            model.train()
            count=0
            sums={}
            clock=time.perf_counter()
            for step,batch in enumerate(loader):
                batch=tuple(x.to(args.device) for x in batch)
                optimizer.zero_grad(set_to_none=True)
                _,logs,probe=objective(model,batch,aux,receipt['train_noise_rms'],
                                       epoch=epoch,backward=True,probe=step==0,
                                       cl_weight=cfg['cl_weight'])
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient; optimizer step NOT applied')
                if step==0:
                    with (args.output/'gradient_probe.jsonl').open('a') as f:
                        f.write(json.dumps(dict(epoch=epoch,batch=1,
                                                output_layer_only=True,**probe),
                                           allow_nan=False)+'\n')
                optimizer.step()
                n=len(batch[0])
                count+=n
                for k,v in logs.items():
                    sums[k]=sums.get(k,0.)+n*v
            if count!=len(datasets['train']):
                raise RuntimeError('Training rows lost')
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            row=dict(epoch=epoch,train_rows=count,
                     updates=count//cfg['batch_size']+int(count%cfg['batch_size']>0),
                     lr=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-clock,
                     **{k:v/count for k,v in sums.items()})
            hist.write(json.dumps(row,allow_nan=False)+'\n')
            hist.flush()
            report=dict(epoch=epoch,**base.evaluate(model,datasets['val']))
            if model.task!='none' and (epoch==args.epochs or (args.uq_every and epoch%args.uq_every==0)):
                report['evidence']=evidence_report(model,datasets['val'],cfg['uq_pairs'],cfg['uq_seed'])
            val.write(json.dumps(report,allow_nan=False)+'\n')
            val.flush()
            print(json.dumps(dict(**row,val_mean_map=report['mean_map']),allow_nan=False),flush=True)
            if epoch%args.save_every==0 or epoch==args.epochs:
                atomic_save(args.output/'last.pt',dict(format=FORMAT,config=cfg,
                    model_config=model.config,model=model.state_dict(),
                    optimizer=optimizer.state_dict(),epoch=epoch,
                    loader_rng=generator.get_state(),rng=rng(),source_sha256=sources,
                    data_receipt=receipt))
    if args.final_test:
        report=base.evaluate(model,datasets['test'])
        if model.task!='none':
            report['evidence']=evidence_report(model,datasets['test'],cfg['uq_pairs'],cfg['uq_seed'])
        write(args.output/'test_metrics.json',report)
    write(args.output/'completion.json',dict(completed_epochs=args.epochs,
          total_updates=args.epochs*math.ceil(len(datasets['train'])/cfg['batch_size']),
          fixed_last_checkpoint=True,test_evaluated=args.final_test,task=model.task))


if __name__=='__main__':
    main()
