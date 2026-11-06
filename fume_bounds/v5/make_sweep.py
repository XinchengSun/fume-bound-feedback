"""Generate a bounded random grid of commands; does NOT launch any training."""
import argparse
import json
import math
from pathlib import Path
import random
import shlex
import sys
if __package__:
    from .run_experiment import parser as train_parser, resolved_config, validate_config
else:
    from run_experiment import parser as train_parser, resolved_config, validate_config


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--grid',type=Path,default=Path(__file__).with_name('tuning_grid.json'))
    p.add_argument('--trials',type=int,default=16,help='Number of configurations, before seed repetitions')
    p.add_argument('--seeds',type=int,nargs='+',default=[2])
    p.add_argument('--sampling-seed',type=int,default=20261106)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--device',default='cuda')
    p.add_argument('--classes',type=int)
    p.add_argument('--label-base',type=int,choices=(0,1))
    args=p.parse_args(argv)
    if args.trials<1 or args.epochs<1 or not args.seeds or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('Positive trials/epochs and unique seeds required')
    if args.root.exists() and any(args.root.iterdir()): raise FileExistsError('Sweep root must be new/empty')
    grid=json.loads(args.grid.read_text(encoding='utf8'))
    allowed={a.dest.replace('_','-') for a in train_parser()._actions}
    forbidden={'help','data','output','resume','warmstart','checkpoint','evaluate','seed','classes','label-base','epochs'}
    if not isinstance(grid,dict) or not grid: raise ValueError('Grid must be a nonempty object')
    for key,values in grid.items():
        if key not in allowed-forbidden or not isinstance(values,list) or not values:
            raise ValueError('Invalid grid key/list: '+key)
        if any(not isinstance(x,(int,float,str)) or isinstance(x,bool) for x in values):
            raise ValueError('Grid values must be scalar CLI arguments')
        if len(set(map(str,values)))!=len(values): raise ValueError('Duplicate grid values')
    keys=sorted(grid);total=math.prod(len(grid[k]) for k in keys)
    if total>sys.maxsize: raise ValueError('Grid is too large; split it into smaller stages')
    selected=random.Random(args.sampling_seed).sample(range(total),min(args.trials,total))
    script=Path(__file__).with_name('run_experiment.py').resolve()
    trials=[];commands=['#!/usr/bin/env bash','set -euo pipefail','# Validation only; test requires an explicit separate command.']
    for trial,flat in enumerate(selected):
        config={};n=flat
        for key in keys:
            config[key]=grid[key][n%len(grid[key])];n//=len(grid[key])
        for seed in args.seeds:
            run=args.root/f'trial_{trial:03d}_seed{seed}'
            cli=['--data',str(args.data.resolve()),'--output',str(run.resolve()),'--epochs',str(args.epochs),
                 '--seed',str(seed),'--device',args.device]
            if args.classes is not None:cli+=['--classes',str(args.classes)]
            if args.label_base is not None:cli+=['--label-base',str(args.label_base)]
            for key,value in config.items():cli+=['--'+key,str(value)]
            cfg=resolved_config(train_parser().parse_args(cli));validate_config(cfg)
            command=shlex.join([sys.executable,str(script),*cli]);commands.append(command)
            trials.append(dict(trial=trial,seed=seed,output=str(run.resolve()),config=cfg,command=command))
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/'commands.sh').write_text('\n'.join(commands)+'\n',encoding='utf8')
    (args.root/'trials.json').write_text(json.dumps(dict(grid=grid,grid_size=total,
        configurations=len(selected),runs=len(trials),sampling_seed=args.sampling_seed,trials=trials),
        indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(configurations=len(selected),runs=len(trials),commands=str(args.root/'commands.sh'),executed=False)))


if __name__=='__main__':main()
