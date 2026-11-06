"""Summarize validation-selected runs. Never reads test reports for ranking."""
import argparse
import csv
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',choices=('map','nll'),default='map')
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Do not overwrite an existing summary')
    metric='mean_map' if a.selection=='map' else 'mean_nll'
    rows=[]
    for path in sorted(a.root.rglob('validation.jsonl')):
        data=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if not data:continue
        chosen=(max if a.selection=='map' else min)(data,key=lambda r:r[metric])
        config=json.loads((path.parent/'provenance.json').read_text())['run_config']
        rows.append(dict(run=str(path.parent),selection_split='val',selection_metric=metric,
            preset=config['preset'],seed=config['seed'],evidence_mode=config['evidence_mode'],
            lr=config['lr'],kl_weight=config['loss']['error_kl'],vicinal_weight=config['loss']['vicinal'],
            noise_weight=config['loss']['noise'],consistency_weight=config['loss']['consistency'],
            complete=(path.parent/'summary.json').exists(),**chosen))
    if not rows:raise ValueError('No validation records found')
    rows.sort(key=lambda r:r[metric],reverse=a.selection=='map')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps(dict(runs=len(rows),output=str(a.output),selection_split='val')))


if __name__=='__main__':main()
