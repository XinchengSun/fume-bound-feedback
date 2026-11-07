"""Summarize one validation-selected checkpoint per arm; never reads test metrics."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    rows=[]
    for path in sorted(Path(root).rglob('validation.jsonl')):
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records:continue
        provenance=json.loads((path.parent/'provenance.json').read_text())
        best=max(records,key=lambda row:row['mean_map'])
        rows.append(dict(run=str(path.parent),selected_by='val_mean_map',
                         classification_source=provenance['run_config']['classification_source'],
                         parameters=provenance['parameters'],
                         **best))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=summarize(a.root)
    if not rows:raise ValueError('No validation results found')
    if a.output.exists():raise FileExistsError('Refusing to overwrite summary')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print(f'{len(rows)} runs summarized. Pair UQ and category UQ are separate columns.')

if __name__=='__main__':main()
