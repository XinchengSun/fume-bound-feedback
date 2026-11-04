"""Copy modules without silently overwriting the user's trainer or controller."""
import argparse
import hashlib
from pathlib import Path
import shutil


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target',type=Path,required=True)
    args=ap.parse_args(); root=Path(__file__).resolve().parent;target=args.target.resolve()
    if not target.is_dir():raise SystemExit('Target directory must already exist.')
    planned=[]
    for module in ('live_reference','fume_control'):
        for src in sorted((root/module).glob('*.py')):
            dst=target/module/src.name
            if dst.exists() and sha(dst)!=sha(src):
                raise SystemExit(f'Refusing to overwrite different source: {dst}. Use a new version directory.')
            if not dst.exists():planned.append((src,dst))
    for src,dst in planned:
        dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst)
    print(f'Copied {len(planned)} files. Existing identical files retained.')
    print('No trainer/adapter/loss/checkpoint/data file was edited. Follow INTEGRATE.md for the two hooks.')

if __name__=='__main__':main()
