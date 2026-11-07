#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
V5=$(cd -- "$HERE/../../.." && pwd)
DATA=/inspire/ssd/project/video-generation/public/sunxincheng/FUME/local_ops/inria_bound_20261101/frozen_fume/datasets/INRIA-Websearch.mat
if [ "$#" -gt 0 ]; then DATA=$1; fi
OUT=/inspire/ssd/project/video-generation/public/sunxincheng/FUME/runs/E_matching_M1_noKL_seed2_150_20261107
if [ "$#" -gt 1 ]; then OUT=$2; fi
if [ "$#" -gt 2 ]; then echo "Usage: $0 [DATA] [NEW_OUTPUT]" >&2; exit 2; fi

if [ -z "$(printenv CUDA_VISIBLE_DEVICES 2>/dev/null || true)" ]; then
  echo 'ERROR: Set CUDA_VISIBLE_DEVICES to physical GPU1 UUID explicitly.' >&2
  exit 2
fi
if [ ! -f "$DATA" ]; then
  echo "ERROR: Missing INRIA data: $DATA" >&2
  exit 2
fi
if [ -e "$OUT" ] || [ -e "$OUT.console.log" ]; then
  echo "ERROR: Output/log already exists; do not overwrite: $OUT" >&2
  exit 2
fi
python - <<'PY'
import torch,os
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count()==1, 'Exactly one visible GPU required'
name=torch.cuda.get_device_name(0)
assert '4090' in name, f'Expected archived GPU1 RTX4090, got {name}'
print('VISIBLE_DEVICE',os.getenv('CUDA_VISIBLE_DEVICES'),name,flush=True)
PY
printf 'REPO_HEAD: '
git -C "$V5/../.." rev-parse HEAD
printf 'SOURCE: %s\nDATA: %s\nOUTPUT: %s\n' "$V5/evidence_on_e/run.py" "$DATA" "$OUT"

python -u "$V5/evidence_on_e/run.py" \
  --data "$DATA" --output "$OUT" --device cuda \
  --task matching --classes 100 --label-base 0 \
  --seed 2 --batch-size 300 --lr 5e-5 --layers 4 --epochs 150 \
  --vicinal 1 --noise 1 --error-kl 0 --pair-clean-weight 0 \
  --mix-beta 10 --noise-beta 1 --noise-multiplier 1 \
  --pair-hidden 128 --pair-negatives 1 --pair-positive-mode overlap \
  --pair-chunk 128 --initial-evidence 2 \
  --uq-every 10 --uq-pairs 20000 --uq-seed 20261107 \
  --require-ab150 --final-test 2>&1 | tee "$OUT.console.log"

python "$HERE/verify_M1.py" \
  --run "$OUT" \
  --reference-order "$V5/diagnostics/20261107_ab150/runs/A_fume/order.jsonl"
