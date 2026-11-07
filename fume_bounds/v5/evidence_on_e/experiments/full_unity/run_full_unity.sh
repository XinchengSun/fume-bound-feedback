#!/usr/bin/env bash
set -euo pipefail

# L_total = 1*FUME_contrastive + 1*E_membership_classification
#         + 1*PairEvidenceCE + 1*MixEvidenceCE + 1*NoiseEvidenceCE.
# KL = 0, no extra branch, no probability-based retrieval re-ranking.

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
V5=$(cd -- "$HERE/../../.." && pwd)
DEFAULT_DATA=/inspire/ssd/project/video-generation/public/sunxincheng/FUME/local_ops/inria_bound_20261101/frozen_fume/datasets/INRIA-Websearch.mat
DEFAULT_OUTPUT=/inspire/ssd/project/video-generation/public/sunxincheng/FUME/runs/E_matching_full_unity_seed2_150_20261107
DATA="$DEFAULT_DATA"
OUTPUT="$DEFAULT_OUTPUT"
if [ "$#" -gt 0 ]; then DATA=$1; fi
if [ "$#" -gt 1 ]; then OUTPUT=$2; fi
if [ "$#" -gt 2 ]; then echo "Usage: $0 [DATA] [NEW_OUTPUT]" >&2; exit 2; fi

if [ -z "$(printenv CUDA_VISIBLE_DEVICES 2>/dev/null || true)" ]; then
  echo "Set CUDA_VISIBLE_DEVICES to GPU1's physical UUID explicitly" >&2
  exit 2
fi
if [ ! -f "$DATA" ]; then echo "Missing data: $DATA" >&2; exit 2; fi
if [ -e "$OUTPUT" ] || [ -e "$OUTPUT.console.log" ]; then
  echo "Refusing to overwrite existing result/log: $OUTPUT" >&2
  exit 2
fi
python - <<'PY'
import os,torch
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count()==1, "Exactly one GPU must be visible"
name=torch.cuda.get_device_name(0)
assert "4090" in name, f"Expected physical GPU1 RTX 4090; got {name}"
print("GPU_UUID",os.environ["CUDA_VISIBLE_DEVICES"],"DEVICE",name,flush=True)
PY

mkdir -p "$(dirname -- "$OUTPUT")"
git -C "$V5/../.." rev-parse HEAD
python -u "$V5/evidence_on_e/run.py" \
  --data "$DATA" --output "$OUTPUT" --device cuda \
  --task matching --classes 100 --label-base 0 \
  --seed 2 --batch-size 300 --lr 5e-5 --layers 4 --epochs 150 \
  --cl-weight 1 \
  --pair-clean-weight 1 \
  --vicinal 1 --noise 1 \
  --error-kl 0 \
  --mix-beta 10 --noise-beta 1 --noise-multiplier 1 \
  --pair-negatives 1 --pair-positive-mode overlap \
  --pair-hidden 128 --pair-chunk 128 --initial-evidence 2 \
  --uq-every 10 --uq-pairs 20000 --uq-seed 20261107 \
  --require-ab150 --final-test 2>&1 | tee "$OUTPUT.console.log"

python "$HERE/verify_full_unity.py" \
  --run "$OUTPUT" \
  --reference-order "$V5/diagnostics/20261107_ab150/runs/A_fume/order.jsonl"
