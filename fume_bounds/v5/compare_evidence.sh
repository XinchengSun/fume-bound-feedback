#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash compare_evidence.sh DATA NEW_ROOT [--device cuda ...]' >&2
  exit 2
fi
DATA=$1
ROOT=$2
shift 2
for arg in "$@"; do
  case "$arg" in
    --evidence-task*|--classification-source*|--data|--data=*|--output*|--resume*|--warmstart*|--evaluate*|--checkpoint*)
      echo "Do not override comparison identity or reuse a checkpoint: $arg" >&2; exit 2 ;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ -e "$ROOT" ]]; then echo 'Comparison root must not exist.' >&2; exit 2; fi
mkdir -p "$ROOT"
# This explicit control uses the SAME ordinary multihot BCE in both arms.
# Legacy v5 posterior-BCE remains untouched in run_experiment.py.
for TASK in category matching; do
  python "$HERE/run_evidence_ab.py" --data "$DATA" --output "$ROOT/$TASK" \
    --evidence-task "$TASK" --classification-source direction --epochs 30 \
    --seed 2 --batch-size 300 --kl-weight 0.01 "$@"
done
python "$HERE/summarize_evidence_ab.py" --root "$ROOT" --output "$ROOT/comparison.csv"
