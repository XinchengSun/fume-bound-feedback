#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo 'Usage: bash coarse_runs.sh DATA NEW_ROOT [extra run_experiment.py flags]' >&2
  exit 2
fi
DATA="$1"; ROOT="$2"; shift 2
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
# Identical initialization, seed, batch and epoch budget; no test evaluation.
for PRESET in control v5 coarse; do
  "$PYTHON" "$HERE/run_experiment.py" --data "$DATA" --output "$ROOT/$PRESET" \
    --preset "$PRESET" --epochs 30 --seed 2 "$@"
done
"$PYTHON" "$HERE/summarize_experiments.py" --root "$ROOT" --output "$ROOT/validation_comparison.csv"
