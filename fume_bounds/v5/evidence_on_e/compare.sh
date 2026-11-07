#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: bash compare.sh DATA NEW_ROOT [run.py options]" >&2
  exit 2
fi
DATA=$1
ROOT=$2
shift 2
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
if [ -e "$ROOT" ]; then
  echo "Output root already exists: $ROOT" >&2
  exit 2
fi
for arg in "$@"; do
  case "$arg" in
    --task|--task=*|--data|--data=*|--output|--output=*|--resume|--resume=*)
      echo "Do not override task/data/output/resume in compare.sh" >&2
      exit 2;;
  esac
done
mkdir -p "$ROOT"
for TASK in none matching category; do
  python "$HERE/run.py" --data "$DATA" --output "$ROOT/$TASK" --task "$TASK" "$@"
done
