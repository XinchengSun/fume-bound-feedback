#!/usr/bin/env bash
set -eu
root=$(cd -- "$(dirname -- "$0")" && pwd -P)
case "${1:-}" in 0|1) card=$1; shift ;; *) echo 'Usage: with-one-gpu.sh 0|1 command [args]' >&2; exit 2 ;; esac
[[ $# -gt 0 ]] || exit 2
[[ $(hostname) == job-ecb844ba-8556-40b0-b386-257ad322d9ed-worker-0 ]] || { echo 'Enter the R2 GPU worker terminal first.' >&2; exit 2; }
folder="$root/gpu$card"
exec 8>>"$folder/workload.lock"
flock -x -w 3 8 || { echo "GPU $card is claimed by a workload." >&2; exit 75; }
exec 9>>"$folder/handoff.lock"
flock -s -w 30 9 || { echo 'Timed out waiting for idle buffers to release.' >&2; exit 75; }
uuid=$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)
pids=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$pids" ]] || { echo "GPU $card already has a compute process." >&2; exit 75; }
export CUDA_VISIBLE_DEVICES="$uuid"
"$@"
