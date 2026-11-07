#!/bin/sh
set -eu
own=/inspire/ssd/project/video-generation/public/sunxincheng
f=$own/FUME
work=$f/local_ops/E_seed2_20261107
out=$f/runs/E_minimal_no_necessity_seed2_20261107
r=$own/workspaces/635fe852-6e5f-48e2-b5b1-72b10d58561b/requests/sxc-4090x2-635fe852-r2
test "$(hostname)" = job-ecb844ba-8556-40b0-b386-257ad322d9ed-worker-0
test ! -e "$out"
if [ "${1:-}" = inside ]; then
  test "$CUDA_VISIBLE_DEVICES" = GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35
  export PYTHONDONTWRITEBYTECODE=1
  cd "$work/repo/fume_bounds/v5"
  echo "worker=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  exec "$f/local_ops/v5_20261106/validation/env_bin/python" -u -B minimal_supervision/minimal_control.py \
    --data "$f/local_ops/inria_bound_20261101/frozen_fume/datasets/INRIA-Websearch.mat" \
    --output "$out" --classes 100 --label-base 0 --epochs 150 --batch-size 300 \
    --lr 5e-5 --layers 4 --seed 2 --device cuda --threads 2 --require-ab150 --final-test
fi
nohup bash "$r/with-one-gpu.sh" 1 sh "$work/launch_E.sh" inside >> "$work/worker.log" 2>&1 </dev/null &
echo "$!" > "$work/wrapper.pid"
