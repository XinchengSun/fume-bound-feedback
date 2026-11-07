#!/bin/sh
set -eu
own=/inspire/ssd/project/video-generation/public/sunxincheng
r=$own/workspaces/635fe852-6e5f-48e2-b5b1-72b10d58561b/requests/sxc-4090x2-635fe852-r2
v=$own/FUME/local_ops/v5_20261106/validation
test "$(hostname)" = job-ecb844ba-8556-40b0-b386-257ad322d9ed-worker-0
export PYTHONPATH="$v/numpy226_site:${PYTHONPATH:-}"
nohup bash "$r/with-one-gpu.sh" 1 sh "$r/run-python.sh" -u -B "$v/recover150.py" >>"$v/compare150-worker.log" 2>&1 </dev/null &
echo "GPU1 wrapper PID $!"
