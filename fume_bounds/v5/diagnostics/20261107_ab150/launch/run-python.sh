#!/bin/sh
set -eu
own=/inspire/ssd/project/video-generation/public/sunxincheng
lib="$own/spatial_audio/tools/ffmpeg-6.1.1/lib"
exec "$lib/ld-linux-x86-64.so.2" --library-path "$lib:$own/runtime/cpu_migration_20261102/python312/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$own/runtime/cpu_migration_20261102/sam/bin/python" "$@"
