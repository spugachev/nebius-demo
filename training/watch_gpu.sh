#!/bin/bash
# Live GPU utilization of a RUNNING training job, across both nodes.
# Usage: bash watch_gpu.sh <JOB_ID> [interval_seconds]
#
# Uses `srun --overlap` to attach to the job's own allocation and read
# nvidia-smi (utilization.gpu = DCGM_FI_DEV_GPU_UTIL equivalent). This is
# robust: the dcgmi sidecar needs nv-hostengine and only sees one node.
#
# Judge utilization only AFTER warmup — the first iteration JIT-compiles
# TransformerEngine + tilelang kernels (~5 min), during which util is choppy.
set -e

JOB_ID="${1:?usage: watch_gpu.sh <JOB_ID> [interval_seconds]}"
INTERVAL="${2:-10}"

echo "Watching GPU util for job $JOB_ID (both nodes, every ${INTERVAL}s). Target: >80%. Ctrl-C to stop."
while squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null | grep -q RUNNING; do
    VALS=$(srun --jobid="$JOB_ID" --overlap --ntasks-per-node=1 --nodes=2 \
                nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd' ')
    AVG=$(echo "$VALS" | tr ' ' '\n' | awk '{s+=$1;n++} END{if(n) printf "%.0f", s/n}')
    echo "$(date '+%H:%M:%S')  mean=${AVG}%   per-gpu: $VALS"
    sleep "$INTERVAL"
done
echo "Job $JOB_ID is no longer RUNNING."
