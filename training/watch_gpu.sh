#!/bin/bash
# Run on login node to watch GPU utilization during training.
# Usage: bash watch_gpu.sh [job_id]
#
# Reads DCGM log written by train.slurm, or polls live via dcgmi.

JOB_ID="${1:-}"
LOG_DIR="/data/logs"

if [ -n "$JOB_ID" ] && [ -f "$LOG_DIR/gpu_util_${JOB_ID}.log" ]; then
    echo "Tailing DCGM log for job $JOB_ID (DCGM_FI_DEV_GPU_UTIL = field 203)"
    tail -f "$LOG_DIR/gpu_util_${JOB_ID}.log"
else
    echo "Live DCGM poll (all GPUs, 5 s interval). Ctrl-C to stop."
    echo "Target: >80% utilization during training"
    echo ""
    # Header
    printf "%-12s %-8s %-8s %s\n" "Time" "GPU" "Util%" "Node"
    while true; do
        squeue --format="%i %R" --noheader 2>/dev/null | grep -q swift && STATUS="RUNNING" || STATUS="no training job"
        echo "--- $(date '+%H:%M:%S') | $STATUS ---"
        srun --ntasks-per-node=1 --nodes=2 --partition=main \
             dcgmi dmon -e 203 -c 1 2>/dev/null | \
             grep -v '^#\|^$' | \
             awk -v ts="$(date '+%H:%M:%S')" '{printf "%-12s GPU%-5s %-8s %s\n", ts, $1, $2, ENVIRON["SLURMD_NODENAME"]}'
        sleep 10
    done
fi
