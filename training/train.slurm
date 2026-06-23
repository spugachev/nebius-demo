#!/bin/bash
#SBATCH --job-name=swift-megatron-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --partition=main
#SBATCH --output=/data/logs/train-%j.out
#SBATCH --time=12:00:00

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR="/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B"
DATASET_TRAIN="/data/datasets/train.jsonl"
DATASET_EVAL="/data/datasets/eval.jsonl"
CHECKPOINT_DIR="/data/checkpoints/qwen3-fc-$(date +%Y%m%d-%H%M)"
LOG_DIR="/data/logs"
IMAGE="/data/images/swift431.sqsh"   # official ms-swift image, imported once

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: $IMAGE not found. Run import_image.slurm first."
    exit 1
fi

# ── Distributed env (torchrun-style; megatron sft reads these from the env) ──
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=8
# NODE_RANK is set inside run_megatron.sh from SLURM_PROCID (per-task), not here.

# ── GPU utilization monitoring (host-side, every 5s) ─────────────────────────
dcgmi dmon -e 203 -d 5000 > "$LOG_DIR/gpu_util_${SLURM_JOB_ID}.log" 2>&1 &
DCGM_PID=$!
trap "kill $DCGM_PID 2>/dev/null" EXIT

echo "Job $SLURM_JOB_ID | nodes=$NNODES | master=$MASTER_ADDR | image=$IMAGE | start=$(date)"
echo "Model:  $MODEL_DIR"
echo "Output: $CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# ── Launch: one srun task per node; megatron sft spawns 8 GPU workers per node ─
srun --export=ALL \
     --container-image="$IMAGE" \
     --container-mounts="/data:/data" \
     bash /data/code/training/run_megatron.sh \
       "$MODEL_DIR" "$DATASET_TRAIN" "$DATASET_EVAL" "$CHECKPOINT_DIR"
