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
VENV="/data/env/swift"

if [ ! -f "$VENV/bin/swift" ]; then
    echo "ERROR: $VENV not found. Run setup_env.slurm first."
    exit 1
fi

# ── Distributed env ──────────────────────────────────────────────────────────
# MASTER_ADDR from the first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=8

# NODE_RANK is intentionally NOT set here — it's set inside run_megatron.sh
# from SLURM_PROCID so each srun task gets its own correct rank.

# ── NCCL / Megatron performance ──────────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=WARN

# Critical for Megatron pipeline parallelism: single GPU-to-GPU channel
# prevents NCCL from competing with pipeline send/recv buffers.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Avoid fragmentation on large-activation workloads
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/data/.cache/huggingface
export MODELSCOPE_CACHE=/data/.cache/modelscope

# ── GPU monitoring ──────────────────────────────────────────────────────────
dcgmi dmon -e 203 -d 5000 > "$LOG_DIR/gpu_util_${SLURM_JOB_ID}.log" 2>&1 &
DCGM_PID=$!
trap "kill $DCGM_PID 2>/dev/null" EXIT

echo "Job $SLURM_JOB_ID | nodes=$NNODES | master=$MASTER_ADDR | start=$(date)"
echo "Model:  $MODEL_DIR"
echo "Output: $CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# ── Launch (one srun task per node; run_megatron.sh spawns 8 GPUs per node) ─
srun --export=ALL \
     --container-image="nvcr.io/nvidia/pytorch:25.03-py3" \
     --container-mounts="/data:/data,/etc/ssl:/etc/ssl:ro" \
     bash /data/code/training/run_megatron.sh \
       "$MODEL_DIR" "$DATASET_TRAIN" "$DATASET_EVAL" "$CHECKPOINT_DIR"
