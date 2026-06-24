#!/bin/bash
#SBATCH --job-name=fc-comparison
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --partition=main
#SBATCH --output=/data/logs/fc-comparison-%j.out
#SBATCH --time=01:30:00

set -e

# Base vs fine-tuned function-calling comparison on ONE node:
#   base  → GPUs 0,1 (TP=2) on :8000
#   tuned → GPUs 2,3 (TP=2) on :8001
# Both served from the ms-swift image (vLLM 0.23.0, supports Qwen3.6 GDN — smoke-tested),
# enable_thinking=false for an apples-to-apples comparison. Then compare.py scores both.

BASE_MODEL="/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B"
TUNED_MODEL="/data/checkpoints/qwen3-fc-20260624-0025/v0-20260624-082630/checkpoint-426"
IMAGE="/data/images/swift431-tl.sqsh"
OUT="/data/logs/comparison-${SLURM_JOB_ID}"

srun --container-image="$IMAGE" --container-mounts="/data:/data" bash -c '
set -e
# Note: enable_thinking=false is passed per-request by compare.py (vLLM 0.23.0
# has no --chat-template-kwargs CLI flag).
common="--tensor-parallel-size 2 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
        --max-model-len 8192 --host 0.0.0.0"

echo "=== launching base (GPU 0,1 :8000) and tuned (GPU 2,3 :8001) ==="
CUDA_VISIBLE_DEVICES=0,1 vllm serve "'"$BASE_MODEL"'"  --served-model-name qwen36-base  $common --port 8000 \
    > /data/logs/vllm-base-'"$SLURM_JOB_ID"'.log  2>&1 &
PID_BASE=$!
CUDA_VISIBLE_DEVICES=2,3 vllm serve "'"$TUNED_MODEL"'" --served-model-name qwen36-tuned $common --port 8001 \
    > /data/logs/vllm-tuned-'"$SLURM_JOB_ID"'.log 2>&1 &
PID_TUNED=$!
trap "kill $PID_BASE $PID_TUNED 2>/dev/null" EXIT

echo "=== running comparison (waits for both /health, up to 25 min) ==="
python /data/code/eval/compare.py \
    --base-url  http://localhost:8000 \
    --tuned-url http://localhost:8001 \
    --prompts   /data/code/eval/prompts.jsonl \
    --out       "'"$OUT"'" \
    --wait      25

echo "=== report ==="
cat "'"$OUT"'.md"
'
