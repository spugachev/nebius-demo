#!/bin/bash
# Serve the BASE Qwen3.6-35B-A3B with vLLM (OpenAI-compatible API) for comparison.
# Run inside an allocation, e.g.:
#   salloc --nodes=1 --gpus-per-node=4 --partition=main --time=02:00:00
#   srun --container-image=/data/images/swift431-tl.sqsh --container-mounts=/data:/data \
#        bash /data/code/inference/serve_base.sh
# Uses GPUs 0,1 (TP=2). Serves on port 8000 as model name "qwen36-base".
set -e

MODEL="${MODEL:-/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# enable_thinking=false matches training and keeps tool calls clean (no <think> preamble),
# so the base-vs-tuned comparison is apples-to-apples on function-calling quality.
exec vllm serve "$MODEL" \
    --served-model-name qwen36-base \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --max-model-len 8192 \
    --host 0.0.0.0 --port "${PORT:-8000}"
