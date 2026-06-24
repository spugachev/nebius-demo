#!/bin/bash
# Serve the FINE-TUNED Qwen3.6 checkpoint with vLLM for comparison.
# The checkpoint is a complete HF dir (--save_safetensors during training) → load directly.
# Run inside an allocation alongside serve_base.sh:
#   srun --container-image=/data/images/swift431-tl.sqsh --container-mounts=/data:/data \
#        bash /data/code/inference/serve_tuned.sh
# Uses GPUs 2,3 (TP=2). Serves on port 8001 as model name "qwen36-tuned".
set -e

MODEL="${MODEL:-/data/checkpoints/qwen3-fc-20260624-0025/v0-20260624-082630/checkpoint-426}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

# enable_thinking=false is passed per-request by the client (no CLI flag in vLLM 0.23.0).
exec vllm serve "$MODEL" \
    --served-model-name qwen36-tuned \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --max-model-len 8192 \
    --host 0.0.0.0 --port "${PORT:-8001}"
