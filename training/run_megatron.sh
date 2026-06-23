#!/bin/bash
# Runs inside the official ms-swift container via srun (one task per node).
# `megatron sft` internally spawns NPROC_PER_NODE GPU workers via torchrun.
# Args modeled on the official tested example:
#   ms-swift/examples/megatron/moe/qwen3_moe.sh  (Qwen3-30B-A3B, PP=2 EP=8, 9.6s/it on 16 GPUs)
set -e

MODEL_DIR="$1"
DATASET_TRAIN="$2"
DATASET_EVAL="$3"
CHECKPOINT_DIR="$4"

# NODE_RANK must be read here (per srun task), NOT pre-exported from the sbatch
# body where SLURM_PROCID is always 0. With --ntasks-per-node=1, SLURM_PROCID is
# 0 on node-0 and 1 on node-1.
export NODE_RANK=${SLURM_PROCID:-0}
export MASTER_PORT=${MASTER_PORT:-29500}

# Megatron pipeline parallelism needs a single GPU->GPU connection so NCCL does
# not contend with pipeline send/recv buffers.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# NCCL over InfiniBand (eu-north2-a fabric, ~475 GB/s busbw measured)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=WARN

# Model is a local path (pre-downloaded); caches still point at shared FS.
export HF_HOME=/data/.cache/huggingface
export MODELSCOPE_CACHE=/data/.cache/modelscope

# Qwen3.6 is a hybrid model with Gated DeltaNet (linear-attention) layers. Their
# backward kernel (fla's chunk_bwd_dqkwg) is numerically WRONG on Hopper with
# Triton>=3.4 (the image has Triton 3.6), so fla requires the `tilelang` backend.
# tilelang + z3-solver are installed --no-deps into this shared dir by
# setup_tilelang.slurm; expose them at runtime (libz3.so.4.15 is symlinked there).
export PYTHONPATH=/data/env/site-extra:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/data/env/site-extra/z3/lib:${LD_LIBRARY_PATH:-}

echo "node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -1

# Full SFT of an instruction-tuned MoE.
#   PP=1 x EP=8 (expert shard per GPU) x TP=1 -> DP=2 (one replica per node) = 16 GPUs.
#   Data-parallel across nodes (not pipeline): both nodes compute in parallel on
#   different data, only an overlappable gradient all-reduce between them -> no
#   cross-node pipeline bubbles, sustained >80% GPU utilization. (PLAN.md documents
#   this as the alternative when PP across nodes idles GPUs — observed exactly that.)
#   micro_batch=1, global_batch=16, DP=2 -> grad-accum 8 per replica.
#   recompute_granularity full -> fits activations; MoE fusions -> max throughput.
#   save_safetensors true -> checkpoint is directly HF/vLLM-loadable (no separate export step).
#   lr 5e-6 (not the example's 1e-5): model is already instruction-tuned (project decision).
megatron sft \
    --model                          "$MODEL_DIR" \
    --dataset                        "$DATASET_TRAIN" \
    --val_dataset                    "$DATASET_EVAL" \
    --split_dataset_ratio            0 \
    --save_safetensors               true \
    \
    --pipeline_model_parallel_size   1 \
    --expert_model_parallel_size     8 \
    --moe_permute_fusion             true \
    --moe_grouped_gemm               true \
    --moe_shared_expert_overlap      true \
    --moe_aux_loss_coeff             1e-3 \
    \
    --micro_batch_size               1 \
    --global_batch_size              16 \
    --packing                        true \
    --max_length                     8192 \
    \
    --recompute_granularity          full \
    --recompute_method               uniform \
    --recompute_num_layers           1 \
    --cross_entropy_loss_fusion      true \
    --sequence_parallel              true \
    --attention_backend              flash \
    \
    --num_train_epochs               3 \
    --finetune                       true \
    --lr                             5e-6 \
    --lr_warmup_fraction             0.05 \
    --min_lr                         5e-7 \
    \
    --output_dir                     "$CHECKPOINT_DIR" \
    --eval_steps                     200 \
    --save_steps                     200 \
    --save_total_limit               2 \
    --no_save_optim                  true \
    --no_save_rng                    true \
    --dataloader_num_workers         8 \
    --dataset_num_proc               8
