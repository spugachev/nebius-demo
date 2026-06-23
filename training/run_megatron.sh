#!/bin/bash
# Runs inside container via srun (one task per node).
# ms-swift spawns 8 GPU workers per node internally via torchrun.
set -e

MODEL_DIR="$1"
DATASET_TRAIN="$2"
DATASET_EVAL="$3"
CHECKPOINT_DIR="$4"

# NODE_RANK from srun context (SLURM_PROCID = 0 on node-0, 1 on node-1).
# Must be read here, not pre-exported from sbatch where SLURM_PROCID=0 always.
export NODE_RANK=${SLURM_PROCID:-0}

export HF_HOME=/data/.cache/huggingface
export MODELSCOPE_CACHE=/data/.cache/modelscope
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT nodes=$NNODES gpus=$NPROC_PER_NODE"

source /data/env/swift/bin/activate

# ── Megatron-SWIFT training ──────────────────────────────────────────────────
# Parallelism: PP=2 (one pipeline stage per node), EP=8 (one expert shard per GPU).
# No TP: MoE uses EP for expert distribution; TP adds unnecessary communication.
#
# GPU utilization >80% levers:
#   - gradient_accumulation_steps=8 → 8 micro-batches → pipeline bubble = 1/9 ≈ 11%
#   - packing=true → fills every 8192-token window even with short sequences
#   - use_flash_attn → faster attention, more compute vs memory-bound time
#   - sequence_parallel → reduces activation memory, allows larger effective batch
#   - CUDA_DEVICE_MAX_CONNECTIONS=1 → prevents NCCL/pipeline buffer contention
swift megatron-sft \
    --model_type                    qwen3_moe_instruct \
    --model_id_or_path              "$MODEL_DIR" \
    --dataset                       "$DATASET_TRAIN" \
    --val_dataset                   "$DATASET_EVAL" \
    --output_dir                    "$CHECKPOINT_DIR" \
    \
    --training_args_cls             MegatronArguments \
    --tensor_model_parallel_size    1 \
    --pipeline_model_parallel_size  2 \
    --expert_model_parallel_size    8 \
    --sequence_parallel             true \
    \
    --num_train_epochs              3 \
    --learning_rate                 5e-6 \
    --lr_scheduler_type             cosine \
    --warmup_ratio                  0.05 \
    \
    --per_device_train_batch_size   2 \
    --gradient_accumulation_steps   8 \
    --max_length                    8192 \
    --packing                       true \
    \
    --use_flash_attn                true \
    --gradient_checkpointing        true \
    \
    --save_steps                    500 \
    --save_total_limit              2 \
    --eval_steps                    500 \
    \
    --bf16                          true \
    --enable_thinking               false \
    \
    --nnodes                        "$NNODES" \
    --nproc_per_node                "$NPROC_PER_NODE" \
    --node_rank                     "$NODE_RANK" \
    --master_addr                   "$MASTER_ADDR" \
    --master_port                   "$MASTER_PORT"
