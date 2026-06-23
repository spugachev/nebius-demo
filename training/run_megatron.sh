#!/bin/bash
# Runs inside the container via srun. Arguments passed from train.slurm.
set -e

MODEL_DIR="$1"
DATASET_TRAIN="$2"
DATASET_EVAL="$3"
CHECKPOINT_DIR="$4"

export HF_HOME=/data/.cache/huggingface
export MODELSCOPE_CACHE=/data/.cache/modelscope
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# Install ms-swift with Megatron support (cached on shared fs after first run)
SWIFT_STAMP=/data/.cache/swift_installed
if [ ! -f "$SWIFT_STAMP" ]; then
    echo "Installing ms-swift[megatron]..."
    pip install -q "ms-swift[megatron]" "mcore-bridge" \
        "transformers>=5.5.0" "accelerate" "deepspeed"
    touch "$SWIFT_STAMP"
fi

swift megatron-sft \
    --model_type          qwen3_moe \
    --model_id_or_path    "$MODEL_DIR" \
    --dataset             "$DATASET_TRAIN" \
    --val_dataset         "$DATASET_EVAL" \
    --output_dir          "$CHECKPOINT_DIR" \
    \
    --training_args_cls   MegatronArguments \
    --tensor_model_parallel_size  1 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size  8 \
    \
    --num_train_epochs    3 \
    --learning_rate       5e-6 \
    --lr_scheduler_type   cosine \
    --warmup_ratio        0.05 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --seq_length          8192 \
    \
    --save_steps          500 \
    --save_total_limit    2 \
    --eval_steps          500 \
    \
    --bf16 true \
    --enable_thinking     false \
    \
    --nnodes              "$NNODES" \
    --nproc_per_node      "$NPROC_PER_NODE" \
    --node_rank           "$NODE_RANK" \
    --master_addr         "$MASTER_ADDR" \
    --master_port         "$MASTER_PORT"
