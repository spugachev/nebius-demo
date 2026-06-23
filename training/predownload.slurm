#!/bin/bash
#SBATCH --job-name=predownload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=main
#SBATCH --output=/data/logs/predownload-%j.out
#SBATCH --time=02:00:00

set -e

MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct"
MODEL_DIR="/data/models/Qwen3-30B-A3B-Instruct"

export HF_HOME=/data/.cache/huggingface
export MODELSCOPE_CACHE=/data/.cache/modelscope

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$MODEL_DIR"

echo "Downloading $MODEL_ID → $MODEL_DIR"
echo "Start: $(date)"

srun --container-image="nvcr.io/nvidia/pytorch:25.03-py3" \
     --container-mounts="/data:/data" \
     bash -c "
set -e
pip install -q huggingface_hub hf_transfer
export HF_HOME=/data/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
python -c \"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.pt', '*.bin'],
)
print('Download complete')
\"
"

echo "Done: $(date)"
du -sh "$MODEL_DIR"
