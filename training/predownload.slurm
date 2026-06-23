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

MODEL_ID="Qwen/Qwen3-30B-A3B"
MODEL_DIR="/data/models/Qwen3-30B-A3B"

export MODELSCOPE_CACHE=/data/.cache/modelscope

mkdir -p "$MODELSCOPE_CACHE" "$MODEL_DIR"

echo "Downloading $MODEL_ID from ModelScope → $MODEL_DIR"
echo "Start: $(date)"

srun --container-image="nvcr.io/nvidia/pytorch:25.03-py3" \
     --container-mounts="/data:/data" \
     bash -c "
set -e
pip install -q modelscope
export MODELSCOPE_CACHE=/data/.cache/modelscope
python -c \"
from modelscope import snapshot_download
snapshot_download(
    model_id='$MODEL_ID',
    cache_dir='$MODEL_DIR',
)
print('Download complete')
\"
"

echo "Done: $(date)"
du -sh "$MODEL_DIR"
