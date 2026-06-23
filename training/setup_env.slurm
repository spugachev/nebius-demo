#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=main
#SBATCH --output=/data/logs/setup-env-%j.out
#SBATCH --time=00:30:00

set -e

VENV=/data/env/swift

echo "=== Installing ms-swift Megatron environment ==="
echo "Target: $VENV"
echo "Start: $(date)"

rm -rf "$VENV"

# Run inside the same container as training so Python/ABI match
srun \
  --container-image="nvcr.io/nvidia/pytorch:25.03-py3" \
  --container-mounts="/data:/data,/etc/ssl:/etc/ssl:ro" \
  bash /data/code/training/install_swift.sh

echo "Done: $(date)"
ls -la "$VENV/bin/swift"
"$VENV/bin/swift" --version
