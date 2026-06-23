#!/bin/bash
#SBATCH --job-name=import-image
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=main
#SBATCH --output=/data/logs/import-image-%j.out
#SBATCH --time=02:00:00

set -e

# Official ms-swift image: torch 2.11 + TransformerEngine + flash-attn + vLLM,
# all pre-built and mutually compatible. us-west-1 mirror = best latency from eu-north2.
IMAGE_URI="docker://modelscope-registry.us-west-1.cr.aliyuncs.com#modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.37.1-swift4.3.1"
SQSH="/data/images/swift431.sqsh"

mkdir -p /data/images

echo "=== Importing image to $SQSH ==="
echo "Start: $(date)"

if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "$IMAGE_URI"
fi

echo "=== Imported. Size: ==="
ls -lh "$SQSH"
echo "Done import: $(date)"

echo "=== Introspecting image contents (via pyxis, the runtime path) ==="
srun --container-image="$SQSH" --container-mounts="/data:/data" \
     python -c '
import importlib
for m in ["torch","transformer_engine","flash_attn","megatron.core","mcore_bridge","swift","transformers","vllm"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  OK   {m:22s} {v}")
    except Exception as e:
        print(f"  MISS {m:22s} {type(e).__name__}: {e}")
import torch
print("torch.cuda:", torch.version.cuda, "| multiprocessing import:", end=" ")
import torch.multiprocessing; print("OK")
'
echo "=== Introspection complete: $(date) ==="
