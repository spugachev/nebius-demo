#!/bin/bash
# Runs inside the container via setup_env.slurm.
# Creates /data/env/swift with ms-swift[megatron] in a clean venv.
# System torch/CUDA injected via .pth AFTER install so pip resolver stays clean.
set -e

VENV=/data/env/swift

python -m venv "$VENV"

# Do NOT list mcore-bridge separately — ms-swift[megatron] pins a compatible
# version internally; adding it explicitly causes ResolutionImpossible.
"$VENV/bin/pip" install -q \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    "ms-swift[megatron]" \
    "accelerate"

# Add system torch/CUDA/NCCL AFTER pip install.
# Added before pip = pip sees lightning_thunder via sys.path → conflict.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SYSSITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$SYSSITE" > "$VENV/lib/python${PYVER}/site-packages/system_torch.pth"

echo "=== Installed versions ==="
"$VENV/bin/pip" show ms-swift transformers megatron-core mcore-bridge 2>/dev/null | grep -E "^(Name|Version):"
echo "=== swift command ==="
"$VENV/bin/swift" --version
echo "=== Install complete ==="
