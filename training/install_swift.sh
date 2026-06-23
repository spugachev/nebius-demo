#!/bin/bash
# Runs inside the container via setup_env.slurm.
# ms-swift[megatron] extra does not exist on PyPI — install components separately.
# mcore-bridge conflicts with ms-swift on datasets version; use --no-deps to bypass.
set -e

VENV=/data/env/swift

python -m venv "$VENV"

PIP="$VENV/bin/pip"
TRUSTED="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org"

echo "--- Step 1: install ms-swift base + accelerate ---"
$PIP install -q $TRUSTED "ms-swift" "accelerate"

echo "--- Step 2: install megatron-core ---"
$PIP install -q $TRUSTED "megatron-core>=0.16,<0.20"

echo "--- Step 3: install mcore-bridge (--no-deps to bypass datasets conflict) ---"
# mcore-bridge declares datasets>=3.0 but ms-swift pins datasets<3.0.
# At runtime mcore-bridge only needs megatron-core + ms-swift (already installed).
$PIP install -q $TRUSTED --no-deps "mcore-bridge"

echo "--- Step 4: upgrade transformers for Qwen3.6 tokenizer support ---"
$PIP install -q $TRUSTED "transformers>=5.5.0"

# Add system torch/CUDA/NCCL AFTER pip install so the runtime finds them
# without polluting the pip resolver.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SYSSITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$SYSSITE" > "$VENV/lib/python${PYVER}/site-packages/system_torch.pth"

echo "=== Installed versions ==="
$PIP show ms-swift transformers megatron-core mcore-bridge 2>/dev/null | grep -E "^(Name|Version):"
echo "=== swift command ==="
"$VENV/bin/swift" --version
echo "=== Install complete ==="
