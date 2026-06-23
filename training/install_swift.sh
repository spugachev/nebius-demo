#!/bin/bash
# Install ms-swift with Megatron support from git.
# The [megatron] extra is defined in the GitHub source but NOT published to PyPI
# releases — installing from git is the correct approach per ms-swift docs.
set -e

VENV=/data/env/swift
PIP="$VENV/bin/pip"
T="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org"

python -m venv "$VENV"

echo "--- Installing ms-swift[megatron] from git ---"
# This installs from HEAD where [megatron] extra is properly defined,
# pulling in megatron-core and mcore-bridge at the right pinned versions.
$PIP install -q $T \
    "ms-swift[megatron] @ git+https://github.com/modelscope/ms-swift.git" \
    "accelerate"

# Add system torch/CUDA/NCCL after pip install so the resolver never sees
# lightning_thunder's stale package pins.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SYSSITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$SYSSITE" > "$VENV/lib/python${PYVER}/site-packages/system_torch.pth"

echo "=== Installed versions ==="
$PIP show ms-swift transformers megatron-core mcore-bridge 2>/dev/null | grep -E "^(Name|Version):"
echo "=== swift ==="
"$VENV/bin/swift" --version
echo "=== Install complete ==="
