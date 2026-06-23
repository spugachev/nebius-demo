#!/bin/bash
# ms-swift has irreconcilable dependency conflicts via pip's backtracking resolver.
# Use uv (SAT-solver based) which finds valid solutions pip cannot.
set -e

VENV=/data/env/swift
PIP="$VENV/bin/pip"
T="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org"

rm -rf "$VENV"
python -m venv "$VENV"

echo "--- Install uv (better SAT-solver resolver) ---"
$PIP install -q $T uv

echo "--- Install ms-swift[megatron] + accelerate via uv ---"
# uv resolves complex dep conflicts that pip's backtracking algorithm cannot.
# ms-swift[megatron] extra is defined in git HEAD but not in any PyPI release.
"$VENV/bin/uv" pip install \
    --python "$VENV/bin/python" \
    --index-url https://pypi.org/simple/ \
    "ms-swift[megatron] @ git+https://github.com/modelscope/ms-swift.git" \
    "accelerate"

# Add system torch/CUDA/NCCL after install so pip/uv resolver never sees
# lightning_thunder's stale package pins.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SYSSITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$SYSSITE" > "$VENV/lib/python${PYVER}/site-packages/system_torch.pth"

echo "=== Versions ==="
$PIP show ms-swift transformers megatron-core mcore-bridge 2>/dev/null | grep -E "^(Name|Version):"
echo "=== swift ==="
"$VENV/bin/swift" --version
echo "=== Install complete ==="
