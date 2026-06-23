#!/bin/bash
# ms-swift has an internal dep conflict (datasets<3.0 vs modelscope>=1.17 which
# pulls datasets>=3.0). Install with --no-deps to bypass, then add runtime deps.
set -e

VENV=/data/env/swift
PIP="$VENV/bin/pip"
T="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org"

python -m venv "$VENV"

echo "--- Step 1: ms-swift --no-deps (bypass internal datasets conflict) ---"
$PIP install -q $T --no-deps "ms-swift"

echo "--- Step 2: core runtime deps ms-swift actually needs ---"
$PIP install -q $T \
    "transformers>=5.5.0" \
    "accelerate" \
    "peft" \
    "datasets" \
    "sentencepiece" \
    "tiktoken" \
    "jinja2" \
    "numpy" \
    "tqdm" \
    "packaging" \
    "einops" \
    "modelscope>=1.17"

echo "--- Step 3: megatron-core ---"
$PIP install -q $T "megatron-core>=0.16,<0.20"

echo "--- Step 4: mcore-bridge --no-deps (datasets version conflict with ms-swift) ---"
$PIP install -q $T --no-deps "mcore-bridge"

# Add system torch/CUDA/NCCL after pip so the resolver never sees lightning_thunder.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SYSSITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$SYSSITE" > "$VENV/lib/python${PYVER}/site-packages/system_torch.pth"

echo "=== Versions ==="
$PIP show ms-swift transformers megatron-core mcore-bridge 2>/dev/null | grep -E "^(Name|Version):"
echo "=== swift ==="
"$VENV/bin/swift" --version
echo "=== Install complete ==="
