#!/bin/bash
#SBATCH --job-name=build-image-tl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=main
#SBATCH --output=/data/logs/build-image-tl-%j.out
#SBATCH --time=01:00:00

set -e

# Build a derived enroot image with tilelang installed PROPERLY on top of the
# official ms-swift image. tilelang is needed for Qwen3.6's Gated DeltaNet
# backward on Hopper (the Triton>=3.4 fla kernel is numerically wrong; fla
# auto-uses the tilelang backend).
#
# Pin tilelang==0.1.9 to MATCH the apache-tvm-ffi 0.1.9 the image already ships
# (required by vllm/flashinfer/xgrammar/quack-kernels). tilelang 0.1.11 would
# force apache-tvm-ffi>=0.1.10, which (a) breaks the vllm stack and (b) makes
# tilelang's bundled TVM double-register FFI types against the image's tvm_ffi
# ("__ffi_repr__ already registered"). With 0.1.9 everything reuses the image's
# single libtvm_ffi.so — no conflict, nothing else upgraded. All of tilelang's
# other deps (torch-c-dlpack-ext, z3-solver<4.15.5, ml-dtypes) are already in
# the image, so this is a clean, minimal, version-matched install.

SRC=/data/images/swift431.sqsh
DST=/data/images/swift431-tl.sqsh
NAME=swift_tl_build

# Keep enroot's scratch on node-local SSD (image unpacks to tens of GB).
export ENROOT_DATA_PATH=/scratch/enroot-data
mkdir -p "$ENROOT_DATA_PATH"

echo "=== Build derived image with tilelang ==="
echo "Start: $(date)"

enroot remove -f "$NAME" 2>/dev/null || true
enroot create --name "$NAME" "$SRC"

enroot start --root --rw --mount /data:/data "$NAME" bash -c '
set -e
# Full dependency resolution, but constrain apache-tvm-ffi==0.1.9 so pip keeps
# the image version (shared with vllm/flashinfer) and selects the matching
# tilelang. All other tilelang deps are already present in the image.
pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "tilelang==0.1.9" "apache-tvm-ffi==0.1.9"
echo "--- versions (apache-tvm-ffi must remain 0.1.9) ---"
pip show tilelang apache-tvm-ffi z3-solver 2>/dev/null | grep -E "^(Name|Version):"
echo "--- import check ---"
python -c "import tilelang, tilelang.language; print(\"tilelang\", tilelang.__version__, \"import OK\")"
'

echo "=== Exporting to $DST ==="
rm -f "$DST"
enroot export --output "$DST" "$NAME"
enroot remove -f "$NAME"

ls -lh "$DST"
echo "Done: $(date)"
