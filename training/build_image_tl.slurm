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

# Build a derived enroot image with tilelang installed PROPERLY (with all its
# declared dependencies) on top of the official ms-swift image. This is the
# robust alternative to runtime --no-deps/symlink/LD_PRELOAD shims:
#   - pip resolves tilelang's real deps: apache-tvm-ffi (provides libtvm_ffi.so),
#     z3-solver<4.15.5 (provides the exact libz3.so.4.15 TVM dlopens), etc.
#   - the image's existing torch 2.11 satisfies tilelang's `torch` requirement,
#     so nothing heavy is reinstalled and nothing conflicts.
# tilelang is needed for Qwen3.6's Gated DeltaNet backward on Hopper (Triton>=3.4
# fla kernel is numerically wrong; fla auto-uses the tilelang backend).

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
pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    tilelang
echo "--- versions ---"
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
