#!/bin/bash
#SBATCH --job-name=setup-tilelang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main
#SBATCH --output=/data/logs/setup-tilelang-%j.out
#SBATCH --time=00:20:00

set -e

# Qwen3.6's Gated DeltaNet layers need the `tilelang` backend for their backward
# pass on Hopper (fla's Triton chunk_bwd_dqkwg is numerically wrong on Triton>=3.4;
# the image ships Triton 3.6). Install tilelang (+ z3-solver, which TVM inside
# tilelang dlopens) --no-deps into a shared dir, exposed at train time via
# PYTHONPATH / LD_LIBRARY_PATH in run_megatron.sh.

DEST=/data/env/site-extra

srun --container-image=/data/images/swift431.sqsh \
     --container-mounts=/data:/data \
     bash -c "
set -e
mkdir -p '$DEST'
T='--trusted-host pypi.org --trusted-host files.pythonhosted.org'
pip install --no-deps --target='$DEST' \$T tilelang z3-solver
# TVM bundled in tilelang dlopens libz3.so.4.15; z3-solver ships .4.16 (ABI-compat).
Z3DIR='$DEST/z3/lib'
ln -sf \"\$Z3DIR\"/libz3.so.4.* \"\$Z3DIR/libz3.so.4.15\" 2>/dev/null || true
ls -l \"\$Z3DIR\"/libz3.so* || true
echo '--- import test ---'
PYTHONPATH='$DEST' LD_LIBRARY_PATH=\"\$Z3DIR:\$LD_LIBRARY_PATH\" \
  python -c 'import tilelang; import tilelang.language; print(\"tilelang\", tilelang.__version__, \"OK\")'
echo '=== tilelang ready ==='
"
