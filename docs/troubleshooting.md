# Troubleshooting

Real issues hit while building this example, and the fixes that worked. Ordered
roughly by where they surface in the pipeline.

## Environment / install

### `ms-swift[megatron]` won't pip-install (backtracks forever or breaks torch)
**Symptom:** `pip install "ms-swift[megatron]"` loops through every version, or
"ResolutionImpossible", or training later dies with `No module named
'torch.multiprocessing'`.
**Cause:** ms-swift 4.x with the `[megatron]` extra is **not on PyPI** (PyPI tops
out at 3.7.1). uv can resolve it but then pulls a torch incompatible with the
container's CUDA.
**Fix:** don't pip-install. Use the **official ms-swift docker image** via enroot
(`import_image.slurm`). It ships a consistent torch/TE/flash-attn/megatron-core/vLLM.

### Don't fix dependency gaps with `--no-deps` / symlinks / `LD_PRELOAD`
**Symptom:** each patch reveals the next missing piece (e.g. `libtvm_ffi.so` not found
after a `--no-deps` install + hand-symlinked `libz3`).
**Fix:** install packages properly (let the resolver pull declared deps; constrain
shared versions) and **bake into a derived image** (`build_image_tl.slurm`).

### enroot import is slow / per-node
**Fix:** import once to a shared `.sqsh` on `/data/images/`; both nodes reuse it.
Use the `us-west-1` ModelScope mirror (better latency than the China registries).

## Model loading

### First training backward crashes: `chunk_bwd_dqkwg ... Please install tilelang`
**Symptom:** training reaches the first backward (~67 GB/GPU loaded) then all ranks
die with `RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results
for gated chunk_bwd_dqkwg`.
**Cause:** Qwen3.6 has **Gated DeltaNet** (linear-attention) layers; their fla
backward kernel is numerically wrong on Hopper with Triton ≥3.4 (the image has 3.6).
**Fix:** install the `tilelang` backend. Pin **`tilelang==0.1.9`** to match the
image's `apache-tvm-ffi 0.1.9` (shared with vLLM/flashinfer). tilelang 0.1.11 forces
apache-tvm-ffi ≥0.1.10, which breaks vLLM and double-registers FFI
(`__ffi_repr__ already registered`). `build_image_tl.slurm` does this.

### Both nodes act as rank 0
**Symptom:** distributed init hangs or two processes claim rank 0; venv created twice.
**Cause:** `NODE_RANK=$SLURM_PROCID` set in the **sbatch body** (where SLURM_PROCID
is always 0), then exported to all srun tasks.
**Fix:** read `NODE_RANK=${SLURM_PROCID}` **inside** the per-node runner
(`run_megatron.sh`), and launch with `srun --export=ALL` so MASTER_ADDR/NNODES propagate.

## Throughput / utilization

### GPU utilization stuck at ~73% or alternating 0%/100%
**Symptom:** with `pipeline_model_parallel_size=2`, util alternates between node-0
and node-1 (pipeline bubbles); or steady but only ~73%.
**Fix:** use **PP=1, EP=8 → DP=16** (no cross-node pipeline). Then raise per-step
compute: **`micro_batch_size=2`** (util 73% → 85%).

### `global batch size (16) is not divisible by micro batch size (2) times data parallel size (16)`
**Cause:** with PP=1/TP=1, `data_parallel_size = 16` (EP is orthogonal, it does not
divide DP for the batch constraint). So `global_batch` must be a multiple of
`micro_batch × 16`.
**Fix:** `micro_batch=2` → set `global_batch_size=32`.

### First iteration takes minutes / util looks bad early
Not a bug — the first step JIT-compiles TransformerEngine + tilelang kernels (~5 min),
counter sits at `0/N`. Judge utilization only after it advances.

## Monitoring

### GPU dashboard shows no >80% / looks empty
- You're on **"Basic metrics"** (CPU/RAM/Disk). Switch to **"GPU metrics"**.
- The run already finished — current util is 0. Narrow the time range to the run window.
- `dcgmi dmon` errors with "unable to connect to host engine": `nv-hostengine` isn't
  running, and it only sees one node anyway. Use `srun --jobid --overlap nvidia-smi`.

## Inference

### vLLM: `unrecognized arguments: --chat-template-kwargs`
**Cause:** vLLM 0.23.0 has no such CLI flag.
**Fix:** disable thinking **per request** — put `"chat_template_kwargs":
{"enable_thinking": false}` in the chat-completion body (compare.py does this).

### Base model emits a `<think>…</think>` preamble before the tool call
Expected for Qwen3.6 with thinking on. Pass `enable_thinking=false` per request so
base and tuned are compared apples-to-apples.

## Infrastructure / Terraform

- `public_o11y_enabled = false` in `terraform.tfvars` — known recipe bug otherwise.
- Install `yq` before `terraform apply`.
- **Filesystems can't be shrunk in place:** to reduce a Nebius filestore size,
  `nebius compute filesystem delete <id>` → `terraform state rm <addr>` → re-apply.
- Tenant `csa-hiring-sandboxK` is shared — use unique resource names; don't create a
  new tenant; don't destroy the lab after submission.
- ModelScope (not HuggingFace) for the model download; the path double-nests:
  `/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B`.
