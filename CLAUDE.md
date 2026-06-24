# CLAUDE.md

## Project

Nebius CSA hiring assignment: end-to-end multi-node LLM fine-tuning for function calling on Nebius Soperator, plus inference comparison. See ASSESMENT.md for requirements, PLAN.md for implementation design.

Deadline: June 26, 2026.

## Stack

- **Python (local only)**: managed with `uv` (not pip/poetry/conda) — used for the local dataset-prep scripts. The *training environment* on the cluster is NOT pip/uv; it is the official ms-swift docker image (see below).
- **Training**: Megatron-SWIFT (`megatron sft` CLI from ms-swift 4.3.1 + Megatron-Core 0.17.1 + mcore-bridge 1.5.0), Expert Parallelism, full SFT. Runs inside the official ModelScope ms-swift enroot image — NOT a pip/uv install (ms-swift 4.x `[megatron]` is not on PyPI).
- **Training image**: `swift431-tl.sqsh` = official `modelscope:...torch2.11.0-vllm0.23.0-...swift4.3.1` (us-west-1 mirror) + `tilelang==0.1.9` baked in (Qwen3.6 needs it on Hopper — see Gotchas). Built once by `build_image_tl.slurm`.
- **Model**: Qwen3.6-35B-A3B (hybrid MoE: 256 experts top-8 + **Gated DeltaNet** linear-attention layers, April 2026, already instruction-tuned). Fallback: Qwen3-30B-A3B-Instruct-2507 (plain MoE, no GDN → no tilelang needed).
- **Dataset**: hypervariance/function-calling-sharegpt (87k raw → ~75k after validation) + 1k custom process-automation examples
- **Infrastructure**: Nebius Soperator (Slurm on Kubernetes), 2 nodes x 8 H200 GPUs, eu-north2-a. Shared FS mounted at `/data`; code at `/data/code/`, model at `/data/models/`, datasets at `/data/datasets/`, checkpoints at `/data/checkpoints/`, images at `/data/images/`, logs at `/data/logs/`.
- **Container runtime**: enroot via Pyxis SPANK plugin (not Docker)
- **Inference**: vLLM with `--tool-call-parser qwen3_coder`, TP=2 (not TP=8, 35B fits on 2 GPUs)
- **IaC**: Terraform (forked from nebius-solution-library/soperator/installations/example/)
- **Login node**: `185.82.70.159:22`, SSH key `./ssh/nebius_ed25519` (gitignored, never commit). `terraform/login.sh` wraps it.

## Commands

Local (dataset preparation):
```bash
uv sync
uv run python training/prepare_dataset.py           # download HF + parse inline + apply_chat_template → train/eval JSONL
uv run python training/validate_dataset.py          # count, validate, token stats, sample review
```

Terraform:
```bash
cd terraform && terraform init && terraform plan -var-file=terraform.tfvars
```

On the Soperator cluster (one-time setup, then train). Order matters:
```bash
sbatch /opt/slurm-test/quickcheck/hello.sh                       # verify Slurm + GPUs
sbatch --nodes=2 /opt/slurm-test/quickcheck/nccl_multi_node.sh   # verify inter-node NCCL (measured ~475 GB/s, target >300)
sbatch /data/code/training/predownload.slurm                     # download Qwen3.6-35B-A3B from ModelScope → /data/models/ (~67GB)
sbatch /data/code/training/import_image.slurm                    # enroot-import official ms-swift image → /data/images/swift431.sqsh (~18GB)
sbatch /data/code/training/build_image_tl.slurm                  # derive swift431-tl.sqsh = base image + tilelang 0.1.9
sbatch /data/code/training/train.slurm                           # launch 2-node training (uses swift431-tl.sqsh)
# NO export step: --save_safetensors=true writes HF safetensors into each checkpoint-N dir → point vLLM straight at it.
sbatch /data/code/eval/run_comparison.slurm                      # base :8000 + tuned :8001 (TP=2), then compare.py → report
# interactive serving + hand queries:
#   srun ... --container-image=/data/images/swift431-tl.sqsh --pty bash   # shell into container on a GPU node
#   bash /data/code/inference/serve_base.sh   # / serve_tuned.sh  (vLLM, TP=2)
#   python3 /data/ask.py <port> <base|tuned> "<question>"                 # one tool-call query
```

Monitoring a running job's GPU utilization (the dcgmi sidecar needs nv-hostengine and only sees one node; this is the robust way):
```bash
srun --jobid=<JOBID> --overlap --ntasks-per-node=1 --nodes=2 \
     nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader   # live, both nodes
bash /data/code/training/watch_gpu.sh <JOBID>                    # convenience wrapper around the above
```

Model path note: ModelScope nests the download, so the actual model dir is
`/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B` (double-nested).

## Code style

- Python 3.11+
- No type stubs or excessive type annotations on simple scripts
- Scripts are CLI tools with argparse or dataclass-based args
- Keep scripts self-contained where practical; avoid deep abstraction layers
- JSONL for data files, HCL for Terraform
- Template-based generation with fixed random seeds for reproducibility

## Key decisions (do not revisit)

- **MoE over dense**: dense models above 32B don't exist in 2026 open-weight. MoE with Megatron-SWIFT EP achieves >80% GPU utilization; naive MoE + DeepSpeed does not (10-20%).
- **Full SFT over LoRA**: LoRA + EP=8 is untested at scale (only documented at EP=2 on 2 GPUs). Expected LoRA GPU utilization on MoE: 50-70%, risks failing >80% requirement. Full SFT is the only documented path for >80%.
- **Catastrophic forgetting mitigation**: dataset includes ~15% non-tool conversations + negative examples. Comparison focuses on function-calling tasks only. General instruction data mix is a stretch goal.
- **Megatron-SWIFT over TRL/DeepSpeed**: 10x throughput for MoE training (Qwen team recommendation). Standard DeepSpeed ZeRO-3 + LoRA + MoE is incompatible (DeepSpeed #7669, TRL #1268).
- **Nebius GPU metric** is DCGM_FI_DEV_GPU_UTIL (time GPUs execute kernels), not SM occupancy or MFU.
- **hypervariance/function-calling-sharegpt as primary dataset**: public (no gate), 87k examples. Parsed inline in `prepare_dataset.py` (tool schemas embedded as free text in system messages, function calls use `<functioncall>` tags, arguments sometimes double-stringified). Expected ~75k valid examples after parsing. No intermediate file — HuggingFace dataset cache makes re-runs fast.
- **apply_chat_template for formatting**: all datasets go through tokenizer.apply_chat_template(messages, tools=tools, enable_thinking=False). Confirmed: Qwen3.6 tokenizer supports structured `tool_calls` field natively. Never manually construct `<tool_call>` XML.
- **Thinking mode disabled**: `enable_thinking=False`. Simpler training, lower latency, cleaner for customer integration.
- **Qwen3.6 tool-call format is XML** (`<function=name><parameter=key>value</parameter></function>`), NOT Hermes JSON. vLLM parser: `qwen3_coder`, not `hermes`.
- **vLLM TP=2 for inference**: 35B MoE in bf16 ≈ 70GB, fits on 2 H200s. TP=8 adds unnecessary communication overhead.
- **Environment = official ms-swift docker image, NOT pip/uv**: ms-swift 4.x with the `[megatron]` extra is not published to PyPI (PyPI tops out at 3.7.1). Every pip/uv attempt either backtracked forever or pulled an incompatible torch (`No module named 'torch.multiprocessing'`). The image ships torch 2.11 + TransformerEngine 2.16 + flash-attn 2.8.3 + megatron-core 0.17.1 + mcore-bridge 1.5.0 + transformers 5.12.1, all mutually consistent. Bake extra deps (tilelang) into a derived image — never `--no-deps`/symlink/`LD_PRELOAD` at runtime.
- **Parallelism = PP=1, EP=8 → DP=16** (NOT PP=2). With TP=1/PP=1, `data_parallel_size = world/(TP*PP) = 16`; EP=8 is *orthogonal* (shards experts within DP). PP=2 across the two nodes idled GPUs (alternating 0%/100% pipeline bubbles, ~73% or worse). PP=1 keeps both nodes fully busy (only an all-reduce + MoE all-to-all per step). Consequence: **global_batch_size must be divisible by micro_batch_size × 16**.
- **micro_batch_size=2, global_batch_size=32 → 85% sustained util**. micro_batch=1/global_batch=16 gave only ~73% (all-reduce every step dominates the tiny per-step compute). Doubling micro_batch doubled per-step GEMMs and lifted util to 85–92%. Memory ~113/143GB — fits. Next lever if needed: micro_batch=4/global_batch=64.
- **`megatron sft` CLI with Megatron-style args** (NOT `swift megatron-sft` / HF args): `--model <path>` (arch auto-detected, no `--model_type`), `--micro_batch_size`/`--global_batch_size` (NOT per_device/grad_accum), `--lr 5e-6`. Gold reference: `ms-swift/examples/megatron/moe/qwen3_moe.sh`. Multi-node launch is torchrun-style env vars (`NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT`), not flags.
- **No checkpoint export step**: `--save_safetensors true` writes a complete HF checkpoint (config.json + index + 16 safetensors shards + tokenizer) directly into each `checkpoint-N` dir. Point vLLM straight at it. (The earlier "export is a 2-node PP=2 job" assumption is obsolete.)
- **Separate Slurm script + runner**: train.slurm submits run_megatron.sh via srun, avoiding shell quoting issues.
- **NODE_RANK = SLURM_PROCID, read INSIDE run_megatron.sh** (per-srun-task), NOT pre-exported from the sbatch body where SLURM_PROCID is always 0 (that made both nodes rank 0).
- **srun --export=ALL**: guarantees MASTER_ADDR/MASTER_PORT/NNODES propagate from sbatch into srun tasks.
- **LR=5e-6 for full SFT**: 1e-5 is too high for an already-instruction-tuned model; risks training instability and excessive forgetting.
- **eval/prompts.jsonl is hand-written**: it is not produced by the data pipeline; it defines expected tool calls for the compare.py evaluation.

## Dataset pipeline

```
hypervariance/function-calling-sharegpt (HuggingFace, auto-cached)
  → prepare_dataset.py (parse hypervariance inline, apply_chat_template, split 96/4)
  → train.jsonl (~65k) + eval.jsonl (~2.7k)
  → validate_dataset.py (counts, JSON check, token stats, sample review)

eval/prompts.jsonl (50 hand-written test cases, NOT pipeline output — created manually)
```

hypervariance format details:
- Single column: `conversations` (list of `{from, value}` objects)
- Roles: `system`, `human`, `gpt`, `function_response`
- Tool schemas embedded in system message as free text (not structured)
- Function calls: `<functioncall> {"name": "...", "arguments": ...} </functioncall>`
- Arguments sometimes double-stringified: `'{"key": "value"}'` string inside JSON

## File layout

```
terraform/         # Soperator Terraform config (forked from nebius-solution-library)
training/
  prepare_dataset.py     # local: HF download + parse hypervariance + apply_chat_template → train/eval JSONL
  validate_dataset.py    # local: counts, JSON check, token stats
  predownload.slurm      # cluster: ModelScope download of Qwen3.6-35B-A3B → /data/models
  import_image.slurm     # cluster: enroot import official ms-swift image → /data/images/swift431.sqsh
  build_image_tl.slurm   # cluster: derive swift431-tl.sqsh = base + tilelang 0.1.9
  train.slurm            # cluster: 2-node sbatch; sets MASTER_ADDR/NNODES, dcgmi+nvidia-smi monitoring, srun runner
  run_megatron.sh        # runs inside container per node; sets NODE_RANK, env, calls `megatron sft`
  watch_gpu.sh           # convenience: live GPU util of a running job via srun --overlap nvidia-smi
inference/
  serve_base.sh / serve_tuned.sh   # vLLM serve (TP=2, qwen3_coder parser); tuned = checkpoint-426
  smoke_test.slurm                 # de-risk: vLLM loads Qwen3.6 GDN + emits a tool call
  ask.py                           # CLI: python3 ask.py <port> <model> "<q>" (also uploaded to /data/ask.py)
eval/
  prompts.jsonl          # 23 hand-written function-calling test cases + expected calls
  mock_tools.py          # canned tool impls for the executable-success metric
  compare.py             # stdlib harness: query both endpoints, score, write report.md+json
  run_comparison.slurm   # one node: serve base+tuned, wait, run compare.py
  results/               # committed comparison.md + comparison.json
docs/                    # architecture.md, monitoring.md, troubleshooting.md, demo_script.md
```

## Nebius / Soperator specifics

- Terraform recipe source: `nebius-solution-library/soperator/installations/example/`
- Both `nebius/nebius-solution-library` and `nebius/nebius-solutions-library` repos have identical content
- Container runtime is enroot (not Docker): `srun --container-image="..." command`
- Built-in tests at `/opt/slurm-test/quickcheck/` on cluster: hello.sh, nccl_single_node.sh, nccl_multi_node.sh, containers.sh
- NCCL multi-node target: average bus bandwidth > 300 GB/s
- SSH to login node via `login.sh` script or `kubectl exec`
- Shared filesystem is jail root; additional storage via jail submounts
- Tenant: `csa-hiring-sandboxK` (shared, use unique resource names)
- Upload to cluster: `scp` to login node, files go to `/shared/` on jail FS

## Verified results (2026-06-24) — both exercises DONE

**Exercise 1 — training (job 65):** full SFT, State=COMPLETED exit 0, **51m33s**, 426/426 steps, 3 epochs, **sustained GPU util 85–92%** on 16×H200. Loss 0.20→0.13, eval 0.196. Checkpoint (HF-ready, 66GB): `/data/checkpoints/qwen3-fc-20260624-0025/v0-20260624-082630/checkpoint-426` (best-eval: `checkpoint-200`). Base model: `/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B`. Dataset packed to 4558 train + 191 val (~8131 tokens avg).

**Exercise 2 — inference (job 84):** vLLM base vs tuned over 23 prompts (`eval/results/comparison.md`). Tuned wins on every metric: appropriate call/no-call 87→**95.7%**, function-name accuracy 82→**100%**, argument exact-match 76→**94%**, executable 65→**78%**. Honest trade-off: clarify-rate dipped 100→67% (tuned more eager to call — candidate for more negative/clarify data). Docs (architecture/monitoring/troubleshooting/demo) in `docs/`.

## Gotchas

- **Qwen3.6 has Gated DeltaNet (linear-attention) layers → needs `tilelang` on Hopper.** Training crashes on the first backward: `RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg ... Please install tilelang`. The image's Triton is 3.6, and fla auto-uses the tilelang backend when importable. Fix: pin **`tilelang==0.1.9`** to match the image's `apache-tvm-ffi 0.1.9` (shared with vllm/flashinfer/xgrammar). tilelang 0.1.11 forces apache-tvm-ffi>=0.1.10 → breaks vllm AND double-registers FFI (`__ffi_repr__ already registered`). `build_image_tl.slurm` bakes it in (`pip install "tilelang==0.1.9" "apache-tvm-ffi==0.1.9"`). The fallback model (Qwen3-30B-A3B) has no GDN → no tilelang.
- **GPU util: `dcgmi dmon` needs `nv-hostengine` running and only sees the sbatch node.** Robust measurement of a running job, both nodes: `srun --jobid=N --overlap --ntasks-per-node=1 --nodes=2 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader`. Judge util only AFTER warmup — the first iteration JIT-compiles TE + tilelang kernels (counter stuck at 0/N with choppy util for ~5 min is normal).
- **Never patch with `--no-deps`/symlinks/`LD_PRELOAD`** (user directive). Install packages properly (let the resolver pull declared deps; constrain shared versions) and bake into the image. `--no-deps` once hid a missing `libtvm_ffi.so` and snowballed.
- **vLLM 0.23.0 serves Qwen3.6 GDN fine** (smoke-tested) and the `qwen3_coder` parser emits structured tool_calls. Model load is slow (~11 min: MoE + GDN + kernel compile).
- **vLLM has NO `--chat-template-kwargs` CLI flag** — passing it makes the server exit at arg-parse. Disable thinking **per request** instead: `"chat_template_kwargs": {"enable_thinking": false}` in the chat-completion body (base model otherwise emits a `<think>…</think>` preamble; tuned model trained with thinking off doesn't).
- **Tuned checkpoint is directly vLLM-loadable** (HF safetensors). For comparison run both on one node (base GPU0-1:8000, tuned GPU2-3:8001) via `run_comparison.slurm`; for hand queries use `/data/ask.py`.
- Set `public_o11y_enabled = false` in terraform.tfvars (known Terraform recipe bug)
- Install `yq` before running terraform apply
- Nebius filesystems can't be shrunk in-place: to reduce size, `nebius compute filesystem delete` + `terraform state rm` + re-apply.
- Do not use the same shared filesystem for 2 different Soperator jails
- ms-swift names checkpoints as `checkpoint-N` (e.g. checkpoint-200, checkpoint-400), not `checkpoint-final`
- Megatron-Core version must be >= 0.16 and < 0.20 (image has 0.17.1); transformers >= 5.5.0 for Qwen3.6 (image has 5.12.1)
- Do not destroy the lab environment after submission
- Join existing tenant `csa-hiring-sandboxK`, do not create a new one
- On fresh clusters, may need to run `ldconfig` in jail for NVIDIA libs (Soperator issue #2468)
- `MODELSCOPE_CACHE` and `HF_HOME` must point to shared filesystem so all nodes share model cache
- Model download uses ModelScope (`snapshot_download`), not HuggingFace (no auth/gate). Result is double-nested: `/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B`.
- enroot image pull/import is slow on first run; do it once to a shared `.sqsh` so both nodes reuse it (`import_image.slurm`)
- Pre-download model weights in a separate Slurm job before training (67GB download wastes GPU time)
- save_steps=200 + save_total_limit=2 to prevent storage overflow (each HF checkpoint ≈ 66GB)
- `CUDA_DEVICE_MAX_CONNECTIONS=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in run_megatron.sh
- hypervariance dataset arguments are sometimes double-stringified — prepare_dataset.py must handle this
- Full SFT on instruction-tuned model risks catastrophic forgetting — keep non-tool examples in data mix
