# Implementation Plan

Deadline: June 26, 2026.

## Summary

Fine-tune a 2026 open-source MoE LLM for function calling on 16 H200 GPUs using Nebius Soperator, then serve and compare base vs tuned model.

Stack: Megatron-SWIFT with Expert Parallelism, Qwen3.6-35B-A3B, Slurm via Soperator, vLLM for inference.

## Implementation status (as-built, 2026-06-24)

This section records what was actually built and how it deviates from the
original design below. `CLAUDE.md` holds the authoritative operational detail.

**Exercise 1 (training): DONE.** Full SFT of Qwen3.6-35B-A3B completed on 16×H200:
State=COMPLETED, 51m33s, 426/426 steps, 3 epochs, **sustained GPU util 85–92%**
(target >80%), loss 0.20→0.13, eval 0.196. HF-ready checkpoint at
`/data/checkpoints/qwen3-fc-20260624-0025/v0-20260624-082630/checkpoint-426`.

Deviations from the original plan that proved necessary:

| Original assumption | What actually worked |
|---|---|
| Env via pip/uv or "Option A/B/C" | **Official ms-swift enroot image only** — ms-swift 4.x `[megatron]` is not on PyPI. Built `swift431-tl.sqsh` (image + tilelang). |
| `pipeline_model_parallel_size = 2` | **PP=1, EP=8 → DP=16.** PP=2 across nodes idled GPUs (pipeline bubbles). EP is orthogonal to DP. |
| "Alternative: DP = 16/8 = 2" | DP is actually **16** (= world/(TP·PP)); EP does not divide DP for the batch constraint. |
| `micro_batch=1, global_batch=16` | **micro_batch=2, global_batch=32** → util 73%→85%. global_batch must be divisible by micro_batch·16. |
| `sequence_parallel=true` lever | No-op at TP=1 (Megatron auto-disables it). Not a util lever here. |
| Checkpoint export (2-node, PP=2) | **No export** — `--save_safetensors=true` writes a full HF checkpoint per save. |
| HuggingFace download | **ModelScope** `snapshot_download` (no gate); path double-nests. |
| `/shared/...` paths | Shared FS mounts at **`/data/`** on this cluster. |
| (not anticipated) | **Qwen3.6 is a hybrid model with Gated DeltaNet layers** whose backward kernel is wrong on Hopper/Triton≥3.4 → requires the **tilelang** backend (pin 0.1.9 to the image's apache-tvm-ffi 0.1.9). |

Everything below is the original design; the rationale (why MoE, why full SFT,
dataset handling, tool-call format) still holds. The code blocks for env setup,
parallelism, export, and storage paths are superseded by the table above and by
the current scripts in `training/`.

## Model

Primary: **Qwen/Qwen3.6-35B-A3B** (April 2026, 35B total / 3B active, 256 experts, Apache 2.0). Already instruction-tuned (no separate base variant published).

Fallback: **Qwen/Qwen3-30B-A3B-Instruct-2507** (July 2025) -- has a tested Megatron-SWIFT example with confirmed 9.6s/iteration on 16 GPUs.

### Why MoE + Megatron-SWIFT

- Qwen team recommends Megatron for MoE training (10x speedup over HuggingFace Transformers)
- Expert Parallelism (EP=8) maps naturally to 8-GPU-per-node topology
- Shows production-grade MoE training knowledge in the demo
- Uses a 2026 model rather than a 2024 dense model

Why not dense: no dense open-weight model above 32B was released in 2026. The industry moved to MoE. Qwen2.5-72B-Instruct (Sep 2024) is the newest dense 72B and would work well with standard TRL/DeepSpeed, but is less impressive for the demo.

Why not MoE + TRL/DeepSpeed: reported 10-20% GPU utilization (Unsloth #2582), ZeRO-3 + LoRA + MoE has documented incompatibilities (DeepSpeed #7669, TRL #1268).

### Why full SFT, not LoRA

LoRA on MoE with Expert Parallelism is only documented at EP=2 on 2 GPUs (5.1s/it). No public example exists for LoRA + EP=8 + PP=2 on 16 GPUs. Expected LoRA GPU utilization on MoE: 50-70%, which risks failing the >80% requirement. Full SFT is the only documented path that achieves >80% GPU utilization with Megatron-SWIFT EP on MoE.

**Catastrophic forgetting mitigation**: since the base model is already instruction-tuned, full SFT risks overwriting general capabilities. Mitigations:
1. Training data includes ~15% non-tool-call conversations (from hypervariance dataset, where the model declines to use tools)
2. Custom examples include 10% "no tool needed" and 10% "should clarify" cases
3. Exercise 2 comparison focuses strictly on function-calling tasks, not general capabilities
4. If time permits: add 10-20% general instruction data (e.g., from SlimOrca or OpenHermes) to the training mix

### Qwen3.6 tool calling format

Qwen3.6 uses XML-based tool calls (shared with Qwen3.5, different from Qwen3 Hermes JSON):

```
<tool_call>
<function=get_weather>
<parameter=location>Paris</parameter>
</function>
</tool_call>
```

Tool definitions go in the system prompt wrapped in `<tools>...</tools>` as JSON schemas. Tool responses are wrapped in `<tool_response>...</tool_response>` inside a user role message. vLLM parser: `--tool-call-parser qwen3_coder` (not `hermes`).

The tokenizer's `apply_chat_template(messages, tools=tools)` handles all formatting automatically -- confirmed to support structured `tool_calls` field in assistant messages (renders to XML). Requires recent transformers (>= 5.5.0).

## Training Framework

**Megatron-SWIFT** (ms-swift with Megatron-Core backend).

Key dependencies:
- ms-swift >= 4.0
- mcore-bridge >= 1.5.0
- Megatron-Core >= 0.16, < 0.20
- TransformerEngine >= 2.3
- flash-attn 2.8.3
- PyTorch 2.8+
- CUDA 12.8+
- transformers >= 5.5.0 (required for Qwen3.6 tokenizer)

**Environment (as-built):** the **official ms-swift enroot image is the only
viable path** — ms-swift 4.x with the `[megatron]` extra is not published to PyPI,
so pip/uv installs into an NGC base image fail (endless backtracking or an
incompatible torch). Option B (shared-FS venv) was tried and abandoned for the
same reason. What works:
1. `import_image.slurm` — `enroot import` the official image
   `modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.37.1-swift4.3.1`
   once to a shared `.sqsh` (so both nodes reuse it). It ships torch 2.11 + TE 2.16
   + flash-attn 2.8.3 + megatron-core 0.17.1 + mcore-bridge 1.5.0 + transformers 5.12.1.
2. `build_image_tl.slurm` — derive `swift431-tl.sqsh` = image + `tilelang==0.1.9`
   (Qwen3.6 Gated DeltaNet needs it on Hopper; pin to the image's apache-tvm-ffi 0.1.9).
3. `train.slurm` runs `srun --container-image=/data/images/swift431-tl.sqsh ...`.

**Parallelism (as-built):** for 2 nodes × 8 GPUs:
- `expert_model_parallel_size = 8` (experts sharded across 8 ranks)
- `pipeline_model_parallel_size = 1`, `tensor_model_parallel_size = 1`
- → `data_parallel_size = world/(TP·PP) = 16`. **EP is orthogonal to DP** — it
  shards experts *within* the DP dimension (DP/EP = 2 expert-replica groups).
- `micro_batch_size = 2`, `global_batch_size = 32` (must be divisible by micro·16).

> The original plan used PP=2 (pipeline across nodes). That idled GPUs with
> cross-node pipeline bubbles (alternating 0%/100%, ≤73% util). PP=1 keeps both
> nodes fully busy (only an all-reduce + MoE all-to-all per step) → 85% util.
> Note the original "alternative DP = 16/8 = 2" was a miscount: DP is 16.

## Dataset

### Strategy

`prepare_dataset.py` does everything in one pass: downloads hypervariance from HuggingFace (cached by `datasets` library), parses it inline, merges with custom examples, applies `tokenizer.apply_chat_template()`, splits, and writes final JSONL.

No intermediate file. HuggingFace dataset cache (`~/.cache/huggingface/datasets`) makes re-runs fast without re-downloading.

### hypervariance format details

**Primary: hypervariance/function-calling-sharegpt** (87k examples, Apache 2.0)
- ShareGPT format: single `conversations` column with `{from, value}` objects
- Roles: `system`, `human`, `gpt`, `function_response`
- Tool schemas embedded as **free text** in the system message (not a structured field)
- Function calls: `<functioncall> {"name": "func", "arguments": {...}} </functioncall>`
- Arguments sometimes double-stringified: `"arguments": "{\"key\": \"value\"}"` (string inside JSON)
- ~15% are non-tool conversations (model declines) — kept for catastrophic forgetting mitigation

Parsing done inline in `prepare_dataset.py`:
1. Extract JSON tool schemas from system message free text
2. Parse `<functioncall>` tags → structured `tool_calls` with `id`/`tool_call_id`
3. Handle double-stringified arguments
4. Map roles: `human`→`user`, `gpt`→`assistant`, `function_response`→`tool`
5. Skip examples with unparseable schemas or malformed calls (~10-15% yield loss)


### Output format (ms-swift JSONL)

`prepare_dataset.py` produces `{"messages": [...]}` where the assistant turn content is already rendered XML — result of calling `apply_chat_template(..., tokenize=False)` and extracting the assistant content. No `tools` field in output (tools are rendered into the system message by the template).

```json
{
  "messages": [
    {"role": "system", "content": "# Tools\n\nYou may call one or more functions...\n\n<tools>\n[{\"type\": \"function\", ...}]\n</tools>"},
    {"role": "user", "content": "Create a ticket for Acme about failed payment"},
    {"role": "assistant", "content": "<tool_call>\n<function=create_ticket>\n<parameter=customer>Acme</parameter>\n<parameter=category>payment_failure</parameter>\n</function>\n</tool_call>"}
  ]
}
```

Thinking mode disabled: `enable_thinking=False` in `apply_chat_template`. Simpler training signal, lower inference latency, cleaner for customer integration.

### Splits

```
Train:   ~65,000  (96% of valid hypervariance examples)
Eval:     ~2,700  (4% held-out)
```

50 demo/test prompts are hand-written in `eval/prompts.jsonl` (not pipeline output).

### Training data volume estimate

- ~85k examples × ~436 tokens avg = ~37M tokens
- With packing into 8192 seq length ≈ 4,500 packed sequences
- global_batch_size=16 → ~280 steps per epoch
- At ~10s/step ≈ 0.8h per epoch
- **Run 3 epochs** → ~2.5h total training (sufficient for GPU dashboard graphs)

Note: examples are short (function-calling conversations are concise). This is normal for the domain.
Data sufficiency is not a concern — 85k examples is more than enough for fine-tuning an already-instruction-tuned model on a narrow task.

### Dataset pipeline

```
hypervariance/function-calling-sharegpt (HuggingFace, auto-cached)
         │
         ├── generate_custom_examples.py (1k examples, fixed seed=42)
         │        └── custom_process_automation.jsonl
         │
         ▼
    prepare_dataset.py
    (load HF dataset, parse hypervariance inline,
     merge with custom examples,
     apply_chat_template per example,
     split 96% train / 4% eval,
     write JSONL)
         │
         ▼
    train.jsonl  (~65k)
    eval.jsonl   (~2.7k)
         │
         ▼
    validate_dataset.py
    (counts, JSON validity, token length distribution, 5 sample examples)
```

## GPU Utilization Strategy

Nebius dashboards show DCGM_FI_DEV_GPU_UTIL ("percentage of time a GPU spends executing tasks"). This is the standard nvidia-smi GPU-Util metric, not SM occupancy or MFU.

Target: >80% on all 16 GPUs during steady-state training. **Achieved: 85–92%**
(measured live via `srun --jobid=N --overlap nvidia-smi`, confirmed on the Nebius
DCGM dashboard). Measure only after warmup — the first iteration JIT-compiles
TransformerEngine + tilelang kernels (~5 min of choppy util).

Levers, in the order that actually mattered:
1. **PP=1 (not PP=2)** — the single biggest lever. PP across nodes idled GPUs.
2. **`micro_batch_size=2` (with `global_batch_size=32`)** — took util 73%→85% by
   giving each GPU bigger GEMMs per step, amortizing the every-step all-reduce +
   MoE all-to-all. Next lever if more is needed: micro_batch=4/global_batch=64.
3. Megatron-SWIFT with EP, full SFT (not LoRA) — avoids naive-MoE's 10-20% util.
4. `--packing true` — critical for short function-calling examples.
5. `--max_length 8192` — longer packed sequences = more compute per batch.
6. `--moe_grouped_gemm true`, `--moe_permute_fusion true`,
   `--moe_shared_expert_overlap true` — fused/overlapped MoE kernels.
7. `--recompute_granularity full` — recompute activations (memory↔compute).
8. `--dataloader_num_workers 8` — avoid data-loading stalls.

Not a lever here: `--sequence_parallel true` — it is a no-op at TP=1 (Megatron
auto-disables it; SP only helps with tensor parallelism).

Pre-flight: run built-in NCCL all-reduce test (see Infrastructure section).

## Infrastructure

### Terraform

Fork the official Soperator recipe from `nebius-solution-library/soperator/installations/example/`. Both `nebius/nebius-solution-library` and `nebius/nebius-solutions-library` repos contain the same content.

Key files to adapt: `main.tf`, `terraform.tf`, `variables.tf`, `terraform.tfvars`, `driver_presets.tf`.

Key `.tfvars` settings:
```hcl
slurm_nodeset_workers = [{
  resource = {
    platform = "gpu-h200-sxm"
    preset   = "8gpu-128vcpu-1600gb"
  }
  size = 2
  gpu_cluster = {
    infiniband_fabric = "eu-north2-a"
  }
}]

filestore_jail = {
  spec = { size_gibibytes = 1024 }      # 1TB shared filesystem for jail
}

# 1TB network SSD via jail submount
filestore_jail_submounts = [{
  name = "data"
  spec = { size_gibibytes = 1024 }
  mount_path = "/data"
}]

public_o11y_enabled = false             # required workaround per assignment
telemetry_enabled   = true              # GPU monitoring via DCGM
```

Pre-requisite: install `yq` on the machine running terraform apply.

Note: tenant is `csa-hiring-sandboxK` (existing, shared). Other candidates may have resources deployed. Use unique naming for our resources.

### Storage layout

```
Shared filesystem (1TB, jail root):
  /                                # jail root, visible from all nodes
  ├── opt/slurm-test/quickcheck/   # built-in NCCL/hello/container tests
  └── shared/                      # our working directory
      ├── .cache/                  # HuggingFace / ModelScope model cache (~70GB)
      ├── code/                    # training scripts, configs
      ├── data/                    # prepared datasets (~2GB)
      ├── checkpoints/             # Megatron training checkpoints (2 × ~70GB = 140GB)
      ├── models/                  # exported HuggingFace checkpoint for vLLM (~70GB)
      └── logs/                    # Slurm job logs

Storage budget: ~70 + 2 + 140 + 70 = ~282GB used of 1TB. Safe margin.

Data submount (1TB, /data):
  Overflow storage. Use for model cache if jail fills up.
  Can also pre-download model weights here.
```

Guideline: do not use the same shared filesystem for 2 different jails.

### Container runtime

Soperator uses **enroot** (not Docker) via the **Pyxis SPANK plugin**. Training jobs run inside containers:
```bash
srun --container-image="nvcr.io/nvidia/pytorch:24.12-py3" nvidia-smi
```

Enroot pulls OCI/Docker images and runs them as unprivileged squashfs containers. Images are pulled at runtime; optionally cached on `node_local_image_disk`.

For Megatron-SWIFT, options:
1. Use ModelScope pre-built image via `--container-image=...`
2. Install dependencies into shared FS and run without container
3. Use NGC PyTorch base image + pip install ms-swift inside container

### Soperator workflow

1. `terraform apply` → Managed K8s + Soperator + SlurmCluster
2. SSH into login node via `login.sh` script (SSH key configured in tfvars)
3. Run pre-flight checks:
   - `sbatch /opt/slurm-test/quickcheck/hello.sh` (basic Slurm + nvidia-smi)
   - `sbatch /opt/slurm-test/quickcheck/containers.sh` (enroot validation)
   - `sbatch --nodes=2 /opt/slurm-test/quickcheck/nccl_multi_node.sh` (multi-node all_reduce, target: avg bus BW > 300 GB/s)
4. Upload code and data:
   - `scp -r training/ eval/ inference/ login-node:/shared/code/`
   - `scp data/train.jsonl data/eval.jsonl login-node:/shared/data/`
5. Pre-download model weights (separate from training job):
   - `sbatch predownload.slurm` (runs `huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir /shared/.cache/...`)
6. Submit training job via `sbatch train.slurm`
7. Monitor via `squeue`, `sacct`, `nvidia-smi dmon`, Nebius console GPU dashboards

### Monitoring

Three levels:
1. **Slurm**: `sinfo`, `squeue`, `sacct -j <id> --format=JobID,State,Elapsed,AllocTRES`
2. **GPU on worker**: SSH to worker, `nvidia-smi`, `nvidia-smi dmon -s pucm`, `nvtop`
3. **Nebius console**: Compute → Soperator → cluster → monitoring dashboards (DCGM metrics: GPU util, memory, power, network)

## Exercise 1: Training

### Model pre-download (run before training)

As-built: download from **ModelScope** (no auth gate), not HuggingFace, inside the
container, into `/data/models`. See `training/predownload.slurm`. Sketch:
```bash
#SBATCH --nodes=1 --ntasks-per-node=1 --output=/data/logs/predownload-%j.out
srun --container-image="nvcr.io/nvidia/pytorch:25.03-py3" --container-mounts="/data:/data" \
  python -c 'from modelscope import snapshot_download;
             snapshot_download("Qwen/Qwen3.6-35B-A3B", cache_dir="/data/models/Qwen3.6-35B-A3B")'
# Result is double-nested: /data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B
```

### Slurm training job

Uses a wrapper script to avoid shell quoting issues with `srun bash -c`:

**train.slurm:**
```bash
#!/bin/bash
#SBATCH --job-name=megatron-qwen36-moe-fc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/shared/logs/%x-%j.out
#SBATCH --error=/shared/logs/%x-%j.err

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# --export=ALL ensures MASTER_ADDR/MASTER_PORT propagate into srun tasks
srun --export=ALL /shared/code/training/run_megatron.sh
```

**run_megatron.sh** (executed by srun on each node):
```bash
#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MODELSCOPE_CACHE=/shared/.cache
export HF_HOME=/shared/.cache/huggingface
export NCCL_DEBUG=WARN

# SLURM_PROCID equals node rank when ntasks-per-node=1; more portable than SLURM_NODEID
NNODES=${SLURM_NNODES} \
NODE_RANK=${SLURM_PROCID} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=8 \
megatron sft \
    --model Qwen/Qwen3.6-35B-A3B \
    --dataset /shared/data/train.jsonl \
    --val_dataset /shared/data/eval.jsonl \
    --save_safetensors true \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 5e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --num_train_epochs 3 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --sequence_parallel true \
    --attention_backend flash \
    --output_dir /shared/checkpoints/qwen36-moe-fc \
    --eval_steps 200 \
    --save_steps 500 \
    --save_total_limit 2 \
    --no_save_optim true \
    --no_save_rng true
```

Note on container: if Megatron-SWIFT is installed inside the jail filesystem (shared FS), no `--container-image` is needed in `srun`. If using a Docker/enroot image instead, add `--container-image="..."` to the `srun` line in train.slurm and specify `--container-mounts=/shared:/shared`.

**Resuming from checkpoint** (if job crashes or hits time limit):

```bash
# Find last checkpoint
ls /shared/checkpoints/qwen36-moe-fc/

# Resume: add --resume_from_checkpoint to run_megatron.sh
NNODES=... megatron sft \
    ... \
    --resume_from_checkpoint /shared/checkpoints/qwen36-moe-fc/checkpoint-500
```


### Checkpoint export

> **As-built: NOT NEEDED.** Training ran with `--save_safetensors true`, which
> writes a complete HuggingFace checkpoint (config.json + model.safetensors.index.json
> + 16 safetensors shards + tokenizer) directly into each `checkpoint-N` dir. We
> verified `checkpoint-426` is a valid HF checkpoint — point vLLM straight at it.
> The Megatron→HF export job below is kept only as a reference for the case where
> `save_safetensors` is off; it is not part of the working pipeline.

Convert Megatron checkpoint to HuggingFace format for vLLM. **Export parallelism must match training parallelism**. Uses the same sbatch + separate script pattern as training to avoid shell quoting issues:

**export_checkpoint.slurm:**
```bash
#!/bin/bash
#SBATCH --job-name=export-checkpoint
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/shared/logs/%x-%j.out

set -euo pipefail

# ms-swift names checkpoints checkpoint-N, not checkpoint-final
export CKPT=$(ls -d /shared/checkpoints/qwen36-moe-fc/checkpoint-* | sort -V | tail -1)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501

srun --export=ALL /shared/code/training/run_export.sh
```

**run_export.sh:**
```bash
#!/bin/bash
set -euo pipefail

export HF_HOME=/shared/.cache/huggingface
export MODELSCOPE_CACHE=/shared/.cache

NNODES=2 \
NODE_RANK=${SLURM_PROCID} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=8 \
megatron export \
    --mcore_model ${CKPT} \
    --output_dir /shared/models/qwen36-moe-fc-hf \
    --to_hf true \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --test_convert_precision true
```

**Fast-path check** — try this first. If `--save_safetensors true` was used during training with mcore-bridge, the checkpoint may already be HF-loadable:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/shared/checkpoints/qwen36-moe-fc/checkpoint-500", device_map="cpu"
)
```
If that succeeds, skip the export step entirely and point vLLM directly at the checkpoint directory.

## Exercise 2: Inference Comparison

### Serving

Two vLLM instances. 35B MoE in bf16 ≈ 70GB, fits on a single H200 (141GB). Use TP=2 for comfortable memory headroom (not TP=8 which adds unnecessary communication overhead).

**How to run on Soperator**: vLLM servers need a GPU worker node, not the login node. Use `salloc` to get an interactive allocation:

```bash
# From login node: allocate 1 node interactively
salloc --nodes=1 --gpus-per-node=4 --exclusive --time=02:00:00
# Then SSH to the allocated node, or use srun:
srun --pty bash
```

**serve_base.sh** (run on worker node):
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.6-35B-A3B \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --host 0.0.0.0 --port 8000
```

**serve_tuned.sh** (run on same worker node, different terminal):
```bash
CUDA_VISIBLE_DEVICES=2,3 vllm serve /shared/models/qwen36-moe-fc-hf \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --host 0.0.0.0 --port 8001
```

Both run simultaneously on the same node (4 GPUs total). `--chat-template-kwargs '{"enable_thinking": false}'` matches the training configuration — without this, Qwen3.6 may enable thinking mode and produce `<think>...</think>` preambles that break the tool call parser.

### Comparison

Send identical prompts to both endpoints with `temperature=0`. compare.py handles:
1. Load prompts from eval/prompts.jsonl
2. Send each prompt to base and tuned endpoints
3. Parse tool calls from responses
4. Compute metrics
5. Generate comparison table + side-by-side examples

Metrics:

| Metric | What it measures |
|--------|-----------------|
| valid_tool_call_rate | % outputs parseable as valid tool call |
| function_name_accuracy | % correct function selected |
| argument_exact_match | % all arguments correct |
| appropriate_call_rate | % correct call/no-call/clarify decisions |
| executable_success_rate | % tool calls executable against mock functions |

Test categories (50 demo examples):
- 15 simple single-tool calls
- 10 multi-argument calls
- 5 missing-required-argument (should clarify)
- 5 no-suitable-tool (should not call)
- 10 process-automation domain examples
- 5 ambiguous tool choice

## Repository Structure

```
nebius-demo/
  PLAN.md                              # this file
  ASSESMENT.md                         # assignment description
  CLAUDE.md                            # project context for Claude Code
  README.md                            # reproduction guide
  pyproject.toml                       # uv project config

  terraform/
    main.tf                            # forked from nebius-solution-library/soperator
    terraform.tf
    variables.tf
    terraform.tfvars
    driver_presets.tf
    outputs.tf

  training/
    prepare_dataset.py                 # download HF dataset, parse inline, apply_chat_template → train/eval JSONL
    validate_dataset.py                # count, validate JSON, token stats, sample review
    predownload.slurm                  # ModelScope download of the model to /data/models (separate from training)
    import_image.slurm                 # enroot import official ms-swift image → /data/images/swift431.sqsh
    build_image_tl.slurm               # derive swift431-tl.sqsh = base image + tilelang 0.1.9
    train.slurm                        # Slurm job (sets MASTER_ADDR/NNODES, monitoring, calls srun w/ container)
    run_megatron.sh                    # `megatron sft` command (executed by srun per node; sets NODE_RANK)
    watch_gpu.sh                       # live GPU util of a running job (srun --overlap nvidia-smi)
    # (no export_checkpoint.slurm — save_safetensors makes the checkpoint HF-ready)

  inference/
    serve_base.sh                      # vLLM base model (TP=2, port 8000)
    serve_tuned.sh                     # vLLM fine-tuned model (TP=2, port 8001)

  eval/
    prompts.jsonl                      # 50 hand-written test cases with expected tool calls (not pipeline output)
    mock_tools.py                      # mock function implementations for executable-test scoring
    compare.py                         # call both endpoints, evaluate, generate report

  docs/
    architecture.md                    # architecture diagram and decisions
    monitoring.md                      # GPU monitoring commands and dashboards
    troubleshooting.md                 # common issues and fixes
    demo_script.md                     # presentation outline
```


## Risks and Fallbacks

| Risk | Likelihood | Impact | Fallback |
|------|-----------|--------|----------|
| Qwen3.6-35B-A3B not supported in Megatron-SWIFT | Medium | High | Use Qwen3-30B-A3B-Instruct-2507 (tested). **Warning**: Qwen3-30B uses Hermes JSON format (not XML). Must also change: vLLM `--tool-call-parser hermes`, prompts.jsonl expected format, compare.py parsing. |
| Megatron-SWIFT install fails in Soperator jail | Medium | High | Use ModelScope Docker image via enroot; if that fails, NGC PyTorch + pip |
| Checkpoint export fails (parallelism mismatch) | Low | High | Check EP=8 PP=2 in export Slurm script; alternatively test if `--save_safetensors true` checkpoint is directly loadable by vLLM |
| GPU utilization < 80% | Low | Medium | Increase global_batch_size, max_length; enable all fusion flags |
| NCCL multi-node communication fails | Low | High | Check InfiniBand via quickcheck scripts, tune NCCL env vars |
| vLLM cannot serve exported checkpoint | Low | Medium | Use `swift infer --infer_backend vllm` instead of raw vLLM CLI |
| hypervariance dataset conversion yields < 50k examples | Medium | Medium | Supplement with xLAM (auto-approve gate, ~60k examples) |
| Qwen3.6 apply_chat_template behaves unexpectedly | Low | Medium | Test locally first; fallback to Qwen3-30B-A3B Hermes JSON format |
| enroot cannot pull ModelScope image | Medium | Medium | Install deps into shared FS; use NGC PyTorch base image |
| Catastrophic forgetting degrades demo | Medium | Medium | Focus comparison on function-calling only; add general data to mix if time |
| Shared tenant resource conflicts | Low | Medium | Use unique resource names; check existing resources before terraform apply |
| Storage overflow from checkpoints | N/A | N/A | Fixed: save_steps=500, save_total_limit=2 |

Critical fallback: if MoE + Megatron-SWIFT is completely blocked, switch to **Qwen2.5-72B-Instruct + TRL + DeepSpeed ZeRO-3 + LoRA**. This is the proven dense path that guarantees >80% GPU utilization with standard tooling. Loses the "2026 model" angle but guarantees a passing submission.

**What actually happened (outcomes):** the two "Medium" risks materialized and were
resolved without falling back. (1) *Megatron-SWIFT install* — pip/uv was a dead end
(4.x `[megatron]` not on PyPI); resolved by using the official enroot image. (2) A
risk not in this table — *Qwen3.6 Gated DeltaNet incompatible with Hopper/Triton≥3.4*
— surfaced as a first-backward crash and was resolved by baking `tilelang 0.1.9`
into the image. *GPU util <80%* also occurred (PP=2 and micro_batch=1) and was fixed
by PP=1 + micro_batch=2. The dense fallback was never needed.

## Timeline

| Day | Focus | Deliverable | Go/No-Go |
|-----|-------|-------------|----------|
| 1 (Tue Jun 24) | Terraform deploy, Soperator access, NCCL test. In parallel: dataset prep locally (normalize, prepare, validate). | Working Slurm cluster + validated train.jsonl | If NCCL test fails → debug InfiniBand. If terraform fails → check tenant resources. |
| 2 (Wed Jun 25) | Megatron-SWIFT env setup (enroot or shared FS install). Upload data. Pre-download model. Single-node smoke test (1 step). | Training runs on 1 node, 1 step completes | If Megatron-SWIFT install fails → try ModelScope Docker. If Qwen3.6 fails → fall back to Qwen3-30B-A3B. |
| 3 (Thu Jun 26) | Multi-node training run. Monitor GPU utilization. Iterate config if < 80%. | Full training run, >80% GPU util confirmed | If GPU util < 80% → increase batch/seq length. If MoE+Megatron completely blocked → SWITCH TO DENSE FALLBACK (Qwen2.5-72B + TRL). |
| 4 (Fri Jun 27) | Checkpoint export. vLLM serving (base + tuned). Run comparison. | Base vs tuned comparison results | |
| 5 (Sat Jun 28) | README, docs, evaluation polish. | Complete documentation | |
| 6 (Sun Jun 29) | Demo script, presentation prep. | Presentation ready | |
| 7 (Mon Jun 30) | Final review, submit. | Email sent with Terraform code | |

Note: deadline per ASSESMENT.md is June 26 (1 week from June 19). If confirmed as June 26, compress days 1-3 into a 48-hour sprint and prioritize Exercise 1 pass over Exercise 2. Day 3 is the hard go/no-go for the MoE path.

## Demo Narrative (5 min)

1. **Customer problem** (30s): Small ML team needs to validate Nebius before committing to 512 H100s. They want end-to-end function-calling fine-tuning.

2. **Why Soperator + Megatron** (1min): Soperator gives familiar Slurm workflows on Kubernetes infrastructure. Megatron-SWIFT with Expert Parallelism is the recommended way to train MoE models -- 10x faster than standard frameworks. For a team without deep infra expertise, Slurm is the most accessible distributed training interface.

3. **Architecture** (1min): 2 nodes, 16 H200 GPUs, EP=8 per node. Shared filesystem for data/checkpoints/models. Enroot containers for reproducible environment. Show diagram.

4. **Training walkthrough** (1min): Dataset preparation (function-calling data + customer domain examples), NCCL pre-flight, sbatch submission, monitoring (squeue, nvidia-smi, Nebius dashboard). Show >80% GPU utilization screenshot.

5. **Results** (1min): Base vs tuned comparison table. Show specific examples where tuned model improves tool selection, argument extraction, and appropriate call/no-call decisions for process automation.

6. **What customer can reuse** (30s): Terraform code, Slurm scripts, dataset pipeline, monitoring guide, inference setup. Path to scaling: same Megatron-SWIFT + EP approach works on 512 H100s with more nodes and larger MoE models (e.g., Qwen3.5-122B-A10B).
