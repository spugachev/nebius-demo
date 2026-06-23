# Implementation Plan

Deadline: June 26, 2026.

## Summary

Fine-tune a 2026 open-source MoE LLM for function calling on 16 H200 GPUs using Nebius Soperator, then serve and compare base vs tuned model.

Stack: Megatron-SWIFT with Expert Parallelism, Qwen3.6-35B-A3B, Slurm via Soperator, vLLM for inference.

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

Environment setup options:
- **Option A (recommended)**: Use ModelScope Docker image via enroot: `srun --container-image="modelscope-registry/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3" ...`
- **Option B**: Install into shared filesystem pip venv accessible from all nodes.
- If neither works, fall back to NGC PyTorch base image + pip install ms-swift.

Parallelism config for 2 nodes x 8 GPUs:
- `expert_model_parallel_size = 8` (experts split across 8 GPUs within a node)
- `pipeline_model_parallel_size = 2` (pipeline across 2 nodes)
- Data parallelism is computed automatically: 16 / (EP * PP) = 1

Alternative parallelism (if PP causes issues):
- `expert_model_parallel_size = 8`
- `pipeline_model_parallel_size = 1`
- DP = 16 / 8 = 2 (data parallel across 2 nodes)

## Dataset

### Strategy: structured messages → apply_chat_template

All source datasets are normalized to a structured intermediate JSONL format with `messages` (OpenAI-style with `tool_calls`) and `tools` fields. Then `prepare_dataset.py` runs `tokenizer.apply_chat_template()` to produce the final training text in correct Qwen3.6 XML format.

This approach guarantees format correctness regardless of source dataset format or Qwen version differences. Confirmed: Qwen3.6 tokenizer natively supports structured `tool_calls` in assistant messages.

### Intermediate format

Single-turn (most examples):
```json
{
  "messages": [
    {"role": "user", "content": "Create a ticket for Acme about failed payment"},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {
        "name": "create_ticket",
        "arguments": {"customer": "Acme", "category": "payment_failure"}
      }}
    ]}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "create_ticket",
      "description": "Create a support ticket",
      "parameters": {
        "type": "object",
        "properties": {
          "customer": {"type": "string"},
          "category": {"type": "string"}
        },
        "required": ["customer", "category"]
      }
    }}
  ]
}
```

Multi-turn with tool response (required for hypervariance dataset):
```json
{
  "messages": [
    {"role": "user", "content": "What's the status of order 12345?"},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {
        "name": "check_order_status",
        "arguments": {"order_id": "12345"}
      }}
    ]},
    {"role": "tool", "name": "check_order_status", "content": "{\"status\": \"shipped\", \"eta\": \"2026-06-24\"}"},
    {"role": "assistant", "content": "Order 12345 has shipped and is expected by June 24."}
  ],
  "tools": [...]
}
```

For non-tool-call conversations (model declines to call a tool), the assistant message has `content` only, no `tool_calls`.

### Final ms-swift format (produced by prepare_dataset.py)

```json
{
  "messages": [
    {"role": "system", "content": "# Tools\n\nYou may call one or more functions...\n\n<tools>\n[...]\n</tools>"},
    {"role": "user", "content": "Create a ticket for Acme about failed payment"},
    {"role": "assistant", "content": "<tool_call>\n<function=create_ticket>\n<parameter=customer>Acme</parameter>\n<parameter=category>payment_failure</parameter>\n</function>\n</tool_call>"}
  ]
}
```

### Sources

**Primary: hypervariance/function-calling-sharegpt** (87k examples)
- Public, Apache 2.0, no access gate
- ShareGPT format: single `conversations` column with `from`/`value` objects
- Roles: `system`, `human`, `gpt`, `function_response`
- Tool schemas are embedded as free text in the system message `value` (not a separate structured field)
- Function calls use `<functioncall>` tags: `<functioncall> {"name": "func", "arguments": {...}} </functioncall>`
- Arguments are sometimes double-stringified (JSON string inside JSON)
- ~15% of examples are non-tool conversations (model declines) -- useful for catastrophic forgetting mitigation
- Filtered from glaive-v2 (removed invalid JSON, code examples)

**Conversion effort (normalize_sharegpt.py)**: MEDIUM, not low as initially estimated.
- Parse system message free text to extract JSON function schemas → structured `tools` field
- Parse `<functioncall>` tags from assistant messages → structured `tool_calls`
- Handle double-stringified arguments (e.g., `'{"key": "value"}'` string inside JSON)
- Map roles: `human`→`user`, `gpt`→`assistant`, `function_response`→`tool`
- Validate: skip examples with unparseable schemas or malformed function calls
- Expected yield: ~75-80k valid examples from 87k raw (some will fail parsing)

**Custom supplement: process-automation examples** (1,000 examples)
- Generated by `generate_custom_examples.py` using templates with fixed random seed for reproducibility
- 16 tool schemas matching customer domain: create_ticket, update_ticket, close_ticket, create_crm_lead, update_crm_record, schedule_meeting, cancel_meeting, approve_invoice, reject_invoice, extract_invoice_fields, send_follow_up_email, assign_task, escalate_case, search_policy, check_order_status, get_customer_info
- Distribution: ~75% normal tool calls, ~10% no tool needed, ~10% should clarify (missing required args), ~5% ambiguous tool choice

### Splits

```
Train:   ~65,000  (62k hypervariance valid + 750 custom tool-call + 250 custom negative)
Eval:     ~2,700  (2.5k hypervariance held-out + 200 custom held-out)
```

The 50 demo/test prompts are hand-written in `eval/prompts.jsonl` (not pipeline output). See Repository Structure.

Note: train size reduced from 80k to ~65k to account for parsing failures during normalization. Still produces ~130M tokens, sufficient for ~3 hours of training.

### Thinking mode

Disabled (`enable_thinking=False` in apply_chat_template). Reasons:
- Simpler training signal (no `<think>` blocks to learn)
- Lower inference latency (no reasoning preamble)
- Cleaner for customer integration
- Can be shown as opt-in feature in demo

### Training data volume estimate

- ~65k examples × ~2000 tokens avg = ~130M tokens
- With packing into 8192 seq length ≈ 16k packed sequences
- global_batch_size=16 → ~1,000 steps per epoch
- At ~10s/step = ~2.8 hours for 1 epoch
- Sufficient for meaningful GPU dashboard graphs

### Dataset pipeline

```
hypervariance/function-calling-sharegpt (87k, HuggingFace)
         │
         ▼
    normalize_sharegpt.py
    (parse system text → extract tool schemas,
     parse <functioncall> → structured tool_calls,
     map roles, validate, filter broken examples)
         │
         ▼
    intermediate.jsonl (~75k valid structured examples)
         │
         ├── + custom_process_automation.jsonl (1k, template-generated with fixed seed)
         │
         ▼
    prepare_dataset.py
    (load Qwen3.6 tokenizer,
     apply_chat_template(messages, tools, enable_thinking=False),
     split train/eval/demo,
     validate output,
     write final JSONL)
         │
         ▼
    train.jsonl  (~65k, ready for Megatron-SWIFT)
    eval.jsonl   (~2.7k held-out)
         │
         ▼
    validate_dataset.py
    (count examples per split,
     check JSON validity,
     print token length distribution,
     sample 5 examples for manual review)
```

## GPU Utilization Strategy

Nebius dashboards show DCGM_FI_DEV_GPU_UTIL ("percentage of time a GPU spends executing tasks"). This is the standard nvidia-smi GPU-Util metric, not SM occupancy or MFU.

Target: >80% on all 16 GPUs during steady-state training.

Levers (in priority order):
1. Megatron-SWIFT with EP (full SFT, not LoRA) -- eliminates the 10-20% utilization problem of naive MoE training
2. `--packing true` -- critical for short function-calling examples
3. `--max_length 8192` -- longer packed sequences = more compute per batch
4. `--moe_grouped_gemm true` -- optimized batched matrix multiply for experts
5. `--moe_permute_fusion true` -- fused token permutation kernels
6. `--moe_shared_expert_overlap true` -- overlaps shared expert computation with communication
7. `--sequence_parallel true` -- distributes sequence dimension across TP ranks
8. `--recompute_granularity full` -- activation recomputation trades memory for compute
9. Increase `--global_batch_size` if utilization is still low
10. Increase `--dataloader_num_workers` to prevent data loading bottleneck

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

```bash
#!/bin/bash
#SBATCH --job-name=predownload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/logs/%x-%j.out

export HF_HOME=/shared/.cache/huggingface
huggingface-cli download Qwen/Qwen3.6-35B-A3B
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
    --num_train_epochs 1 \
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

Key changes from previous version:
- Split into train.slurm + run_megatron.sh to avoid shell quoting issues
- Added `--val_dataset` for evaluation during training
- `--save_steps 500` (was 100) to prevent storage overflow: 2 checkpoints × ~70GB = 140GB
- `--save_total_limit 2` to keep only the last 2 checkpoints
- Added `--eval_steps 200` (less frequent eval to avoid slowdowns)
- `NCCL_DEBUG=WARN` instead of INFO (less log noise)
- `NODE_RANK` now uses `SLURM_PROCID` (more portable than `SLURM_NODEID`)
- `srun --export=ALL` to guarantee MASTER_ADDR propagation
- LR lowered to `5e-6` (from 1e-5): full SFT on an already-instruction-tuned model at 1e-5 risks unstable loss; 5e-6 is safer
- Removed `--load_from_cache_file true` (not a valid ms-swift arg)

### Checkpoint export

Convert Megatron checkpoint to HuggingFace format for vLLM. **Export parallelism must match training parallelism** (EP=8, PP=2). This requires both nodes:

```bash
# export_checkpoint.sh — run as a 2-node Slurm job
#!/bin/bash
#SBATCH --job-name=export-checkpoint
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/shared/logs/%x-%j.out

# Find the latest checkpoint (ms-swift names them checkpoint-N, not checkpoint-final)
CKPT=$(ls -d /shared/checkpoints/qwen36-moe-fc/checkpoint-* | sort -V | tail -1)

srun --export=ALL bash -c "
  NNODES=2 NODE_RANK=\${SLURM_PROCID} \
  NPROC_PER_NODE=8 \
  megatron export \
      --mcore_model ${CKPT} \
      --output_dir /shared/models/qwen36-moe-fc-hf \
      --to_hf true \
      --pipeline_model_parallel_size 2 \
      --expert_model_parallel_size 8 \
      --test_convert_precision true
"
```

Alternative: if `--save_safetensors true` was used during training with mcore-bridge, the checkpoint may already be HF-loadable. Test with:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/shared/checkpoints/qwen36-moe-fc/checkpoint-500")
```
If that succeeds, skip the export step entirely.

## Exercise 2: Inference Comparison

### Serving

Two vLLM instances. 35B MoE in bf16 ≈ 70GB, fits on a single H200 (141GB). Use TP=2 for comfortable memory headroom (not TP=8 which adds unnecessary communication overhead):

```bash
# Base model (2 GPUs, node 0)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.6-35B-A3B \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --host 0.0.0.0 --port 8000

# Fine-tuned model (2 GPUs, node 0)
CUDA_VISIBLE_DEVICES=2,3 vllm serve /shared/models/qwen36-moe-fc-hf \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --host 0.0.0.0 --port 8001
```

Both can run simultaneously on the same node (only 4 GPUs needed total).

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
    normalize_sharegpt.py              # convert hypervariance ShareGPT → intermediate JSONL
    generate_custom_examples.py        # generate process-automation examples (template + fixed seed)
    prepare_dataset.py                 # apply_chat_template → final ms-swift JSONL + train/eval split
    validate_dataset.py                # count, validate JSON, token stats, sample review
    train.slurm                        # Slurm job script (sets MASTER_ADDR, calls srun)
    run_megatron.sh                    # Megatron-SWIFT training command (executed by srun per node)
    predownload.slurm                  # pre-download model weights to shared cache (separate from training)
    export_checkpoint.slurm            # Megatron → HuggingFace conversion (2-node job, EP must match training)

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

Changes from previous version:
- Added `validate_dataset.py` (was missing)
- Added `run_megatron.sh` (split from train.slurm to fix quoting)
- Added `predownload.slurm` (model download separate from training)
- Consolidated eval: `eval/report.py` removed, `inference/compare.py` moved to `eval/compare.py`. Single script does: call endpoints → evaluate → report.
- Removed `eval/evaluate.py` (merged into compare.py)

## Risks and Fallbacks

| Risk | Likelihood | Impact | Fallback |
|------|-----------|--------|----------|
| Qwen3.6-35B-A3B not supported in Megatron-SWIFT | Medium | High | Use Qwen3-30B-A3B-Instruct-2507 (tested) |
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
