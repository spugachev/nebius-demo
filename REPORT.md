# Multi-Node Function-Calling Fine-Tuning on Nebius Soperator — Report

## Summary

This project delivers an end-to-end, multi-node LLM fine-tuning workflow on
[Nebius Soperator](https://github.com/nebius/soperator) (Slurm on Managed
Kubernetes), built to show a PoC team how they would train and serve a
function-calling model on reserved GPU capacity. We fine-tune
[**Qwen3.6-35B-A3B**](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B) — a 2026
open-weight Mixture-of-Experts model (256 experts, ~3B active of 35B, with hybrid
Gated-DeltaNet attention) that is already instruction-tuned — for tool/function
calling. Training uses [**Megatron-SWIFT**](https://github.com/modelscope/ms-swift)
(ms-swift on a Megatron-Core backend) with **full supervised fine-tuning and
[Expert Parallelism](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html)**
across **2 nodes × 8 [H200](https://www.nvidia.com/en-us/data-center/h200/) GPUs** —
the Qwen-team-recommended path for MoE that keeps GPUs busy where naive MoE training
stalls at 10–20%. The data is the public
[**hypervariance/function-calling-sharegpt**](https://huggingface.co/datasets/hypervariance/function-calling-sharegpt)
corpus (~75k usable conversations) normalized through the tokenizer's chat template.
The run completed in **51 minutes** at **85–92% sustained GPU utilization** (the
assignment's >80% target), verified on the Nebius DCGM dashboard. We then serve the
base and fine-tuned models side-by-side with [**vLLM**](https://github.com/vllm-project/vllm)
and evaluate on a 23-case function-calling suite
([`eval/results/comparison.md`](eval/results/comparison.md)): the tuned model improves
**function-name accuracy 82→100%**, **argument exact-match 76→94%**, and
**appropriate call/no-call 87→96%**.

## Background & Objective

The customer is a small (<20-person), VC-funded startup building process-automation
products with AI agents, evaluating Nebius for a **512× H100, 6-month reservation**
and currently at the proof-of-concept stage. The PoC team is mostly ML engineers
with limited cloud and infrastructure experience, so the example has to be
reproducible and approachable rather than bespoke. The goal is an end-to-end,
multi-node fine-tuning workflow for an open-source function-calling LLM on the
allocated PoC capacity (**2 nodes × 8 [H200](https://www.nvidia.com/en-us/data-center/h200/)
GPUs**, fabric `eu-north2-a`, shared SSD storage), using the GPUs efficiently
(**>80% utilization**, verifiable on the Nebius console) and comparing the base and
fine-tuned models. Scheduler, fine-tuning framework, storage, and dataset are left
to the implementer; this report documents and justifies each of those choices.

## Infrastructure

The cluster is provisioned with [Terraform](terraform/), forked from the official
[`nebius-solution-library`](https://github.com/nebius/soperator) Soperator recipe.
**Soperator** runs Slurm on top of Managed Kubernetes, which is the deliberate
choice here: it gives the customer's ML engineers a familiar `sbatch`/`squeue`/`sacct`
workflow without requiring Kubernetes expertise, while Nebius manages the control
plane. The deployment is **2 worker nodes × 8 H200 (141 GB each)** on the
`eu-north2-a` InfiniBand fabric, with a single **shared filesystem mounted at
`/data`** on every node holding code, the model, datasets, the container image,
checkpoints, and logs.

The container runtime is **[enroot](https://github.com/NVIDIA/enroot) via the Pyxis
SPANK plugin** (not Docker): jobs run with `srun --container-image=...`, and a single
`.sqsh` image on the shared filesystem is reused by both nodes for a reproducible,
drift-free environment. Before training we ran the built-in NCCL multi-node
all-reduce check and measured **~475 GB/s** bus bandwidth (target > 300 GB/s),
confirming the inter-node interconnect. Operational details (tenant
`csa-hiring-sandboxK`, the `public_o11y_enabled = false` recipe workaround, storage
sizing) are in [`CLAUDE.md`](CLAUDE.md); the full topology diagram is in
[`docs/architecture.md`](docs/architecture.md).

## Model and Framework

**Model — [Qwen3.6-35B-A3B](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B).** A
2026 open-weight (Apache-2.0) Mixture-of-Experts model: 256 experts with top-8
routing (~3B active parameters of 35B total), 40 layers, and a **hybrid attention
stack** that interleaves full attention with **Gated DeltaNet** (linear-attention)
layers. It ships already instruction-tuned, so we do task-specialization rather than
base alignment. We chose MoE over a dense model because no dense open-weight model
above 32B was released in 2026 — the field moved to MoE — and training one is the
more representative, forward-looking demonstration for a customer reserving large
capacity.

**Framework — [Megatron-SWIFT](https://github.com/modelscope/ms-swift)** (ms-swift on
a Megatron-Core backend) with **full supervised fine-tuning and Expert Parallelism**.
This is the path the Qwen team recommends for MoE: it delivers roughly an order of
magnitude more throughput than HuggingFace/DeepSpeed for MoE and, crucially, keeps
GPU utilization high — naive MoE training (e.g. DeepSpeed ZeRO-3 + LoRA) is reported
to stall at 10–20% and has documented incompatibilities. We use **full SFT rather
than LoRA** because LoRA on MoE with Expert Parallelism at this scale is undocumented
and risks falling below the 80% utilization requirement; full SFT is the proven path
to >80%. Catastrophic forgetting from full SFT is mitigated by keeping non-tool
conversations in the data mix.

**Tool-call format.** Qwen3.6 emits XML-style tool calls
(`<function=name><parameter=key>value</parameter></function>`), so inference uses the
vLLM `qwen3_coder` parser (not Hermes JSON). Tool schemas and rendering are handled
by the tokenizer's `apply_chat_template`, never by hand.

## Dataset

The primary dataset is the public
[**hypervariance/function-calling-sharegpt**](https://huggingface.co/datasets/hypervariance/function-calling-sharegpt)
corpus (~87k examples, no access gate). It is in ShareGPT form — a single
`conversations` column — with tool schemas embedded as **free text** in the system
message, function calls wrapped in `<functioncall>` tags, and arguments that are
sometimes double-stringified. [`training/prepare_dataset.py`](training/prepare_dataset.py)
parses this inline: it extracts the JSON tool schemas, converts `<functioncall>` tags
into structured `tool_calls`, handles the double-stringified arguments, maps roles,
and drops unparseable examples (~75k survive). Every example is then rendered through
`tokenizer.apply_chat_template(..., enable_thinking=False)` so the training signal
exactly matches what the model will see at inference, and split 96/4 into train/eval
([`validate_dataset.py`](training/validate_dataset.py) checks counts, JSON validity,
and token statistics). With **sequence packing** to 8192 tokens, the ~75k short
conversations compact into **4,558 train + 191 eval packed sequences** (~8,131 tokens
each). `enable_thinking=False` is kept throughout — it yields a simpler training
signal, lower inference latency, and cleaner output for an automation integration.

## Training

Training runs from the **official ms-swift enroot image** with `tilelang` baked in
(`swift431-tl.sqsh`; see *Engineering Challenges*), invoked as `megatron sft` via
[`train.slurm`](training/train.slurm) → [`run_megatron.sh`](training/run_megatron.sh)
(an sbatch script that sets `MASTER_ADDR`/`NNODES` and an `srun` runner that sets
`NODE_RANK` per node — the separation avoids quoting issues and rank collisions).

The parallelism is **PP=1, TP=1, EP=8 ⇒ data-parallel size 16**. Expert Parallelism
shards the 256 experts across 8 ranks; with no pipeline parallelism across nodes,
both nodes run as full data-parallel replicas and stay busy (the only inter-node
traffic is an overlappable gradient all-reduce, plus the MoE all-to-all). Key knobs:
`micro_batch_size=2`, `global_batch_size=32`, `--packing true`, `--recompute_granularity
full`, `--attention_backend flash`, the MoE fusions (`grouped_gemm`, `permute_fusion`,
`shared_expert_overlap`), `lr=5e-6` (lower than the 1e-5 reference because the model is
already instruction-tuned), bf16, 3 epochs. `--save_safetensors true` makes each
checkpoint a complete HuggingFace directory, so **no Megatron→HF export step is
needed**.

The verified run (job 65) **completed in 51m33s**, 426 steps, loss **0.20 → 0.13**,
eval loss **0.196**, peak memory **113 / 143 GB per GPU** (no OOM). The final
checkpoint (`checkpoint-426`, 66 GB of safetensors) is directly loadable by vLLM.

## GPU Utilization

The assignment's key metric is **DCGM_FI_DEV_GPU_UTIL** (fraction of time the GPU
executes kernels). We **sustained 85–92% across all 16 H200 GPUs** (mean ≈ 85%) during
steady-state training — comfortably above the >80% target. It was measured two ways:
live from the cluster with an overlapping step on the job's own allocation
(`srun --jobid=N --overlap … nvidia-smi`, wrapped by
[`training/watch_gpu.sh`](training/watch_gpu.sh)), and on the **Nebius console DCGM
dashboard** (GPU-metrics view), which agrees. Getting there took two changes from the
initial design: switching from pipeline parallelism (`PP=2`, which idled GPUs with
cross-node pipeline bubbles, ~73% or worse) to **data-parallel `PP=1`**, and raising
**`micro_batch_size` 1 → 2** so each GPU does enough compute per step to amortize the
all-reduce and MoE all-to-all (73% → 85%). The measurement method and a "healthy run"
reference are in [`docs/monitoring.md`](docs/monitoring.md).

## Inference and Evaluation

Both models are served with [**vLLM**](https://github.com/vllm-project/vllm) 0.23.0
(from the same image) at tensor-parallel size 2 — 35B in bf16 (~70 GB) fits
comfortably on 2 H200s, and TP=8 would only add communication overhead. A smoke test
first de-risked that vLLM loads the hybrid Gated-DeltaNet architecture and emits a
parsed tool call via `--tool-call-parser qwen3_coder`. For an apples-to-apples
comparison both models run with thinking disabled (`chat_template_kwargs={"enable_thinking":
false}` per request), matching training.

[`eval/run_comparison.slurm`](eval/run_comparison.slurm) stands up base (`:8000`) and
tuned (`:8001`) on one node and runs [`eval/compare.py`](eval/compare.py), a
dependency-light harness that sends [23 hand-written prompts](eval/prompts.jsonl)
across six categories (single-tool, multi-argument, should-clarify, no-tool-needed,
process-automation, ambiguous), parses the tool calls, and scores them — including
*executability* against [mock tool implementations](eval/mock_tools.py). The
fine-tuned model improves on every aggregate metric
([full report](eval/results/comparison.md)):

| Metric | Base | Tuned | Δ |
|---|---:|---:|---:|
| Appropriate call / no-call | 87.0% | **95.7%** | +8.7 |
| Function-name accuracy (call cases) | 82.4% | **100.0%** | +17.6 |
| Argument exact-match (call cases) | 76.5% | **94.1%** | +17.6 |
| Executable against mock tools | 65.2% | **78.3%** | +13.1 |

Concretely, the tuned model now reliably translates ("good morning" → Japanese),
resolves ambiguous asks (`get_stock_price(TSLA)`; `set_reminder` instead of
`get_weather`), and fills multi-argument process-automation calls (tickets, refunds,
VM provisioning) correctly. There is one honest trade-off: the *clarify* rate dipped
(100% → 67%) — function-calling SFT makes the model more eager to call a tool rather
than ask for a missing argument. That is the expected effect and the clear next
improvement: add more negative/clarification examples to the data mix.

## Engineering Challenges

The interesting work was diagnosing and *properly* fixing several non-obvious issues
(no runtime patches — see [`docs/troubleshooting.md`](docs/troubleshooting.md) for the
full log):

- **ms-swift 4.x `[megatron]` is not on PyPI.** Every pip/uv install backtracked
  forever or pulled an incompatible torch (`No module named 'torch.multiprocessing'`).
  Fix: use the **official ms-swift docker image** via enroot — it ships a consistent
  torch 2.11 / TransformerEngine / flash-attn / megatron-core / vLLM stack.
- **Qwen3.6's Gated DeltaNet crashes on the first backward on Hopper.** Its
  `fla` Triton kernel is numerically wrong on Triton ≥ 3.4 (the image has 3.6), so it
  hard-requires the **`tilelang`** backend. Fix: bake `tilelang==0.1.9` into a derived
  image, pinned to the image's `apache-tvm-ffi 0.1.9` (shared with vLLM) so nothing
  upgrades and FFI types don't double-register.
- **Pipeline parallelism idled GPUs**, and **batch divisibility**: with PP=1, the data
  parallel size is 16 (EP is orthogonal), so `global_batch` must be a multiple of
  `micro_batch × 16` — `micro_batch=2` ⇒ `global_batch=32`.
- **vLLM 0.23.0 has no `--chat-template-kwargs` CLI flag** (servers exit at arg-parse);
  thinking is disabled **per request** in the body instead.

## Reproducibility and Scaling

The whole workflow is scripted and documented for reproduction — provision with
Terraform, prepare data locally, then four `sbatch` jobs (predownload → import image →
build image → train), with serving and evaluation as two more; step-by-step commands
are in the [`README.md`](README.md). Everything the customer keeps is reusable: the
Terraform cluster definition, the Slurm scripts, the dataset pipeline, the monitoring
guide, and the inference + evaluation harness.

The approach **scales to the planned 512× H100 reservation without changing the
workflow**: the same Megatron-SWIFT + Expert-Parallelism recipe extends by adding
nodes (more data-parallel replicas) and/or moving to larger MoE models, while the
Slurm interface, container image, and shared-filesystem layout stay identical.

## Conclusion

Both exercises are complete. We fine-tuned a 2026 35B MoE function-calling model
across 16 H200 GPUs on Nebius Soperator at **85–92% sustained GPU utilization**
(requirement: >80%) in under an hour, and demonstrated a **measurable quality
improvement** of the tuned model over the base (function-name accuracy 82 → 100%,
argument exact-match 76 → 94%). The result is a reproducible, well-documented,
end-to-end template the PoC team can run themselves and scale to their full
reservation. A 5-minute demo-day walkthrough is in
[`docs/demo_script.md`](docs/demo_script.md).
