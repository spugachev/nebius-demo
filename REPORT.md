# Multi-Node Function-Calling Fine-Tuning on Nebius Soperator

**Scope:** a reproducible, end-to-end proof of concept covering both exercises:
multi-node training (Exercise 1) and base-vs-tuned inference comparison (Exercise 2).

### Results at a glance

|                     |                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------- |
| **Model**           | Qwen3.6-35B-A3B (Mixture-of-Experts, 35B total / ~3B active)                                  |
| **Cluster**         | 2 nodes × 8 H200 on Nebius Soperator (Slurm on Kubernetes)                                    |
| **GPU utilization** | **85-92% sustained** across all 16 GPUs (target: >80%)                                        |
| **Training**        | full SFT, 3 epochs, completed in **51m33s**, no out-of-memory errors                          |
| **Tuned vs base**   | function-name accuracy **82→100%**, arguments **76→94%**, appropriate call/no-call **87→96%** |

## Summary

This is a multi-node LLM fine-tuning workflow on
[Nebius Soperator](https://github.com/nebius/soperator) (Slurm on Managed
Kubernetes), built to show a PoC team how to train and serve a
function-calling model on reserved GPU capacity. We fine-tuned
[**Qwen3.6-35B-A3B**](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B) (a 2026
open-weight Mixture-of-Experts (MoE) model: 256 experts, ~3B active of 35B, with hybrid
Gated DeltaNet attention; already instruction-tuned) for function calling.
Training used [**Megatron-SWIFT**](https://github.com/modelscope/ms-swift)
(ms-swift on a Megatron-Core backend) with **full supervised fine-tuning and
[Expert Parallelism](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html)**
across **2 nodes × 8 [H200](https://www.nvidia.com/en-us/data-center/h200/) GPUs**,
the Qwen-team-recommended path for MoE that keeps GPUs busy where naive MoE training
stalls at 10-20%. The data is the public
[**hypervariance/function-calling-sharegpt**](https://huggingface.co/datasets/hypervariance/function-calling-sharegpt)
corpus (~75k usable conversations) normalized through the tokenizer's chat template.
The run completed in 51 minutes at **85-92% sustained GPU utilization** (the
assignment's >80% target), verified on the Nebius dashboard. We then served the
base and fine-tuned models side-by-side with [**vLLM**](https://github.com/vllm-project/vllm)
and evaluated them on a 23-case function-calling suite, where the fine-tuned model
improved on every metric (detailed in _Inference and Evaluation_).

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

The cluster is provisioned with Terraform, forked from the official
[`nebius-solution-library`](https://github.com/nebius/soperator) Soperator recipe.
**Soperator** runs Slurm on top of Managed Kubernetes, so the team gets a familiar
`sbatch`/`squeue`/`sacct` workflow without needing Kubernetes expertise while Nebius
manages the control plane. The deployment is **2 worker nodes × 8 H200 (141 GB each)** on the
`eu-north2-a` InfiniBand fabric, with a single **shared filesystem mounted at
`/data`** on every node holding code, the model, datasets, the container image,
checkpoints, and logs. We keep everything on one shared filesystem rather than
per-node local disk: every rank reads the same model weights and dataset and writes to
one checkpoint path, and the container image is imported once and reused by both nodes
without copying. The 1 TB SSD allocation holds the model (~67 GB), the image (~18 GB),
and rolling checkpoints (~66 GB each, capped at two).

The container runtime is **[enroot](https://github.com/NVIDIA/enroot) via the Pyxis
SPANK plugin** (not Docker): jobs run with `srun --container-image=...`, and a single
`.sqsh` image on the shared filesystem is reused by both nodes for a reproducible,
drift-free environment. Before training we ran the built-in NCCL (NVIDIA Collective
Communications Library) multi-node all-reduce check and measured **~475 GB/s** bus
bandwidth (target >300 GB/s), confirming the inter-node interconnect.

## Model and Framework

**Model:** [Qwen3.6-35B-A3B](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B), a
2026 open-weight (Apache-2.0) Mixture-of-Experts model: 256 experts with top-8
routing (~3B active parameters of 35B total), 40 layers, and a **hybrid attention
stack** that interleaves full attention with **Gated DeltaNet** (linear-attention)
layers. It ships already instruction-tuned, so we do task-specialization rather than
base alignment.

We chose MoE over a dense model partly out of necessity: no dense open-weight model
above ~32B was released in 2026 (the field has moved to MoE), so a dense choice would
have meant a year-old model (the newest comparable option, Qwen2.5-72B, is from 2024).
MoE is also harder to train efficiently. Sparse top-k routing, expert load-balancing,
and all-to-all expert communication are what make naive MoE training stall at 10-20%
utilization, so keeping a sparse 35B MoE above 80% across two nodes is where most of
the work went. That is the capability a customer reserving large GPU capacity needs to
see on a current architecture rather than a legacy dense one.

**Framework:** [Megatron-SWIFT](https://github.com/modelscope/ms-swift) (ms-swift on
a Megatron-Core backend) with **full supervised fine-tuning and Expert Parallelism**.
The Qwen team recommends it for MoE: it is roughly 10x faster than HuggingFace and
DeepSpeed for MoE training and keeps GPU utilization high, where naive MoE training
(e.g. DeepSpeed ZeRO-3 + LoRA) is reported to stall at 10-20% and has documented
incompatibilities. We use **full SFT rather
than LoRA** because LoRA on MoE with Expert Parallelism at this scale is undocumented
and risks falling below the 80% utilization requirement; full SFT is the proven path
to >80%.

**Tool-call format.** Qwen3.6 emits XML-style tool calls
(`<function=name><parameter=key>value</parameter></function>`), so inference uses the
vLLM `qwen3_coder` parser (not Hermes JSON). The tokenizer's `apply_chat_template`
handles tool schemas and rendering; we never construct the XML by hand.

## Dataset

The primary dataset is the public
[**hypervariance/function-calling-sharegpt**](https://huggingface.co/datasets/hypervariance/function-calling-sharegpt)
corpus (~87k examples, no access gate). It is in ShareGPT form (a single
`conversations` column) with tool schemas embedded as **free text** in the system
message, function calls wrapped in `<functioncall>` tags, and arguments that are
sometimes double-stringified. A preparation script parses this inline: it extracts
the JSON tool schemas, converts `<functioncall>` tags into structured `tool_calls`,
handles the double-stringified arguments, maps roles, and drops unparseable examples
(~75k survive). Every example is then rendered through
`tokenizer.apply_chat_template(..., enable_thinking=False)` so the training signal
matches what the model sees at inference, and split 96/4 into train/eval
(a validation pass checks counts, JSON validity, and token statistics). With
**sequence packing** to 8192 tokens, the ~75k short conversations compact into
**4,558 train + 191 eval packed sequences** (~8,131 tokens each). `enable_thinking=False`
is kept throughout: it yields a simpler training signal, lower inference latency, and
cleaner output for an automation integration.

## Training

Training runs from the **official ms-swift enroot image** with `tilelang` baked in
(see _Engineering Challenges_), invoked as `megatron sft` from a two-script pattern:
an sbatch script sets `MASTER_ADDR`/`NNODES`, and an `srun` runner sets `NODE_RANK`
per node (the separation avoids quoting issues and rank collisions).

The parallelism is **pipeline=1, tensor=1, expert=8 (PP/TP/EP)**, which leaves a
data-parallel size of 16. Expert Parallelism shards the 256 experts across 8 ranks;
with no pipeline parallelism across nodes, both nodes run as full data-parallel
replicas and stay busy (the only inter-node traffic is an overlappable gradient
all-reduce plus the MoE all-to-all).
The main configuration:

| Setting      | Value                           | Rationale                                                     |
| ------------ | ------------------------------- | ------------------------------------------------------------- |
| Parallelism  | PP=1, TP=1, EP=8 (DP=16)        | experts sharded within a node; no cross-node pipeline bubbles |
| Batch        | micro 2, global 32              | enough compute per step to keep utilization >80%              |
| Sequence     | packing on, length 8192         | packs short conversations into full windows                   |
| Memory       | full activation recompute, bf16 | fits the model + optimizer in 143 GB                          |
| Attention    | flash                           | throughput                                                    |
| Optimization | lr 5e-6, cosine, 3 epochs       | conservative for an already instruction-tuned model           |
| Checkpoint   | `save_safetensors`              | written as a complete HuggingFace directory                   |

We enable the MoE fusions (`grouped_gemm`, `permute_fusion`, `shared_expert_overlap`)
for throughput. Because `save_safetensors` writes a standard HuggingFace checkpoint,
serving needs **no Megatron→HF export step**.

The verified run **completed in 51m33s**, 426 steps, loss **0.20 → 0.13**, eval loss
**0.196**, peak memory **113 / 143 GB per GPU** (no out-of-memory errors). The final
checkpoint
(66 GB of safetensors) is directly loadable by vLLM.

## GPU Utilization

The assignment's key metric is **DCGM_FI_DEV_GPU_UTIL** (from NVIDIA's Data Center GPU
Manager): the fraction of time the GPU executes kernels. We **sustained 85-92% across all 16 H200 GPUs** (mean ≈ 85%) during
steady-state training, above the >80% target. We measured utilization two
ways that agree: live from the cluster with an overlapping step on the job's own
allocation (`srun --jobid=N --overlap ... nvidia-smi`), and on the **Nebius console dashboard** (GPU-metrics view). A representative single sample of all 16 GPUs
(8 per node) during steady-state training illustrates the balance:

```
node 0:  84  80  86  91  86  82  92  84
node 1:  90  81  88  85  82  86  80  83        (mean 85%)
```

Reaching this took two changes from the initial design. First, switching from
pipeline parallelism (`PP=2`), which idled GPUs with cross-node pipeline bubbles
(alternating 0%/100%, averaging ~73% or worse), to **data-parallel `PP=1`** so both
nodes compute continuously. Second, raising **`micro_batch_size` from 1 to 2** so each
GPU does enough compute per optimizer step to amortize the all-reduce and MoE
all-to-all communication, which lifted utilization from 73% to 85%.

## Inference and Evaluation

Both models are served with [**vLLM**](https://github.com/vllm-project/vllm) 0.23.0
(from the same image) at tensor-parallel size 2: 35B in bf16 (~70 GB) fits on 2 H200s,
and TP=8 would only add communication overhead. A smoke test first confirmed that vLLM
loads the hybrid Gated DeltaNet architecture and emits a parsed tool call via
`--tool-call-parser qwen3_coder`. Both models run with thinking disabled
(`chat_template_kwargs={"enable_thinking": false}` per request) to match training, so
the two are compared under identical settings.

A comparison job stands up base (`:8000`) and tuned (`:8001`) on one node and runs a
harness that sends the same 23 hand-written prompts to each model at temperature 0
(so the comparison is deterministic and reproducible), across six
categories (single-tool, multi-argument, should-clarify, no-tool-needed,
process-automation, ambiguous). It parses the returned tool calls and scores them,
including _executability_ against mock tool implementations. The fine-tuned model
improves on every aggregate metric:

| Metric                              |  Base |      Tuned |     Δ |
| ----------------------------------- | ----: | ---------: | ----: |
| Appropriate call / no-call          | 87.0% |  **95.7%** |  +8.7 |
| Function-name accuracy (call cases) | 82.4% | **100.0%** | +17.6 |
| Argument exact-match (call cases)   | 76.5% |  **94.1%** | +17.6 |
| Executable against mock tools       | 65.2% |  **78.3%** | +13.1 |

Here _appropriate_ counts the model correct when it calls a tool exactly when one is
warranted and declines otherwise; _call cases_ are the prompts where a tool call was
expected; and _executable_ means the predicted call runs against a mock implementation
without error. Broken down by category (appropriate call/no-call rate), the gains concentrate where
the base model was weakest, with no regressions except _clarify_:

| Category           |   n | Base |    Tuned |
| ------------------ | --: | ---: | -------: |
| ambiguous          |   2 |   0% | **100%** |
| single_tool        |   6 |  83% | **100%** |
| multi_arg          |   5 | 100% |     100% |
| no_tool            |   3 | 100% |     100% |
| process_automation |   4 | 100% |     100% |
| clarify            |   3 | 100% |  **67%** |

The fine-tuned model now reliably translates ("good morning" → Japanese), resolves
ambiguous asks (`get_stock_price(TSLA)`; `set_reminder` instead of `get_weather`), and
fills multi-argument process-automation calls (tickets, refunds, VM provisioning)
correctly. The one regression is _clarify_ (100% → 67%): function-calling SFT makes the
model more eager to call a tool rather than ask for a missing argument. This is
expected; the fix is to add more negative and clarification examples to the data mix.

## Engineering Challenges

Several non-obvious issues each needed a real fix rather than a runtime patch:

- **ms-swift 4.x `[megatron]` is not on PyPI.** Every pip/uv install backtracked
  forever or pulled an incompatible torch (`No module named 'torch.multiprocessing'`).
  Fix: use the **official ms-swift Docker image** via enroot; it ships a consistent
  torch 2.11 / TransformerEngine / flash-attn / megatron-core / vLLM stack.
- **Qwen3.6's Gated DeltaNet crashes on the first backward on Hopper.** Its
  `fla` Triton kernel is numerically wrong on Triton ≥ 3.4 (the image has 3.6), so it
  hard-requires the **`tilelang`** backend. Fix: bake `tilelang==0.1.9` into a derived
  image, pinned to the image's `apache-tvm-ffi 0.1.9` (shared with vLLM) so nothing
  upgrades and FFI types don't double-register.
- **Pipeline parallelism idled GPUs**, and **batch divisibility**: with PP=1, the data
  parallel size is 16 (EP is orthogonal), so `global_batch` must be a multiple of
  `micro_batch × 16`, so `micro_batch=2` ⇒ `global_batch=32`.
- **vLLM 0.23.0 has no `--chat-template-kwargs` CLI flag** (servers exit at arg-parse);
  thinking is disabled **per request** in the body instead.

## Deliverables

The customer receives a self-contained repository:

- **`terraform/`**: the cluster definition (a forked Soperator recipe) to stand up an
  identical 2-node H200 environment.
- **`training/`**: the dataset pipeline (download, parse, validate) and the Slurm jobs
  for the one-time environment build and the multi-node training run.
- **`inference/`**: vLLM serving scripts for the base and tuned models, plus a small
  command-line client for ad-hoc tool-call queries.
- **`eval/`**: the 23-case test suite, mock tools, the comparison harness, and the
  saved results.
