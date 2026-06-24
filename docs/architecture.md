# Architecture

End-to-end function-calling fine-tuning on Nebius Soperator: 2 nodes × 8 H200,
Megatron-SWIFT (Expert Parallelism) for training, vLLM for inference.

## Topology

```
                         Nebius Soperator (Slurm on Managed Kubernetes)
                         tenant: csa-hiring-sandboxK · fabric eu-north2-a
   ┌───────────────────────────────────────────────────────────────────────┐
   │  login node (185.82.70.159)  ──sbatch/squeue──►  Slurm controller       │
   │                                                                         │
   │   worker-0 (8× H200, 141 GB)        worker-1 (8× H200, 141 GB)          │
   │   ┌─────────────────────────┐       ┌─────────────────────────┐        │
   │   │ enroot: swift431-tl.sqsh│◄─NCCL/IB ~475 GB/s──►│ same image│        │
   │   │ megatron sft  GPU 0..7  │       │ megatron sft  GPU 0..7  │        │
   │   └─────────────────────────┘       └─────────────────────────┘        │
   │                    ▲ both mount shared FS ▼                            │
   │   /data  (shared filesystem): code/ models/ datasets/ images/          │
   │                                checkpoints/ logs/                       │
   └───────────────────────────────────────────────────────────────────────┘
```

- **Scheduler:** Soperator gives a familiar Slurm interface on top of Managed
  Kubernetes — the most accessible distributed-training UX for an ML team without
  deep infra expertise. Provisioned with Terraform (forked Soperator recipe).
- **Container runtime:** enroot via the Pyxis SPANK plugin (`srun --container-image=...`),
  not Docker. One `.sqsh` on the shared FS is reused by both nodes.
- **Storage:** a single shared filesystem mounted at `/data` on every node holds
  code, the model, datasets, the container image, checkpoints, and logs.

## Training stack

```
ms-swift `megatron sft`  →  Megatron-Core 0.17.1  →  TransformerEngine 2.16
        (full SFT)              (+ mcore-bridge 1.5.0, tilelang 0.1.9)
                                         │
              PP=1 · TP=1 · EP=8  →  data_parallel_size = 16
```

- **Why Megatron-SWIFT + EP:** Qwen-team-recommended path for MoE; reaches >80%
  GPU util where naive MoE + DeepSpeed/TRL stalls at 10–20%.
- **Parallelism:** `pipeline=1, tensor=1, expert=8` ⇒ data-parallel size 16 (EP is
  orthogonal — it shards the 256 experts within the DP dimension). No pipeline
  parallelism across nodes ⇒ no cross-node pipeline bubbles ⇒ both nodes stay busy.
- **Environment:** the official ms-swift docker image (torch 2.11 + TE + flash-attn
  + vLLM + megatron-core, all consistent), with `tilelang` baked in for Qwen3.6's
  Gated DeltaNet layers on Hopper. No pip/uv at run time.

## Model

**Qwen3.6-35B-A3B** — a 2026 hybrid MoE: 256 experts (top-8, ~3B active of 35B),
40 layers, with **Gated DeltaNet** (linear-attention) layers interleaved with full
attention (`linear_attention_freq = [1,1,1,0]×10`). Already instruction-tuned;
we do full SFT for function calling. Tool-call format is XML (`<function=…>
<parameter=…>`), parsed by vLLM `qwen3_coder`.

## Inference stack

```
vLLM 0.23.0 (same image)
  base   : /data/models/Qwen3.6-35B-A3B…            TP=2  :8000
  tuned  : /data/checkpoints/…/checkpoint-426       TP=2  :8001
  --tool-call-parser qwen3_coder   ·  enable_thinking=false (per request)
```

`--save_safetensors=true` during training writes a complete HF checkpoint, so vLLM
loads the tuned model directly — **no Megatron→HF export step**.

## Data flow

```
hypervariance/function-calling-sharegpt (+ custom process-automation examples)
   → prepare_dataset.py  (parse, apply_chat_template, split)  → train/eval JSONL
   → megatron sft (3 epochs, packing 8192)                    → checkpoint-426 (HF)
   → vLLM base + tuned  ─►  eval/compare.py (23 prompts)       → comparison report
```

## Key decisions (one line each)

| Decision | Rationale |
|---|---|
| MoE (Qwen3.6) over dense | No dense >32B open-weight in 2026; shows production MoE skill |
| Full SFT over LoRA | Only documented path to >80% util on MoE+EP at this scale |
| Megatron-SWIFT over TRL/DeepSpeed | 10× MoE throughput; DeepSpeed ZeRO-3+LoRA+MoE is broken |
| PP=1/EP=8/DP=16 | PP across nodes idled GPUs; DP keeps both nodes busy |
| micro_batch=2/global_batch=32 | Lifted util 73%→85% (more compute per all-reduce) |
| Official image + tilelang | ms-swift 4.x not on PyPI; Qwen3.6 GDN needs tilelang on Hopper |
| vLLM TP=2 | 35B bf16 ≈ 70 GB fits on 2 H200; TP=8 adds needless comms |
