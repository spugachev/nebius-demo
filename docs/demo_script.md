# Demo-day script (~5 min)

Audience: a <20-person, VC-funded startup building AI-agent process-automation,
evaluating Nebius before reserving 512 H100s for 6 months. PoC team is ML
engineers with limited infra/cloud experience.

**One-liner:** "We fine-tuned a 2026 35B MoE model for function calling across 16
H200s on Soperator at **85–92% GPU utilization**, and the tuned model is measurably
better at picking the right tool with the right arguments — all with a familiar
Slurm workflow your team can reuse and scale to 512 GPUs."

---

### 1. The customer problem (30s)
Your team wants to validate Nebius with a real, end-to-end fine-tuning workflow —
function calling for your automation agents — without becoming infra experts first.

### 2. Why Soperator + Megatron-SWIFT (1 min)
- **Soperator** = Slurm on Managed Kubernetes → the distributed-training interface
  your ML engineers already know (`sbatch`, `squeue`, `sacct`), no K8s expertise needed.
- **Megatron-SWIFT with Expert Parallelism** is the recommended way to train MoE
  models efficiently — it keeps GPUs busy (>80%) where naive MoE training stalls at
  10–20%. We use a **2026 MoE model (Qwen3.6-35B-A3B)**, not a 2024 dense one.

### 3. Architecture (1 min) — show `docs/architecture.md` diagram
- 2 nodes × 8 H200, InfiniBand `eu-north2-a` (**~475 GB/s** NCCL all-reduce).
- One enroot container image shared by both nodes; one shared filesystem for
  code/data/checkpoints. Reproducible, no per-node drift.
- Parallelism: PP=1 · EP=8 ⇒ data-parallel across both nodes (16-way).

### 4. Training walkthrough (1 min) — show the GPU dashboard
- Pipeline: `prepare_dataset.py` → `sbatch train.slurm` → monitor.
- **Show the Nebius GPU-Utilization dashboard: all 16 GPUs at 80–92%, mean 85%.**
- Run facts: full SFT, 3 epochs, **51m33s**, loss 0.20 → 0.13, no OOM (113/143 GB).
- Reproducible & honest: we hit real issues (Qwen3.6's Gated DeltaNet needed the
  `tilelang` backend; PP-vs-DP and batch sizing for util) and documented every fix
  in `docs/troubleshooting.md`.

### 5. Results — base vs tuned (1 min) — show `eval/results/comparison.md`

| Metric | Base | Tuned | Δ |
|---|---:|---:|---:|
| Appropriate call / no-call | 87.0% | **95.7%** | +8.7 |
| Function-name accuracy | 82.4% | **100%** | +17.6 |
| Argument exact-match | 76.5% | **94.1%** | +17.6 |
| Executable against mock tools | 65.2% | **78.3%** | +13.1 |

Concrete wins: the tuned model now reliably calls `translate_text`, resolves
ambiguous asks (`get_stock_price(TSLA)`, `set_reminder` vs `get_weather`), and gets
multi-argument process-automation calls right (tickets, refunds, VM provisioning).
Honest trade-off: it became more eager to call tools (clarify-rate dipped) — fixable
with more negative/clarify examples in the mix.

### 6. What you can reuse + scale (30s)
- Take home: Terraform for the cluster, Slurm scripts, the dataset pipeline, the
  monitoring guide, and the inference+eval harness.
- **Scaling path:** the same Megatron-SWIFT + EP approach goes to 512 H100s by adding
  nodes (more data-parallel replicas) and/or larger MoE models — no workflow change.

---

### Q&A cheat-sheet
- **Why 85% and not 99%?** MoE has unavoidable all-to-all/all-reduce comms; 85%
  sustained on a real 35B MoE is strong. Lever for more: micro_batch=4 / global_batch=64.
- **Why no LoRA?** LoRA+EP at this scale is undocumented and risks <80% util; full SFT
  is the proven >80% path. We mitigate forgetting with non-tool examples in the data.
- **Checkpoint → serving?** `--save_safetensors` makes the checkpoint directly
  vLLM-loadable; no export step.
- **Can it serve our format?** Yes — vLLM `--tool-call-parser qwen3_coder`, OpenAI-
  compatible API; verified with a tool-call smoke test before the full eval.
