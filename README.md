# Nebius Soperator — Multi-Node LLM Fine-Tuning for Function Calling

End-to-end example for a Nebius PoC: fine-tune a 2026 open-weight MoE LLM
(**Qwen3.6-35B-A3B**) for function calling on **Nebius Soperator** (Slurm on
Kubernetes) across **2 nodes × 8 H200 GPUs**, then serve and compare base vs
tuned with vLLM.

> **Status: both exercises complete.**
> - **Exercise 1 (training):** full SFT ran in **51m33s** on 16×H200 at **85–92%
>   sustained GPU utilization** (requirement: >80%), loss 0.20→0.13, eval 0.196.
> - **Exercise 2 (inference):** tuned beats base on every metric — function-name
>   accuracy 82→**100%**, argument exact-match 76→**94%**, appropriate call/no-call
>   87→**96%** (`eval/results/comparison.md`).
> - **Docs:** architecture / monitoring / troubleshooting / demo script in `docs/`.
>
> See `CLAUDE.md` for the authoritative as-built configuration and `PLAN.md` for
> the design rationale.

## What this demonstrates

- Multi-node MoE training with **Megatron-SWIFT** (Expert Parallelism), the
  Qwen-team-recommended path that hits >80% GPU util on MoE (DeepSpeed/TRL do not).
- Reproducible environment via **enroot** containers (the official ms-swift image,
  no fragile pip installs).
- >80% GPU utilization on the Nebius DCGM dashboards — the assignment's key metric.
- A realistic function-calling dataset pipeline and a base-vs-tuned evaluation.

## Stack

| Layer | Choice |
|---|---|
| Scheduler | Nebius Soperator (Slurm on K8s), 2×8 H200, fabric `eu-north2-a` |
| Training | Megatron-SWIFT (`megatron sft`), full SFT, PP=1 · EP=8 · DP=16 |
| Model | Qwen3.6-35B-A3B (hybrid MoE: 256 experts + Gated DeltaNet layers) |
| Container | enroot/Pyxis; official ms-swift image `swift4.3.1` + `tilelang 0.1.9` |
| Inference | vLLM, `--tool-call-parser qwen3_coder`, TP=2 |
| IaC | Terraform (forked Soperator recipe) |
| Local tooling | `uv` (dataset prep only) |

## Repository layout

```
terraform/   Soperator cluster (forked nebius-solution-library recipe)
training/    dataset prep (local) + Slurm jobs (predownload, image build, train)
inference/   vLLM serving scripts (serve_base.sh, serve_tuned.sh, smoke_test.slurm)
eval/        prompts.jsonl, mock_tools.py, compare.py, run_comparison.slurm, results/
docs/        architecture.md, monitoring.md, troubleshooting.md, demo_script.md
```

See `CLAUDE.md` → "File layout" for a per-file description.

## Reproduce

### 0. Prerequisites (local)
- `uv`, `terraform`, `yq`, the Nebius CLI, and an SSH key at `./ssh/nebius_ed25519`.

### 1. Provision the cluster
```bash
cd terraform && terraform init && terraform apply -var-file=terraform.tfvars
# tenant: csa-hiring-sandboxK (existing, shared). public_o11y_enabled=false (recipe bug).
bash terraform/login.sh   # SSH to the login node
```

### 2. Prepare the dataset (local)
```bash
uv sync
uv run python training/prepare_dataset.py    # → data/train.jsonl + data/eval.jsonl
uv run python training/validate_dataset.py
scp -i ssh/nebius_ed25519 data/train.jsonl data/eval.jsonl root@<login>:/data/datasets/
scp -i ssh/nebius_ed25519 -r training/ root@<login>:/data/code/
```

### 3. One-time cluster setup, then train
```bash
# on the login node
sbatch --nodes=2 /opt/slurm-test/quickcheck/nccl_multi_node.sh   # ~475 GB/s busbw (>300 target)
sbatch /data/code/training/predownload.slurm        # ModelScope → /data/models (~67GB)
sbatch /data/code/training/import_image.slurm       # official ms-swift image → /data/images/swift431.sqsh
sbatch /data/code/training/build_image_tl.slurm     # + tilelang → swift431-tl.sqsh
sbatch /data/code/training/train.slurm              # 2-node full SFT
```

### 4. Watch GPU utilization (the deliverable metric)
```bash
bash /data/code/training/watch_gpu.sh <JOBID>   # live, both nodes, via srun --overlap nvidia-smi
# Also visible on the Nebius console DCGM dashboards (DCGM_FI_DEV_GPU_UTIL).
```

### 5. Serve & compare (Exercise 2)
`--save_safetensors=true` makes each `checkpoint-N` directly vLLM-loadable, so
**no export step is needed**. One job stands up both models and scores them:
```bash
sbatch /data/code/eval/run_comparison.slurm   # base :8000 + tuned :8001, then compare.py
# → /data/logs/comparison-<JOB>.md  (saved in repo: eval/results/comparison.md)
```
Both serve with `--tool-call-parser qwen3_coder`; the client sends
`chat_template_kwargs={"enable_thinking": false}` for an apples-to-apples comparison.
For interactive/demo serving use `inference/serve_base.sh` / `serve_tuned.sh`.

## Key engineering notes

The hard-won, non-obvious facts (full detail in `CLAUDE.md`):

1. **Environment must be the official ms-swift docker image, not pip/uv** — ms-swift
   4.x with the `[megatron]` extra is not on PyPI; pip/uv either backtrack forever
   or pull an incompatible torch.
2. **Qwen3.6 has Gated DeltaNet layers that need `tilelang` on Hopper** — pin
   `tilelang==0.1.9` to match the image's `apache-tvm-ffi 0.1.9` (shared with vLLM);
   bake it into a derived image, no runtime patches.
3. **Parallelism: PP=1 / EP=8 → DP=16** (EP is orthogonal to DP). PP=2 across nodes
   idled GPUs with pipeline bubbles.
4. **micro_batch_size=2 / global_batch_size=32 → 85% util** (micro_batch=1 gave 73%);
   `global_batch` must be divisible by `micro_batch × 16`.
5. **No checkpoint export** — `--save_safetensors=true` writes a complete HF checkpoint.
