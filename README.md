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

### Query both models by hand (interactive)

The login node has no GPUs — drop into the container on a GPU worker, start both
servers, then `curl` them.

**1. Shell into the container on a GPU node (4 GPUs: 2 per model):**
```bash
srun --nodes=1 --gpus-per-node=4 --cpus-per-task=48 --mem=400G \
     --partition=main --time=02:00:00 \
     --container-image=/data/images/swift431-tl.sqsh \
     --container-mounts=/data:/data --pty bash
```

**2. Start both models in the background:**
```bash
BASE=/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B
TUNED=/data/checkpoints/qwen3-fc-20260624-0025/v0-20260624-082630/checkpoint-426
OPTS="--tensor-parallel-size 2 --tool-call-parser qwen3_coder --enable-auto-tool-choice --max-model-len 8192 --host 0.0.0.0"

CUDA_VISIBLE_DEVICES=0,1 vllm serve $BASE  --served-model-name base  $OPTS --port 8000 > /data/logs/base.log  2>&1 &
CUDA_VISIBLE_DEVICES=2,3 vllm serve $TUNED --served-model-name tuned $OPTS --port 8001 > /data/logs/tuned.log 2>&1 &

# loading is ~11 min each (MoE + GDN + kernel compile); wait for both:
until curl -sf localhost:8000/health && curl -sf localhost:8001/health; do sleep 10; done; echo READY
```

**3. Write a tiny helper** (Python handles JSON/quoting cleanly — far more
paste-safe than a multi-line bash function; the quoted `'PY'` prevents any shell
expansion). Disables thinking per request, like training:
```bash
cat > /data/ask.py <<'PY'
import sys, json, urllib.request
port, model, q = sys.argv[1], sys.argv[2], sys.argv[3]
tools = [
  {"type":"function","function":{"name":"get_weather","description":"Get current weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
  {"type":"function","function":{"name":"create_ticket","description":"Create a support ticket",
    "parameters":{"type":"object","properties":{"customer":{"type":"string"},"category":{"type":"string"},
    "priority":{"type":"string"}},"required":["customer","category"]}}},
]
payload = {"model": model, "messages":[{"role":"user","content":q}], "tools":tools,
           "tool_choice":"auto", "temperature":0, "chat_template_kwargs":{"enable_thinking":False}}
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
      data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
m = json.load(urllib.request.urlopen(req))["choices"][0]["message"]
print("content:   ", m.get("content"))
print("tool_calls:", json.dumps(m.get("tool_calls"), ensure_ascii=False, indent=2))
PY
```

**4. Ask both models:**
```bash
Q="Open a high priority support ticket for Acme Corp about a payment_failure"
echo "── BASE ──";  python3 /data/ask.py 8000 base  "$Q"
echo "── TUNED ──"; python3 /data/ask.py 8001 tuned "$Q"
```

Edit `$Q` for your own cases and the `tools` list in `/data/ask.py` for your own
functions. For one model only, use `--gpus-per-node=2` and start a single
`vllm serve … --port 8000`. `exit` frees the GPUs (servers stop with the allocation).

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
