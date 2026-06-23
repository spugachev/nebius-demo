# CLAUDE.md

## Project

Nebius CSA hiring assignment: end-to-end multi-node LLM fine-tuning for function calling on Nebius Soperator, plus inference comparison. See ASSESMENT.md for requirements, PLAN.md for implementation design.

Deadline: June 26, 2026.

## Stack

- **Python**: managed with `uv` (not pip/poetry/conda). Use `uv add` to add dependencies, `uv run` to execute scripts, `uv sync` to install.
- **Training**: Megatron-SWIFT (ms-swift + Megatron-Core + mcore-bridge) with Expert Parallelism, full SFT
- **Model**: Qwen3.6-35B-A3B (MoE, April 2026, already instruction-tuned). Fallback: Qwen3-30B-A3B-Instruct-2507
- **Dataset**: hypervariance/function-calling-sharegpt (87k raw → ~75k after validation) + 1k custom process-automation examples
- **Infrastructure**: Nebius Soperator (Slurm on Kubernetes), 2 nodes x 8 H200 GPUs, eu-north2-a
- **Container runtime**: enroot via Pyxis SPANK plugin (not Docker)
- **Inference**: vLLM with `--tool-call-parser qwen3_coder`, TP=2 (not TP=8, 35B fits on 2 GPUs)
- **IaC**: Terraform (forked from nebius-solution-library/soperator/installations/example/)

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

On the Soperator cluster:
```bash
sbatch /opt/slurm-test/quickcheck/hello.sh                       # verify Slurm + GPUs
sbatch --nodes=2 /opt/slurm-test/quickcheck/nccl_multi_node.sh   # verify inter-node NCCL (target: >300 GB/s)
sbatch /shared/code/training/predownload.slurm                    # pre-download model weights
sbatch /shared/code/training/train.slurm                          # launch training
sbatch /shared/code/training/export_checkpoint.slurm              # convert checkpoint → HF format (2-node, EP must match training)
# Or if --save_safetensors true checkpoint is directly HF-loadable: skip export, point vLLM at checkpoint dir
bash /shared/code/inference/serve_base.sh                         # start vLLM base model
bash /shared/code/inference/serve_tuned.sh                        # start vLLM tuned model
python /shared/code/eval/compare.py                               # run comparison
```

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
- **Separate Slurm script + runner**: train.slurm submits run_megatron.sh via srun, avoiding shell quoting issues.
- **NODE_RANK uses SLURM_PROCID**: more portable than SLURM_NODEID when ntasks-per-node=1.
- **srun --export=ALL**: guarantees MASTER_ADDR/MASTER_PORT propagate from sbatch into srun tasks.
- **Checkpoint export is a 2-node Slurm job**: EP must match training (EP=8, PP=2 = 16 GPUs). Do not run as a single-node bash script.
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
training/          # dataset normalization, preparation, validation, slurm jobs, checkpoint export
inference/         # vLLM serving scripts
eval/              # test prompts, mock tools, comparison script (endpoints → metrics → report)
docs/              # architecture, monitoring, troubleshooting, demo script
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

## Gotchas

- Set `public_o11y_enabled = false` in terraform.tfvars (known Terraform recipe bug)
- Install `yq` before running terraform apply
- Do not use the same shared filesystem for 2 different Soperator jails
- Megatron checkpoint export parallelism sizes must match training config (EP=8, PP=2) — export needs both nodes
- ms-swift names checkpoints as `checkpoint-N` (e.g. checkpoint-500, checkpoint-1000), not `checkpoint-final`
- Megatron-Core version must be >= 0.16 and < 0.20
- transformers must be >= 5.5.0 for Qwen3.6 tokenizer support
- Do not destroy the lab environment after submission
- Join existing tenant `csa-hiring-sandboxK`, do not create a new one
- On fresh clusters, may need to run `ldconfig` in jail for NVIDIA libs (Soperator issue #2468)
- `MODELSCOPE_CACHE` and `HF_HOME` must point to shared filesystem so all nodes share model cache
- enroot image pull can be slow on first run; consider pre-pulling or using node_local_image_disk
- Pre-download model weights in a separate Slurm job before training (70GB download wastes GPU time)
- save_steps=500 + save_total_limit=2 to prevent storage overflow (each checkpoint ≈ 70GB)
- hypervariance dataset arguments are sometimes double-stringified — prepare_dataset.py must handle this
- Full SFT on instruction-tuned model risks catastrophic forgetting — keep non-tool examples in data mix
