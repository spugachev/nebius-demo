# Monitoring

How to watch the training run and prove the >80% GPU-utilization requirement.

## The metric

The assignment's target is **DCGM_FI_DEV_GPU_UTIL** — "percentage of time the GPU
spent executing kernels" (the standard nvidia-smi GPU-Util counter, not SM
occupancy or MFU). **Achieved: 85–92% sustained across all 16 GPUs** during
steady-state training (see screenshot for the demo).

## 1. Nebius console (DCGM dashboards) — the official source

Console → your project → Kubernetes → cluster `soperator-spugachev` → **Metrics**
tab → toggle **GPU metrics** (NOT "Basic metrics" — that only shows CPU/RAM/Disk).

- Panel **GPU Utilization** shows per-GPU util for every card on both worker nodes
  (node IDs `computeinstance-…`, 8 GPUs each = 16 lines).
- Set the time range to the training window for a clean plateau (a wide "Last 12h"
  range compresses a ~50-min run into spiky strokes and averages in idle time).
- Companion panels: Memory Utilization, Free Frame Buffer (drops as weights load).

## 2. Live from the cluster (both nodes, robust)

`dcgmi dmon` needs `nv-hostengine` running and only sees the node it runs on, so the
reliable way to read a **running** job's util across both nodes is an overlapping
step on the job's own allocation:

```bash
srun --jobid=<JOBID> --overlap --ntasks-per-node=1 --nodes=2 \
     nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

Convenience wrapper (prints a running mean every 10 s):

```bash
bash /data/code/training/watch_gpu.sh <JOBID>
# 09:18:42  mean=85%   per-gpu: 84 80 86 91 86 82 92 84 90 81 88 85 82 86 80 83
```

> Measure **after warmup** — the first iteration JIT-compiles TransformerEngine +
> tilelang kernels (~5 min); util is choppy and the step counter sits at 0/N until
> compilation finishes. This is normal.

## 3. Slurm job state

```bash
squeue                                   # queued / running jobs
squeue -h -j <JOBID> -o '%T %M'          # state + elapsed
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed   # final outcome
```

## 4. Training progress (loss / throughput)

ms-swift logs a JSON metrics line every `logging_steps`:

```bash
grep -E "'loss':|'eval_loss':" /data/logs/train-<JOBID>.out | tail
# {'loss': 0.143, 'grad_norm': 0.21, 'learning_rate': 5.5e-07,
#  'iteration': '400/426', 'memory(GiB)': 113.6, 'train_speed(s/it)': 4.6}
```

Also written to `output_dir/logging.jsonl` and TensorBoard under `output_dir/runs/`.

## What "healthy" looks like (run 65)

| Signal | Value |
|---|---|
| GPU util (16×H200) | 85–92% sustained |
| Memory per GPU | ~113 / 143 GB (no OOM) |
| Throughput | ~4.6 s/it after warmup |
| Loss | 0.20 → 0.13 (eval 0.196) |
| Wall clock | 51m33s, 3 epochs, 426 steps |
| NCCL all-reduce busbw (pre-flight) | ~475 GB/s (target >300) |
