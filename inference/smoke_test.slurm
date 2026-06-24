#!/bin/bash
#SBATCH --job-name=vllm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=main
#SBATCH --output=/data/logs/vllm-smoke-%j.out
#SBATCH --time=00:40:00

set -e

# Smoke test: can vLLM (in the ms-swift image) load Qwen3.6 (hybrid MoE +
# Gated DeltaNet) and emit a tool call with the qwen3_coder parser? Serve the BASE
# model on 2 GPUs, send one tool-call request, print the result, shut down.

MODEL="/data/models/Qwen3.6-35B-A3B/Qwen/Qwen3.6-35B-A3B"
IMAGE="/data/images/swift431-tl.sqsh"   # has vLLM 0.23.0 + tilelang (GDN may need it at inference too)
PORT=8000

srun --container-image="$IMAGE" --container-mounts="/data:/data" bash -c '
set -e
export VLLM_USE_MODELSCOPE=false
echo "=== vLLM version ==="; python -c "import vllm; print(vllm.__version__)"

echo "=== starting vLLM server (TP=2) ==="
vllm serve "'"$MODEL"'" \
    --served-model-name qwen36-base \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --max-model-len 8192 \
    --host 0.0.0.0 --port '"$PORT"' > /data/logs/vllm-smoke-server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT

echo "=== waiting for /health (up to ~15 min) ==="
for i in $(seq 1 90); do
    if curl -sf http://localhost:'"$PORT"'/health >/dev/null 2>&1; then echo "server ready after ${i}0s"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED, tail of log:"; tail -40 /data/logs/vllm-smoke-server.log; exit 1; fi
    sleep 10
done

echo "=== tool-call request ==="
curl -s http://localhost:'"$PORT"'/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"qwen36-base\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Paris right now?\"}],
    \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get current weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}],
    \"tool_choice\": \"auto\",
    \"temperature\": 0
  }" | python -c "import sys, json; d=json.load(sys.stdin); m=d[\"choices\"][0][\"message\"]; print(\"--- content:\", m.get(\"content\")); print(\"--- tool_calls:\", json.dumps(m.get(\"tool_calls\"), indent=2))"

echo "=== smoke test done ==="
'
