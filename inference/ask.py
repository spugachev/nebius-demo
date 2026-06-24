#!/usr/bin/env python3
"""Query one running vLLM endpoint with a tool-call request, from inside the
GPU-worker container. Usage:

    python3 /data/ask.py <port> <model> "<question>"
    # e.g.
    python3 /data/ask.py 8000 base  "What's the weather in Paris?"
    python3 /data/ask.py 8001 tuned "Open a high priority ticket for Acme about a payment_failure"

Disables Qwen3.6 thinking per request (chat_template_kwargs), matching training.
Edit `tools` below to expose your own functions.
"""
import sys, json, urllib.request

if len(sys.argv) != 4:
    sys.exit('usage: python3 ask.py <port> <model:base|tuned> "<question>"')

port, model, q = sys.argv[1], sys.argv[2], sys.argv[3]

tools = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "create_ticket", "description": "Create a support ticket",
        "parameters": {"type": "object",
                       "properties": {"customer": {"type": "string"},
                                      "category": {"type": "string"},
                                      "priority": {"type": "string"}},
                       "required": ["customer", "category"]}}},
]

payload = {
    "model": model,
    "messages": [{"role": "user", "content": q}],
    "tools": tools,
    "tool_choice": "auto",
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}

req = urllib.request.Request(
    f"http://localhost:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
m = json.load(urllib.request.urlopen(req))["choices"][0]["message"]
print("content:   ", m.get("content"))
print("tool_calls:", json.dumps(m.get("tool_calls"), ensure_ascii=False, indent=2))
