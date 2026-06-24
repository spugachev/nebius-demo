#!/usr/bin/env python3
"""Compare base vs fine-tuned Qwen3.6 on function-calling.

Sends each prompt in prompts.jsonl to two OpenAI-compatible vLLM endpoints
(base + tuned), parses the tool calls, scores them, and writes a report.

Stdlib only (urllib) so it runs anywhere — on the login node or inside the
container — without extra deps. Model names are auto-detected via /v1/models.

Usage:
    python compare.py \
        --base-url  http://localhost:8000 \
        --tuned-url http://localhost:8001 \
        --prompts   /data/code/eval/prompts.jsonl \
        --out       /data/logs/comparison
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

try:
    from mock_tools import MOCK_TOOLS
except Exception:  # allow running from any cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mock_tools import MOCK_TOOLS


# ── HTTP helpers ─────────────────────────────────────────────────────────────
def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def detect_model(base_url: str) -> str:
    return _get(f"{base_url}/v1/models")["data"][0]["id"]


def wait_ready(base_url: str, minutes: float = 20.0) -> None:
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=5)
            return
        except Exception:
            time.sleep(10)
    raise TimeoutError(f"{base_url} not ready after {minutes} min")


def call_model(base_url: str, model: str, query: str, tools: list) -> dict:
    """Return {'tool_calls': [{name, arguments(dict|None)}], 'content': str, 'raw': msg}."""
    resp = _post(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
        },
    )
    msg = resp["choices"][0]["message"]
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = None  # unparseable arguments
        calls.append({"name": fn.get("name"), "arguments": args, "args_raw": args_raw})
    return {"tool_calls": calls, "content": msg.get("content"), "raw": msg}


# ── Scoring ──────────────────────────────────────────────────────────────────
def _norm(v) -> str:
    return str(v).strip().lower()


def args_match(expected: dict, predicted: dict) -> bool:
    """Expected args must all appear in predicted (subset match), value-normalized.
    Numbers compared numerically; strings case-insensitively."""
    if not expected:
        return True
    if not isinstance(predicted, dict):
        return False
    for k, ev in expected.items():
        if k not in predicted:
            return False
        pv = predicted[k]
        if isinstance(ev, (int, float)) and not isinstance(ev, bool):
            try:
                if float(pv) != float(ev):
                    return False
            except Exception:
                return False
        else:
            if _norm(pv) != _norm(ev):
                return False
    return True


def score_one(expected: dict, result: dict, tools: list) -> dict:
    calls = result["tool_calls"]
    called = len(calls) > 0
    exp_name = expected.get("name")  # None means "should NOT call / should clarify"
    first = calls[0] if called else None

    valid = called and first is not None and isinstance(first.get("arguments"), dict)
    name_ok = bool(called and exp_name is not None and first and first["name"] == exp_name)
    args_ok = bool(name_ok and args_match(expected.get("arguments", {}), first.get("arguments", {})))

    if exp_name is None:
        appropriate = not called            # should have declined / asked to clarify
    else:
        appropriate = name_ok               # should have called the right function

    # executable: function exists in mocks and runs with the predicted args
    executable = False
    if valid and first and first["name"] in MOCK_TOOLS:
        try:
            MOCK_TOOLS[first["name"]](**first["arguments"])
            executable = True
        except Exception:
            executable = False

    return {
        "called": called,
        "valid_tool_call": valid if exp_name is not None else (not called),
        "name_correct": name_ok,
        "args_match": args_ok,
        "appropriate": appropriate,
        "executable": executable,
        "predicted": ({"name": first["name"], "arguments": first.get("arguments")} if first else None),
    }


METRICS = ["appropriate", "name_correct", "args_match", "executable"]


def aggregate(rows: list, key: str) -> dict:
    out = {}
    n = len(rows)
    for m in METRICS:
        out[m] = round(100.0 * sum(1 for r in rows if r[key][m]) / n, 1) if n else 0.0
    # name/args accuracy only meaningful on "should call" cases
    call_rows = [r for r in rows if r["expected"].get("name") is not None]
    nc = len(call_rows)
    out["name_acc_on_calls"] = round(100.0 * sum(1 for r in call_rows if r[key]["name_correct"]) / nc, 1) if nc else 0.0
    out["args_acc_on_calls"] = round(100.0 * sum(1 for r in call_rows if r[key]["args_match"]) / nc, 1) if nc else 0.0
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--tuned-url", default="http://localhost:8001")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "prompts.jsonl"))
    ap.add_argument("--out", default="comparison")
    ap.add_argument("--wait", type=float, default=20.0, help="minutes to wait for servers")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(prompts)} prompts")

    print("Waiting for both servers...")
    wait_ready(args.base_url, args.wait)
    wait_ready(args.tuned_url, args.wait)
    base_model = detect_model(args.base_url)
    tuned_model = detect_model(args.tuned_url)
    print(f"base  = {base_model} @ {args.base_url}")
    print(f"tuned = {tuned_model} @ {args.tuned_url}")

    rows = []
    for p in prompts:
        base_res = call_model(args.base_url, base_model, p["query"], p["tools"])
        tuned_res = call_model(args.tuned_url, tuned_model, p["query"], p["tools"])
        row = {
            "id": p["id"],
            "category": p["category"],
            "query": p["query"],
            "expected": p["expected"],
            "base": score_one(p["expected"], base_res, p["tools"]),
            "tuned": score_one(p["expected"], tuned_res, p["tools"]),
            "base_content": base_res["content"],
            "tuned_content": tuned_res["content"],
        }
        rows.append(row)
        b, t = row["base"], row["tuned"]
        flag = "↑" if (t["appropriate"] and not b["appropriate"]) else (" " if t["appropriate"] == b["appropriate"] else "↓")
        print(f"  [{flag}] {p['id']:<14} {p['category']:<18} base.appropriate={b['appropriate']!s:<5} tuned.appropriate={t['appropriate']}")

    base_agg = aggregate(rows, "base")
    tuned_agg = aggregate(rows, "tuned")

    # ── report ──
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(
        {"base_model": base_model, "tuned_model": tuned_model, "n": len(rows),
         "base": base_agg, "tuned": tuned_agg, "rows": rows}, indent=2))

    lines = [
        "# Base vs Fine-tuned — Function Calling Comparison\n",
        f"- Prompts: **{len(rows)}**  |  base: `{base_model}`  |  tuned: `{tuned_model}`\n",
        "\n## Metrics (%)\n",
        "| Metric | Base | Tuned | Δ |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "appropriate": "Appropriate call/no-call",
        "name_acc_on_calls": "Function-name accuracy (call cases)",
        "args_acc_on_calls": "Argument exact-match (call cases)",
        "executable": "Executable against mock tools",
    }
    for k, lab in labels.items():
        b, t = base_agg[k], tuned_agg[k]
        lines.append(f"| {lab} | {b} | {t} | {round(t - b, 1):+} |")
    lines.append("\n## Per-category appropriate-rate\n")
    cats = sorted({r["category"] for r in rows})
    lines += ["| Category | n | Base | Tuned |", "|---|---:|---:|---:|"]
    for c in cats:
        cr = [r for r in rows if r["category"] == c]
        b = round(100 * sum(1 for r in cr if r["base"]["appropriate"]) / len(cr))
        t = round(100 * sum(1 for r in cr if r["tuned"]["appropriate"]) / len(cr))
        lines.append(f"| {c} | {len(cr)} | {b}% | {t}% |")
    # a few illustrative diffs
    lines.append("\n## Examples where tuned improved\n")
    for r in rows:
        if r["tuned"]["appropriate"] and not r["base"]["appropriate"]:
            lines.append(f"- **{r['id']}** ({r['category']}): _{r['query']}_")
            lines.append(f"  - base → `{r['base']['predicted']}`")
            lines.append(f"  - tuned → `{r['tuned']['predicted']}` (expected `{r['expected'].get('name')}`)")
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")

    print("\n=== SUMMARY ===")
    for k, lab in labels.items():
        print(f"  {lab:<38} base {base_agg[k]:>5}%   tuned {tuned_agg[k]:>5}%   Δ {tuned_agg[k]-base_agg[k]:+.1f}")
    print(f"\nReport: {args.out}.md  |  raw: {args.out}.json")


if __name__ == "__main__":
    main()
