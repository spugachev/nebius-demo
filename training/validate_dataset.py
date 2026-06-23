"""
Validate prepared train.jsonl and eval.jsonl before uploading to the cluster.

Checks structure, token length distribution, tool call ratio, and prints
sample examples for manual review.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def validate_file(path: Path, tokenizer, n_samples: int = 3) -> bool:
    if not path.exists():
        print(f"MISSING: {path}")
        return False

    examples = []
    broken = 0

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                continue
            if not isinstance(obj.get("text"), str) or not obj["text"]:
                broken += 1
                continue
            examples.append(obj["text"])

    print(f"=== {path} ===")
    print(f"Examples:    {len(examples):,}")

    if broken:
        print(f"BROKEN JSON: {broken}  ← fix before uploading")

    if not examples:
        print("ERROR: no valid examples")
        return False

    # Token lengths
    lengths = [len(tokenizer.encode(t)) for t in examples]
    arr = np.array(lengths)
    over_limit = int((arr > 8192).sum())

    print(f"\nToken lengths:")
    print(f"  min={arr.min()}  p50={int(np.percentile(arr,50))}  "
          f"p90={int(np.percentile(arr,90))}  p99={int(np.percentile(arr,99))}  max={arr.max()}")
    print(f"  avg={arr.mean():.0f}  total={arr.sum()/1e6:.1f}M tokens")
    if over_limit:
        print(f"  over 8192: {over_limit} ({over_limit/len(arr)*100:.1f}%)  ← will be truncated")
    else:
        print(f"  over 8192: 0  ✓")

    # Tool call ratio
    with_tools = sum(1 for t in examples if "<tool_call>" in t or "<function=" in t)
    no_tools = len(examples) - with_tools
    print(f"\nContent:")
    print(f"  with tool calls: {with_tools:,} ({with_tools/len(examples)*100:.1f}%)")
    print(f"  no tool calls:   {no_tools:,} ({no_tools/len(examples)*100:.1f}%)")

    # Random samples
    print(f"\n--- {n_samples} random samples ---")
    rng = random.Random(0)
    for i, text in enumerate(rng.sample(examples, min(n_samples, len(examples))), 1):
        print(f"\n[Sample {i}]")
        print(text[:600])
        if len(text) > 600:
            print(f"  ... ({len(text)} chars total)")

    return broken == 0


def main():
    parser = argparse.ArgumentParser(description="Validate prepared dataset files")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ok = True
    for name in ("train.jsonl", "eval.jsonl"):
        print()
        ok &= validate_file(data_dir / name, tokenizer, args.samples)

    print()
    if ok:
        print("✓ Dataset looks good — ready to upload to cluster")
    else:
        print("✗ Fix errors above before uploading")
        sys.exit(1)


if __name__ == "__main__":
    main()
