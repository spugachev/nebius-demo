"""
Prepare training data from hypervariance/function-calling-sharegpt.

Downloads the dataset (cached by HuggingFace), parses ShareGPT format into
OpenAI-style messages, applies Qwen3.6 chat template, splits 96/4, and writes
train.jsonl + eval.jsonl ready for Megatron-SWIFT.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


# ── ShareGPT parsing ──────────────────────────────────────────────────────────

def _extract_tools(system_text: str) -> list[dict] | None:
    """Extract JSON tool schema(s) embedded as free text in the system message.

    hypervariance uses a single {…} object (not an array) per conversation.
    """
    # Try JSON array first (future-proof)
    for match in re.finditer(r'\[.*?\]', system_text, re.DOTALL):
        try:
            schemas = json.loads(match.group())
            if isinstance(schemas, list) and schemas and all(
                isinstance(s, dict) and 'name' in s for s in schemas
            ):
                return [{"type": "function", "function": s} for s in schemas]
        except (json.JSONDecodeError, TypeError):
            continue

    # Try single object — the common case in hypervariance
    # Use a simple brace-matching approach to find the outermost {...}
    start = system_text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(system_text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = system_text[start:i + 1]
                try:
                    schema = json.loads(candidate)
                    if isinstance(schema, dict) and 'name' in schema:
                        return [{"type": "function", "function": schema}]
                except json.JSONDecodeError:
                    pass
                break
    return None


def _parse_arguments(raw: str) -> dict | None:
    """Parse function arguments handling all hypervariance encoding variants:

    1. Direct dict embedded in JSON:     "arguments": {"key": "val"}
    2. Double-quoted JSON string:        "arguments": "{\"key\": \"val\"}"
    3. Single-quoted string (common):    "arguments": '{"key": "val"}'
    """
    # Already a dict — shouldn't happen at call site but guard anyway
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        # Try direct JSON parse
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Replace Python-style single-quoted strings with double-quoted
        # e.g. '{"key": "val"}' → {"key": "val"}
        unquoted = raw.strip()
        if unquoted.startswith("'") and unquoted.endswith("'"):
            unquoted = unquoted[1:-1]
            try:
                result = json.loads(unquoted)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    return None


def _parse_functioncall(text: str) -> dict | None:
    """Parse <functioncall> … </functioncall> into {name, arguments}."""
    match = re.search(r'<functioncall>\s*(.*?)\s*</functioncall>', text, re.DOTALL)
    if not match:
        return None

    call_text = match.group(1).strip()

    # Extract name — always a regular JSON string key
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', call_text)
    if not name_match:
        return None
    name = name_match.group(1)

    # Extract arguments — three formats (see _parse_arguments)
    args = None

    # Case 3: single-quoted string value (most common in this dataset)
    # "arguments": '{"country": "US"}'
    sq_match = re.search(r'"arguments"\s*:\s*\'(.*?)\'(?=\s*[,}])', call_text, re.DOTALL)
    if sq_match:
        args = _parse_arguments(sq_match.group(1))

    if args is None:
        # Cases 1 & 2: try to parse the whole functioncall as JSON
        try:
            parsed = json.loads(call_text)
            raw_args = parsed.get('arguments')
            args = _parse_arguments(raw_args) if raw_args is not None else {}
        except json.JSONDecodeError:
            pass

    if not isinstance(args, dict):
        return None

    return {"name": name, "arguments": args}


def _parse_conversation(conversations: list[dict]) -> dict | None:
    """Convert one hypervariance conversation to OpenAI-style messages + tools.

    Returns None if the example is malformed.
    """
    messages = []
    tools = None
    call_counter = 0

    for turn in conversations:
        role = turn.get('from', '')
        value = turn.get('value', '')

        if role == 'system':
            tools = _extract_tools(value)
            # Discard hypervariance system boilerplate — it's always generic
            # ("You are a helpful assistant with access to the following functions…").
            # apply_chat_template(tools=tools) will inject the proper Qwen3.6 system prompt.

        elif role == 'human':
            messages.append({"role": "user", "content": value})

        elif role == 'gpt':
            if '<functioncall>' in value:
                call = _parse_functioncall(value)
                if call is None:
                    return None  # Malformed call — skip example
                call_id = f"call_{call_counter:03d}"
                call_counter += 1
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }],
                })
            else:
                messages.append({"role": "assistant", "content": value})

        elif role == 'function_response':
            # Must follow an assistant turn with tool_calls
            if not messages or "tool_calls" not in messages[-1]:
                return None
            last_call = messages[-1]["tool_calls"][-1]
            messages.append({
                "role": "tool",
                "tool_call_id": last_call["id"],
                "name": last_call["function"]["name"],
                "content": value,
            })

    if not any(m["role"] == "user" for m in messages):
        return None

    return {"messages": messages, "tools": tools}


# ── Chat template application ─────────────────────────────────────────────────

def _apply_template(example: dict, tokenizer) -> dict | None:
    """Apply Qwen3.6 chat template and return {"text": rendered_string}.

    Using {"text": ...} avoids fragile re-parsing of the rendered output back
    into message dicts. ms-swift accepts this format directly.
    """
    try:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tools=example.get("tools"),
            enable_thinking=False,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return None

    if not text:
        return None

    return {"text": text}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Megatron-SWIFT training data")
    parser.add_argument("--output-dir", default="data", help="Output directory for JSONL files")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="Tokenizer model ID")
    parser.add_argument("--eval-ratio", type=float, default=0.04, help="Fraction held out for eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None, help="Cap for quick testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading hypervariance/function-calling-sharegpt ...")
    dataset = load_dataset("hypervariance/function-calling-sharegpt", split="train")
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    print(f"  {len(dataset):,} raw examples")

    examples = []
    n_skipped = 0

    for row in tqdm(dataset, desc="Parsing + templating"):
        parsed = _parse_conversation(row["conversations"])
        if parsed is None:
            n_skipped += 1
            continue

        rendered = _apply_template(parsed, tokenizer)
        if rendered is None:
            n_skipped += 1
            continue

        examples.append(rendered)

    print(f"  {len(examples):,} valid  |  {n_skipped:,} skipped ({n_skipped / len(dataset) * 100:.1f}%)")

    if not examples:
        print("ERROR: no valid examples produced", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    n_eval = max(1, int(len(examples) * args.eval_ratio))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    print(f"  train: {len(train_examples):,}  |  eval: {len(eval_examples):,}")

    def write_jsonl(path: Path, data: list[dict]):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    write_jsonl(output_dir / "train.jsonl", train_examples)
    write_jsonl(output_dir / "eval.jsonl", eval_examples)


if __name__ == "__main__":
    main()
