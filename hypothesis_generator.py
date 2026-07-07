"""
Hypothesis generator for autoresearch experiment loop.

Builds a prompt from train.py + results history, calls an inference
backend for a structured edit proposal, and applies edits to train.py.
Designed to be called from a deterministic bash loop (run-dgx-local.sh)
— no agent framework needed.

Backends:
    ollama  — Ollama's native /api/chat with format=json (default)
    openai  — any OpenAI-compatible /chat/completions endpoint
              (llama-server, vLLM, ds4, ...) with response_format
              json_object. This is the integration point for engines
              with capabilities Ollama lacks, e.g. speculative decoding
              (llama-server --draft-model, DSpark).

Usage:
    python3 hypothesis_generator.py propose --model qwen3.6:27b
    python3 hypothesis_generator.py propose --backend openai \
        --url http://localhost:8080/v1 --model deepseek-v4-flash-dwarf
    python3 hypothesis_generator.py apply --edits-json '{"description":"...","edits":[...]}'

Environment variables (CLI flags take precedence):
    INFERENCE_BACKEND   ollama | openai
    INFERENCE_URL       backend base URL
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

BACKEND_DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:8080/v1",
}

# ---------------------------------------------------------------------------
# System prompt — distilled from program.md
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a machine learning researcher. Your task is to propose a single edit \
to a PyTorch training script (train.py) to minimize validation loss (val_bpb).

## Constraints
- Fixed 5-minute training budget (wall clock). You cannot change this.
- Single GPU (NVIDIA GB10 Blackwell, 128 GB unified memory).
- You can only edit train.py. The evaluation harness (prepare.py) is read-only.
- VRAM is a soft constraint — moderate increases are OK for meaningful val_bpb gains.

## What you can change
- Hyperparameters: learning rates, batch sizes, depth, aspect ratio, warmup/warmdown schedules
- Model architecture: attention patterns, layer structure, activations, normalization
- Optimizer: algorithm, weight decay, betas, parameter grouping
- Anything else in train.py that might improve val_bpb

## Strategy
- If no experiments have been run yet, suggest running the baseline unchanged (empty edits).
- If recent edits improved val_bpb, continue exploring that direction.
- If recent edits failed or didn't improve, try a different approach.
- Prefer small, targeted changes over large rewrites.
- Simpler is better: removing complexity for equal performance is a win.

## Response format
Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
  "description": "short description of the change",
  "edits": [
    {"old_string": "exact text to find in train.py", "new_string": "replacement text"}
  ]
}

The "old_string" must be an EXACT substring of train.py (including whitespace and comments).
The "new_string" is what replaces it. Keep edits minimal — only change what's necessary.
For the baseline (first run), return: {"description": "baseline", "edits": []}

## Examples

Example 1 — change a hyperparameter:
{"description": "increase DEPTH from 4 to 6", "edits": [{"old_string": "DEPTH = 4               # number of transformer layers (reduced from 8)", "new_string": "DEPTH = 6               # number of transformer layers"}]}

Example 2 — change learning rate:
{"description": "double matrix learning rate", "edits": [{"old_string": "MATRIX_LR = 0.04", "new_string": "MATRIX_LR = 0.08"}]}

Example 3 — baseline (no changes):
{"description": "baseline", "edits": []}
"""

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

HYPERPARAM_SECTION_RE = re.compile(
    r'(# -{10,}\n# Hyperparameters.*?\n# -{10,}\n)(.*?)(# -{10,})',
    re.DOTALL
)


def build_user_prompt(train_py_path: str, results_path: str, max_results: int = 10) -> str:
    """Build the user message from train.py content and results history."""
    train_content = Path(train_py_path).read_text()

    # Highlight the hyperparameters section
    match = HYPERPARAM_SECTION_RE.search(train_content)
    if match:
        start, params, end = match.group(1), match.group(2), match.group(3)
        highlighted = f"{start}# >>> EDITABLE HYPERPARAMETERS (primary target for changes) <<<\n{params}{end}"
        train_content = train_content[:match.start()] + highlighted + train_content[match.end():]

    parts = [
        "## Current train.py\n```python\n" + train_content + "\n```\n"
    ]

    # Add results history if available
    results_file = Path(results_path)
    if results_file.exists():
        lines = results_file.read_text().strip().split('\n')
        if len(lines) > 1:  # has data beyond header
            header = lines[0]
            data_lines = lines[1:]
            # Take last N results
            recent = data_lines[-max_results:] if len(data_lines) > max_results else data_lines
            parts.append("## Experiment history (most recent last)\n```\n")
            parts.append(header + "\n")
            parts.append("\n".join(recent))
            parts.append("\n```\n")

            # Extract best val_bpb for context
            best_bpb = None
            for line in data_lines:
                fields = line.split('\t')
                if len(fields) >= 4 and fields[3] == 'keep':
                    try:
                        bpb = float(fields[1])
                        if bpb > 0 and (best_bpb is None or bpb < best_bpb):
                            best_bpb = bpb
                    except ValueError:
                        pass
            if best_bpb is not None:
                parts.append(f"\n**Current best val_bpb: {best_bpb}** — propose an edit to beat this.\n")
        else:
            parts.append("\nNo experiments run yet. Propose the baseline (empty edits).\n")
    else:
        parts.append("\nNo experiments run yet. Propose the baseline (empty edits).\n")

    parts.append("\nRespond with ONLY a JSON object. No markdown fences, no explanation.")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def call_ollama(model: str, system_prompt: str, user_prompt: str,
                ollama_url: str = "http://localhost:11434",
                num_predict: int = 4096, temperature: float = 0.7) -> dict:
    """Call Ollama chat API and return the parsed response."""
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    eval_count = data.get("eval_count", 0)

    return {
        "content": content,
        "elapsed_s": elapsed,
        "tokens_generated": eval_count,
    }


def call_openai_compat(model: str, system_prompt: str, user_prompt: str,
                       base_url: str = "http://localhost:8080/v1",
                       num_predict: int = 4096, temperature: float = 0.7) -> dict:
    """Call an OpenAI-compatible /chat/completions endpoint.

    Works with llama-server, vLLM, and other OpenAI-compatible servers.
    Uses response_format json_object to constrain output to valid JSON
    (the equivalent of Ollama's format=json).
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": num_predict,
        "temperature": temperature,
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI-compat API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compat API returned no choices: {json.dumps(data)[:200]}")
    content = (choices[0].get("message") or {}).get("content", "")
    usage = data.get("usage") or {}

    return {
        "content": content,
        "elapsed_s": elapsed,
        "tokens_generated": usage.get("completion_tokens", 0) or 0,
    }


def call_backend(backend: str, model: str, system_prompt: str, user_prompt: str,
                 url: str, num_predict: int, temperature: float) -> dict:
    """Dispatch to the selected inference backend."""
    if backend == "ollama":
        return call_ollama(model, system_prompt, user_prompt,
                           ollama_url=url, num_predict=num_predict,
                           temperature=temperature)
    if backend == "openai":
        return call_openai_compat(model, system_prompt, user_prompt,
                                  base_url=url, num_predict=num_predict,
                                  temperature=temperature)
    raise ValueError(f"Unknown backend: {backend} (use 'ollama' or 'openai')")


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

def validate_response(content: str) -> dict:
    """Parse and validate the model's JSON response.

    Returns the validated dict or raises ValueError with a description.
    """
    # Strip markdown fences if present (some models wrap JSON in ```json...```)
    content = content.strip()
    if content.startswith("```"):
        lines = content.split('\n')
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    if "description" not in data:
        raise ValueError("Missing 'description' field")
    if not isinstance(data["description"], str):
        raise ValueError(f"'description' must be string, got {type(data['description']).__name__}")

    if "edits" not in data:
        raise ValueError("Missing 'edits' field")
    if not isinstance(data["edits"], list):
        raise ValueError(f"'edits' must be array, got {type(data['edits']).__name__}")

    for i, edit in enumerate(data["edits"]):
        if not isinstance(edit, dict):
            raise ValueError(f"edits[{i}] must be object, got {type(edit).__name__}")
        if "old_string" not in edit:
            raise ValueError(f"edits[{i}] missing 'old_string'")
        if "new_string" not in edit:
            raise ValueError(f"edits[{i}] missing 'new_string'")
        if not isinstance(edit["old_string"], str):
            raise ValueError(f"edits[{i}].old_string must be string")
        if not isinstance(edit["new_string"], str):
            raise ValueError(f"edits[{i}].new_string must be string")

    return data


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------

def apply_edits(train_py_path: str, edits: list) -> list:
    """Apply edits to train.py. Returns list of errors (empty = success)."""
    content = Path(train_py_path).read_text()
    errors = []

    for i, edit in enumerate(edits):
        old = edit["old_string"]
        new = edit["new_string"]

        if old == new:
            errors.append(f"edits[{i}]: old_string == new_string (no-op)")
            continue

        if old in content:
            content = content.replace(old, new, 1)
        else:
            # Try with trailing whitespace stripped from both
            old_stripped = "\n".join(line.rstrip() for line in old.split("\n"))
            content_stripped = "\n".join(line.rstrip() for line in content.split("\n"))
            if old_stripped in content_stripped:
                # Find the actual position in the original content
                # and replace using the stripped version
                content = content_stripped.replace(old_stripped, new, 1)
            else:
                errors.append(
                    f"edits[{i}]: old_string not found in train.py: "
                    f"{old[:80]!r}..."
                )

    if not errors:
        Path(train_py_path).write_text(content)

    return errors


# ---------------------------------------------------------------------------
# CLI: propose
# ---------------------------------------------------------------------------

def cmd_propose(args):
    """Build prompt, call the inference backend, validate, print JSON to stdout."""
    user_prompt = build_user_prompt(args.train_py, args.results, args.max_results)

    # Resolve backend + URL: CLI flag > env var > default.
    # --ollama-url is kept as a back-compat alias for --url with the
    # ollama backend (run-dgx-local.sh and older scripts pass it).
    backend = args.backend or os.environ.get("INFERENCE_BACKEND", "ollama")
    url = args.url or os.environ.get("INFERENCE_URL")
    if not url and backend == "ollama":
        url = args.ollama_url
    if not url:
        url = BACKEND_DEFAULT_URLS.get(backend)

    max_retries = args.retries
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            result = call_backend(
                backend=backend,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                url=url,
                num_predict=args.num_predict,
                temperature=args.temperature,
            )
            validated = validate_response(result["content"])
            # Add metadata
            validated["_meta"] = {
                "model": args.model,
                "backend": backend,
                "elapsed_s": round(result["elapsed_s"], 1),
                "tokens_generated": result["tokens_generated"],
                "attempt": attempt,
            }
            print(json.dumps(validated))
            return 0

        except (ValueError, RuntimeError, requests.RequestException) as e:
            last_error = str(e)
            print(f"  Attempt {attempt}/{max_retries} failed: {last_error}",
                  file=sys.stderr)

    # All retries exhausted
    print(json.dumps({"error": last_error}))
    return 1


# ---------------------------------------------------------------------------
# CLI: apply
# ---------------------------------------------------------------------------

def cmd_apply(args):
    """Apply edits from JSON to train.py."""
    try:
        data = json.loads(args.edits_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
        return 1

    edits = data.get("edits", [])
    if not edits:
        print("No edits to apply (baseline run).")
        return 0

    errors = apply_edits(args.train_py, edits)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"Applied {len(edits)} edit(s) to {args.train_py}")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hypothesis generator for autoresearch experiment loop"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # propose
    p_propose = sub.add_parser("propose", help="Propose an edit to train.py")
    p_propose.add_argument("--model", default="qwen3.6:27b",
                           help="Model name (default: qwen3.6:27b)")
    p_propose.add_argument("--backend", choices=["ollama", "openai"], default=None,
                           help="Inference backend (default: $INFERENCE_BACKEND or ollama). "
                                "'openai' works with any OpenAI-compatible server: "
                                "llama-server, vLLM, ds4")
    p_propose.add_argument("--url", default=None,
                           help="Backend base URL (default: $INFERENCE_URL, or "
                                "http://localhost:11434 for ollama / "
                                "http://localhost:8080/v1 for openai)")
    p_propose.add_argument("--ollama-url", default="http://localhost:11434",
                           help="Ollama API URL (back-compat alias for --url)")
    p_propose.add_argument("--train-py", default="train.py",
                           help="Path to train.py (default: train.py)")
    p_propose.add_argument("--results", default="results.tsv",
                           help="Path to results.tsv (default: results.tsv)")
    p_propose.add_argument("--max-results", type=int, default=10,
                           help="Max results history rows to include (default: 10)")
    p_propose.add_argument("--num-predict", type=int, default=4096,
                           help="Max output tokens (default: 4096)")
    p_propose.add_argument("--temperature", type=float, default=0.7,
                           help="Sampling temperature (default: 0.7)")
    p_propose.add_argument("--retries", type=int, default=3,
                           help="Max retries on invalid response (default: 3)")

    # apply
    p_apply = sub.add_parser("apply", help="Apply edits to train.py")
    p_apply.add_argument("--edits-json", required=True,
                         help="JSON string with description and edits array")
    p_apply.add_argument("--train-py", default="train.py",
                         help="Path to train.py (default: train.py)")

    args = parser.parse_args()

    if args.command == "propose":
        sys.exit(cmd_propose(args))
    elif args.command == "apply":
        sys.exit(cmd_apply(args))


if __name__ == "__main__":
    main()
