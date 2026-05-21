#!/usr/bin/env python3
"""
Level 2 Benchmark: Harness Comparison

Compares different agent/tool frameworks on the same task: propose and apply
a single edit to train.py. Tests raw Ollama API, Aider, Claude Code, and
OpenHands with the same model and context.

No GPU needed for most tests — only checks whether edits are valid.

Usage:
    python benchmark/bench_harness.py \
        --adapters ollama_raw aider \
        --model qwen3.6:27b \
        --trials 20
"""

import argparse
import csv
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters.base import HarnessAdapter

# Registry of available adapters
ADAPTER_REGISTRY: dict[str, type[HarnessAdapter]] = {}


def _register_adapters():
    """Import and register all available adapters."""
    try:
        from benchmark.adapters.ollama_raw import OllamaRawAdapter
        ADAPTER_REGISTRY["ollama_raw"] = OllamaRawAdapter
    except ImportError:
        pass
    try:
        from benchmark.adapters.aider_adapter import AiderAdapter
        ADAPTER_REGISTRY["aider"] = AiderAdapter
    except ImportError:
        pass
    try:
        from benchmark.adapters.claude_code import ClaudeCodeAdapter
        ADAPTER_REGISTRY["claude_code"] = ClaudeCodeAdapter
    except ImportError:
        pass
    try:
        from benchmark.adapters.openhands_adapter import OpenHandsAdapter
        ADAPTER_REGISTRY["openhands"] = OpenHandsAdapter
    except ImportError:
        pass


def run_benchmark(adapter_names: list[str], model: str, trials: int,
                  ollama_url: str, fixture: str, results_fixture: str | None,
                  output: str | None) -> list[dict]:
    """Run harness comparison benchmark."""
    _register_adapters()

    fixture_path = Path(fixture)
    results_fixture_path = Path(results_fixture) if results_fixture else None
    all_results = []

    for adapter_name in adapter_names:
        if adapter_name not in ADAPTER_REGISTRY:
            print(f"WARNING: Adapter '{adapter_name}' not found. Available: "
                  f"{list(ADAPTER_REGISTRY.keys())}", file=sys.stderr)
            continue

        adapter = ADAPTER_REGISTRY[adapter_name](ollama_url=ollama_url)

        if not adapter.is_available():
            print(f"WARNING: Adapter '{adapter_name}' dependencies not available. Skipping.")
            continue

        print(f"\n=== Adapter: {adapter_name} (model: {model}) ===")
        adapter.setup()
        successes = 0

        for trial in range(1, trials + 1):
            with tempfile.TemporaryDirectory() as tmpdir:
                workdir = Path(tmpdir)
                shutil.copy(fixture_path, workdir / "train.py")
                if results_fixture_path:
                    shutil.copy(results_fixture_path, workdir / "results.tsv")

                result = adapter.propose_and_apply(
                    workdir, model,
                    results_tsv=workdir / "results.tsv" if results_fixture_path else None,
                )

            row = {
                "adapter": adapter_name,
                "model": model,
                "trial": trial,
                "success": result.success,
                "syntax_ok": result.syntax_ok,
                "is_meaningful": result.is_meaningful,
                "description": result.description[:80],
                "time_s": round(result.time_s, 1),
                "tokens_generated": result.tokens_generated,
                "error": (result.error or "")[:100],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            all_results.append(row)

            if result.success:
                successes += 1

            status = "OK" if result.success else "FAIL"
            print(f"  Trial {trial:3d}/{trials}: {status}  "
                  f"syntax={result.syntax_ok} meaningful={result.is_meaningful}  "
                  f"{result.time_s:.1f}s  {result.description[:40]}")

        adapter.cleanup()
        rate = successes / trials * 100
        print(f"  Summary: {successes}/{trials} ({rate:.0f}%) successful")

    # Write results
    if output and all_results:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {output_path}")

    return all_results


def print_summary(results: list[dict]):
    """Print a comparative summary table."""
    adapters = sorted(set(r["adapter"] for r in results))

    print("\n=== Harness Comparison Summary ===")
    print(f"{'Adapter':<15} {'Success':>8} {'Syntax':>8} {'Meaningful':>11} "
          f"{'Mean Time':>10} {'P50 Time':>9} {'P95 Time':>9}")
    print("-" * 78)

    for adapter in adapters:
        rows = [r for r in results if r["adapter"] == adapter]
        n = len(rows)
        success = sum(1 for r in rows if r["success"]) / n * 100
        syntax = sum(1 for r in rows if r["syntax_ok"]) / n * 100
        meaningful = sum(1 for r in rows if r["is_meaningful"]) / n * 100
        times = sorted(r["time_s"] for r in rows)
        mean_t = sum(times) / n
        p50 = times[n // 2]
        p95 = times[int(n * 0.95)]

        print(f"{adapter:<15} {success:>7.0f}% {syntax:>7.0f}% {meaningful:>10.0f}% "
              f"{mean_t:>9.1f}s {p50:>8.1f}s {p95:>8.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Level 2 Benchmark: Harness comparison"
    )
    _register_adapters()

    parser.add_argument("--adapters", nargs="+",
                        default=list(ADAPTER_REGISTRY.keys()),
                        help=f"Adapters to test (available: {list(ADAPTER_REGISTRY.keys())})")
    parser.add_argument("--model", default="qwen3.6:27b",
                        help="Ollama model name")
    parser.add_argument("--trials", type=int, default=20,
                        help="Trials per adapter (default: 20)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--fixture",
                        default=str(REPO_ROOT / "benchmark/fixtures/train_baseline.py"))
    parser.add_argument("--results-fixture",
                        default=str(REPO_ROOT / "benchmark/fixtures/results_5_experiments.tsv"))
    parser.add_argument("--output",
                        default=str(REPO_ROOT / "benchmark/results/harness_comparison.tsv"))

    args = parser.parse_args()
    results_fixture = None if args.results_fixture == "none" else args.results_fixture

    results = run_benchmark(
        adapter_names=args.adapters,
        model=args.model,
        trials=args.trials,
        ollama_url=args.ollama_url,
        fixture=args.fixture,
        results_fixture=results_fixture,
        output=args.output,
    )
    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
