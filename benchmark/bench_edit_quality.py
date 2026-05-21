#!/usr/bin/env python3
"""
Level 1 Benchmark: Edit Quality

Tests whether a model can produce valid, applicable, syntactically correct
edits to train.py via the raw Ollama API (hypothesis generator pattern).
No GPU needed — only checks edit quality, not training outcomes.

Usage:
    python benchmark/bench_edit_quality.py \
        --models qwen3.6:27b gemma4:26b \
        --trials 10 \
        --ollama-url http://localhost:11434
"""

import argparse
import csv
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters.ollama_raw import OllamaRawAdapter


def run_benchmark(models: list[str], trials: int, ollama_url: str,
                  fixture: str, results_fixture: str,
                  output: str | None) -> list[dict]:
    """Run edit quality benchmark across models and trials."""
    adapter = OllamaRawAdapter(ollama_url=ollama_url)

    if not adapter.is_available():
        print(f"ERROR: Ollama not available at {ollama_url}", file=sys.stderr)
        sys.exit(1)

    fixture_path = Path(fixture)
    results_fixture_path = Path(results_fixture) if results_fixture else None

    all_results = []

    for model in models:
        print(f"\n=== Model: {model} ===")
        successes = 0

        for trial in range(1, trials + 1):
            # Copy fixture to temp dir for isolation
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
                "model": model,
                "trial": trial,
                "success": result.success,
                "json_valid": result.json_valid,
                "schema_valid": result.schema_valid,
                "edits_apply": result.edits_apply,
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
                  f"json={result.json_valid} schema={result.schema_valid} "
                  f"apply={result.edits_apply} syntax={result.syntax_ok} "
                  f"meaningful={result.is_meaningful}  "
                  f"{result.time_s:.1f}s  "
                  f"{result.description[:40]}")

        rate = successes / trials * 100
        print(f"  Summary: {successes}/{trials} ({rate:.0f}%) successful edits")

    # Write results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {output_path}")

    return all_results


def print_summary(results: list[dict]):
    """Print a summary table across all models."""
    models = sorted(set(r["model"] for r in results))
    print("\n=== Summary ===")
    print(f"{'Model':<25} {'Success':>8} {'JSON':>6} {'Schema':>8} "
          f"{'Apply':>7} {'Syntax':>8} {'Mean Time':>10}")
    print("-" * 80)

    for model in models:
        model_results = [r for r in results if r["model"] == model]
        n = len(model_results)
        success = sum(1 for r in model_results if r["success"]) / n * 100
        json_v = sum(1 for r in model_results if r["json_valid"]) / n * 100
        schema = sum(1 for r in model_results if r["schema_valid"]) / n * 100
        apply_ = sum(1 for r in model_results if r["edits_apply"]) / n * 100
        syntax = sum(1 for r in model_results if r["syntax_ok"]) / n * 100
        mean_t = sum(r["time_s"] for r in model_results) / n

        print(f"{model:<25} {success:>7.0f}% {json_v:>5.0f}% {schema:>7.0f}% "
              f"{apply_:>6.0f}% {syntax:>7.0f}% {mean_t:>9.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Level 1 Benchmark: Edit quality (no GPU needed)"
    )
    parser.add_argument("--models", nargs="+", default=["qwen3.6:27b"],
                        help="Ollama model names to test")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials per model (default: 10)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--fixture", default=str(REPO_ROOT / "benchmark/fixtures/train_baseline.py"),
                        help="Path to frozen train.py fixture")
    parser.add_argument("--results-fixture",
                        default=str(REPO_ROOT / "benchmark/fixtures/results_5_experiments.tsv"),
                        help="Path to results history fixture (or 'none')")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmark/results/edit_quality.tsv"),
                        help="Output TSV path")

    args = parser.parse_args()
    results_fixture = None if args.results_fixture == "none" else args.results_fixture

    results = run_benchmark(
        models=args.models,
        trials=args.trials,
        ollama_url=args.ollama_url,
        fixture=args.fixture,
        results_fixture=results_fixture,
        output=args.output,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
