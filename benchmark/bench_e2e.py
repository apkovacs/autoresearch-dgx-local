#!/usr/bin/env python3
"""
Level 3 Benchmark: End-to-End Evaluation

Full experiment loop on GPU: launches the hypothesis generator or full agent,
runs N experiments, and measures final val_bpb, experiment count, and failure rate.

Requires GPU (DGX Spark or equivalent).

Usage:
    python benchmark/bench_e2e.py \
        --harness hyp \
        --model qwen3.6:27b \
        --budget 20
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HARNESS_SCRIPTS = {
    "hyp": "run-dgx-local.sh",
    "agent": "run-dgx-agent.sh",
}


def parse_results_tsv(path: Path) -> list[dict]:
    """Parse results.tsv into a list of dicts."""
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().strip().split('\n')
    if len(lines) < 2:
        return []
    header = lines[0].split('\t')
    for line in lines[1:]:
        fields = line.split('\t')
        if len(fields) >= len(header):
            rows.append(dict(zip(header, fields)))
    return rows


def run_e2e(harness: str, model: str, budget: int,
            docker_image: str | None, output: str | None) -> dict:
    """Run end-to-end benchmark with the specified harness."""
    script = HARNESS_SCRIPTS.get(harness)
    if not script:
        print(f"ERROR: Unknown harness '{harness}'. Available: {list(HARNESS_SCRIPTS.keys())}",
              file=sys.stderr)
        sys.exit(1)

    script_path = REPO_ROOT / script
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    # Build command
    env_vars = {
        "OLLAMA_MODEL": model,
    }
    if docker_image:
        env_vars["DOCKER_IMAGE"] = docker_image

    cmd = ["bash", str(script_path)]
    if harness == "hyp":
        cmd.extend(["--max-experiments", str(budget)])
    elif harness == "agent":
        cmd.extend(["--max-restarts", "0"])  # single session
        cmd.extend(["--experiments-per-session", str(budget)])

    import os
    full_env = {**os.environ, **env_vars}

    print(f"=== End-to-End Benchmark ===")
    print(f"  Harness:     {harness} ({script})")
    print(f"  Model:       {model}")
    print(f"  Budget:      {budget} experiments")
    print(f"  Docker:      {docker_image or 'default'}")
    print()

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=full_env,
        timeout=budget * 600,  # 10 min per experiment max
    )
    elapsed = time.time() - t0

    # Parse results
    results_path = REPO_ROOT / "results.tsv"
    rows = parse_results_tsv(results_path)

    total = len(rows)
    kept = sum(1 for r in rows if r.get("status") == "keep")
    discarded = sum(1 for r in rows if r.get("status") == "discard")
    crashed = sum(1 for r in rows if r.get("status") == "crash")

    # Find best and baseline val_bpb
    best_bpb = None
    baseline_bpb = None
    for r in rows:
        try:
            bpb = float(r.get("val_bpb", "0"))
            if bpb > 0:
                if best_bpb is None or bpb < best_bpb:
                    best_bpb = bpb
                if "baseline" in r.get("description", "").lower():
                    baseline_bpb = bpb
        except ValueError:
            pass

    result = {
        "harness": harness,
        "model": model,
        "budget": budget,
        "total_experiments": total,
        "kept": kept,
        "discarded": discarded,
        "crashed": crashed,
        "success_rate": round(kept / total * 100, 1) if total > 0 else 0,
        "baseline_bpb": baseline_bpb or 0,
        "best_bpb": best_bpb or 0,
        "improvement": round((baseline_bpb or 0) - (best_bpb or 0), 6),
        "wall_time_s": round(elapsed, 0),
        "exit_code": proc.returncode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n=== Results ===")
    print(f"  Experiments:  {total}/{budget}")
    print(f"  Kept:         {kept}")
    print(f"  Discarded:    {discarded}")
    print(f"  Crashed:      {crashed}")
    print(f"  Baseline:     {baseline_bpb}")
    print(f"  Best:         {best_bpb}")
    print(f"  Improvement:  {result['improvement']}")
    print(f"  Wall time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists()
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys(), delimiter='\t')
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        print(f"\n  Appended to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Level 3 Benchmark: End-to-end evaluation (requires GPU)"
    )
    parser.add_argument("--harness", choices=list(HARNESS_SCRIPTS.keys()),
                        default="hyp",
                        help="Harness to benchmark (default: hyp)")
    parser.add_argument("--model", default="qwen3.6:27b",
                        help="Ollama model name")
    parser.add_argument("--budget", type=int, default=20,
                        help="Number of experiments to run (default: 20)")
    parser.add_argument("--docker-image", default=None,
                        help="Docker image override")
    parser.add_argument("--output",
                        default=str(REPO_ROOT / "benchmark/results/e2e.tsv"),
                        help="Output TSV path (appends)")

    args = parser.parse_args()
    run_e2e(
        harness=args.harness,
        model=args.model,
        budget=args.budget,
        docker_image=args.docker_image,
        output=args.output,
    )


if __name__ == "__main__":
    main()
