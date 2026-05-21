#!/usr/bin/env bash
# run-bench.sh — Run benchmarks inside the containerized benchmark suite
#
# Builds the benchmark Docker image (if needed) and runs the specified
# benchmark level with all adapter dependencies pre-installed.
#
# Usage:
#   bash benchmark/run-bench.sh edit-quality --models qwen3.6:27b gemma4:26b --trials 10
#   bash benchmark/run-bench.sh harness --adapters ollama_raw aider --model qwen3.6:27b
#   bash benchmark/run-bench.sh e2e --harness hyp --model qwen3.6:27b --budget 10
#   bash benchmark/run-bench.sh shell    # drop into a shell inside the container
#
# Prerequisites:
#   - Docker installed
#   - Ollama running on the host (localhost:11434)
#   - Models pulled: ollama pull qwen3.6:27b
#
# Environment variables:
#   BENCH_IMAGE    Docker image name (default: autoresearch-bench)
#   OLLAMA_URL     Ollama API URL (default: http://localhost:11434)
#   BENCH_GPU      Set to "1" to pass --gpus all (needed for Level 3)

set -euo pipefail

BENCH_IMAGE="${BENCH_IMAGE:-autoresearch-bench}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
BENCH_GPU="${BENCH_GPU:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Build image if needed ---
if ! docker image inspect "$BENCH_IMAGE" &>/dev/null; then
    echo "Building benchmark image: $BENCH_IMAGE"
    docker build -t "$BENCH_IMAGE" "$SCRIPT_DIR"
    echo ""
fi

# --- Parse benchmark level ---
if [ $# -lt 1 ]; then
    echo "Usage: bash benchmark/run-bench.sh <level> [options]"
    echo ""
    echo "Levels:"
    echo "  edit-quality   Level 1: Edit quality (no GPU)"
    echo "  harness        Level 2: Harness comparison (mostly no GPU)"
    echo "  e2e            Level 3: End-to-end (requires GPU)"
    echo "  shell          Interactive shell inside the benchmark container"
    echo ""
    echo "Examples:"
    echo "  bash benchmark/run-bench.sh edit-quality --models qwen3.6:27b --trials 5"
    echo "  bash benchmark/run-bench.sh harness --adapters ollama_raw aider --trials 10"
    echo "  BENCH_GPU=1 bash benchmark/run-bench.sh e2e --budget 10"
    exit 0
fi

LEVEL="$1"
shift

case "$LEVEL" in
    edit-quality|edit_quality|l1)
        CMD="python benchmark/bench_edit_quality.py $*"
        ;;
    harness|l2)
        CMD="python benchmark/bench_harness.py $*"
        ;;
    e2e|l3)
        BENCH_GPU=1
        CMD="python benchmark/bench_e2e.py $*"
        ;;
    shell)
        CMD="bash"
        ;;
    *)
        echo "Unknown level: $LEVEL"
        echo "Use: edit-quality, harness, e2e, or shell"
        exit 1
        ;;
esac

# --- Build docker run command ---
DOCKER_ARGS=(
    docker run --rm -it
    --network host
    -v "$REPO_ROOT":/workspace
    -v /var/run/docker.sock:/var/run/docker.sock
    -e "OLLAMA_URL=$OLLAMA_URL"
)

if [ "$BENCH_GPU" = "1" ]; then
    DOCKER_ARGS+=(--gpus all)
fi

echo "=== Autoresearch Benchmark ==="
echo "  Level:   $LEVEL"
echo "  Image:   $BENCH_IMAGE"
echo "  Ollama:  $OLLAMA_URL"
echo "  GPU:     $([ "$BENCH_GPU" = "1" ] && echo "enabled" || echo "disabled")"
echo "  Command: $CMD"
echo ""

"${DOCKER_ARGS[@]}" "$BENCH_IMAGE" $CMD
