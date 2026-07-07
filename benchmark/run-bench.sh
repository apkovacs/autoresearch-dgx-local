#!/usr/bin/env bash
# run-bench.sh — Run benchmarks inside the containerized benchmark suite
#
# Builds the benchmark Docker image (if needed), starts Ollama inside the
# container, pulls the requested model(s), and runs the benchmark.
#
# Usage:
#   bash benchmark/run-bench.sh edit-quality --models qwen3.6:27b gemma4:26b --trials 10
#   bash benchmark/run-bench.sh harness --adapters ollama_raw aider --model qwen3.6:27b
#   bash benchmark/run-bench.sh e2e --harness hyp --model qwen3.6:27b --budget 10
#   bash benchmark/run-bench.sh dashboard    # generate results dashboard
#   bash benchmark/run-bench.sh shell        # interactive shell inside the container
#
# Prerequisites:
#   - Docker installed
#   - Model weights are cached on the host (~/.ollama/models) and mounted in
#
# Environment variables:
#   BENCH_IMAGE     Docker image name (default: autoresearch-bench)
#   OLLAMA_MODELS   Host path for Ollama model cache (default: ~/.ollama/models)
#   OLLAMA_MODEL    Default model for benchmarks (default: qwen3.6:27b)
#   BENCH_GPU       Set to "1" to pass --gpus all (needed for Level 3)

set -euo pipefail

BENCH_IMAGE="${BENCH_IMAGE:-autoresearch-bench}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"
OLLAMA_GGUF="${OLLAMA_GGUF:-}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-32768}"
BENCH_GPU="${BENCH_GPU:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Create persistent directories ---
mkdir -p "$OLLAMA_MODELS"

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
    echo "  trace          Level 4: Trace quality — agentic overhead from transcripts (no GPU)"
    echo "  dashboard      Generate HTML dashboard from results"
    echo "  shell          Interactive shell inside the benchmark container"
    echo ""
    echo "Examples:"
    echo "  bash benchmark/run-bench.sh edit-quality --models qwen3.6:27b --trials 5"
    echo "  bash benchmark/run-bench.sh harness --adapters ollama_raw aider --trials 10"
    echo "  BENCH_GPU=1 bash benchmark/run-bench.sh e2e --budget 10"
    echo "  bash benchmark/run-bench.sh dashboard"
    echo ""
    echo "Environment variables:"
    echo "  OLLAMA_MODELS   Host Ollama model cache (default: ~/.ollama/models)"
    echo "  OLLAMA_MODEL    Default model (default: qwen3.6:27b)"
    echo "  BENCH_GPU       Enable GPU (default: 0)"
    exit 0
fi

LEVEL="$1"
shift

case "$LEVEL" in
    edit-quality|edit_quality|l1)
        CMD=(python benchmark/bench_edit_quality.py "$@")
        ;;
    harness|l2)
        CMD=(python benchmark/bench_harness.py "$@")
        ;;
    e2e|l3)
        BENCH_GPU=1
        CMD=(python benchmark/bench_e2e.py "$@")
        ;;
    trace|l4)
        CMD=(python benchmark/bench_trace_quality.py "$@")
        ;;
    dashboard)
        CMD=(python benchmark/bench_dashboard.py "$@")
        ;;
    shell)
        CMD=(bash)
        ;;
    *)
        echo "Unknown level: $LEVEL"
        echo "Use: edit-quality, harness, e2e, trace, dashboard, or shell"
        exit 1
        ;;
esac

# --- Build docker run command ---
DOCKER_ARGS=(
    docker run --rm -it
    -v "$REPO_ROOT":/workspace
    -v "$OLLAMA_MODELS":/root/.ollama/models
    -v /var/run/docker.sock:/var/run/docker.sock
    --add-host=host.docker.internal:host-gateway
    -e "OLLAMA_MODEL=$OLLAMA_MODEL"
    -e "OLLAMA_KEEP_ALIVE=0"
    -e "INFERENCE_BACKEND=${INFERENCE_BACKEND:-ollama}"
    -e "INFERENCE_URL=${INFERENCE_URL:-}"
)

if [ "$BENCH_GPU" = "1" ]; then
    DOCKER_ARGS+=(--gpus all)
fi

# Custom GGUF import (community quants not in the Ollama library)
if [ -n "$OLLAMA_GGUF" ]; then
    if [ ! -f "$OLLAMA_GGUF" ]; then
        echo "ERROR: OLLAMA_GGUF file not found: $OLLAMA_GGUF"
        exit 1
    fi
    DOCKER_ARGS+=(
        -v "$OLLAMA_GGUF":/gguf/model.gguf:ro
        -e OLLAMA_GGUF_FILE=/gguf/model.gguf
        -e OLLAMA_NUM_CTX="$OLLAMA_NUM_CTX"
    )
fi

echo "=== Autoresearch Benchmark ==="
echo "  Level:        $LEVEL"
echo "  Image:        $BENCH_IMAGE"
echo "  Ollama model: $OLLAMA_MODEL"
echo "  Model cache:  $OLLAMA_MODELS"
echo "  GPU:          $([ "$BENCH_GPU" = "1" ] && echo "enabled" || echo "disabled")"
echo "  Command:      ${CMD[*]}"
echo ""

"${DOCKER_ARGS[@]}" "$BENCH_IMAGE" "${CMD[@]}"
