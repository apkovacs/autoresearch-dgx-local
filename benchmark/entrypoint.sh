#!/usr/bin/env bash
# Benchmark container entrypoint — starts Ollama, pulls model, runs benchmark.
set -euo pipefail

OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}"
export OLLAMA_KEEP_ALIVE

# Start Ollama server in background
echo "[bench] Starting Ollama server (keep-alive: $OLLAMA_KEEP_ALIVE)..."
ollama serve &>/dev/null &
OLLAMA_PID=$!

# Wait for Ollama to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "[bench] Ollama ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Ollama failed to start after 30s."
        exit 1
    fi
    sleep 1
done

# Pull or import model if OLLAMA_MODEL is set (run-bench.sh passes this)
if [ -n "${OLLAMA_GGUF_FILE:-}" ]; then
    # Custom GGUF import — one-time cost, model store is a persistent mount
    if ollama show "${OLLAMA_MODEL:?OLLAMA_MODEL must be set with OLLAMA_GGUF}" &>/dev/null; then
        echo "[bench] Custom model already imported: $OLLAMA_MODEL"
    else
        echo "[bench] Importing GGUF as $OLLAMA_MODEL (num_ctx=${OLLAMA_NUM_CTX:-32768})..."
        ollama create "$OLLAMA_MODEL" -f /dev/stdin <<GGUFMODELFILE
FROM $OLLAMA_GGUF_FILE
PARAMETER num_ctx ${OLLAMA_NUM_CTX:-32768}
GGUFMODELFILE
    fi
elif [ -n "${OLLAMA_MODEL:-}" ]; then
    echo "[bench] Pulling model: $OLLAMA_MODEL"
    ollama pull "$OLLAMA_MODEL"
    echo "[bench] Model ready."
fi

# Also pull any models specified in --models flag (for bench_edit_quality.py)
# Parse --models from the command line arguments
MODELS_TO_PULL=()
NEXT_IS_MODEL=false
for arg in "$@"; do
    if [ "$NEXT_IS_MODEL" = true ]; then
        # Collect model names until we hit another flag
        if [[ "$arg" == --* ]]; then
            NEXT_IS_MODEL=false
        else
            MODELS_TO_PULL+=("$arg")
            continue
        fi
    fi
    if [ "$arg" = "--models" ]; then
        NEXT_IS_MODEL=true
    fi
    if [ "$arg" = "--model" ]; then
        NEXT_IS_MODEL=true
    fi
done

for model in "${MODELS_TO_PULL[@]}"; do
    if [ "$model" != "${OLLAMA_MODEL:-}" ]; then
        echo "[bench] Pulling additional model: $model"
        ollama pull "$model"
    fi
done

echo "[bench] Running: $*"
echo ""

# Run the actual benchmark command
exec "$@"
