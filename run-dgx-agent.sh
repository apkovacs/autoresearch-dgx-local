#!/usr/bin/env bash
# run-dgx-agent.sh — Launch autonomous autoresearch agent on DGX Spark
#
# Runs the full experiment loop: Ollama (local LLM) + Claude Code + training.
# No external API calls needed — everything runs on-device.
#
# Docker configuration inspired by:
#   github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark
# Claude Code local model setup based on:
#   medium.com/@luongnv89/run-claude-code-on-local-cloud-models-in-5-minutes
#
# Usage:
#   bash run-dgx-agent.sh                              # default (Qwen3.6 27B)
#   OLLAMA_MODEL=gemma4:27b bash run-dgx-agent.sh      # use Gemma 4
#   OLLAMA_MODEL=qwen2.5-coder:14b bash run-dgx-agent.sh  # smaller model
#
# Environment variables:
#   OLLAMA_MODEL      Ollama model tag (default: qwen3.6:27b)
#   SHARD_CACHE_DIR   Host path for persistent training shards (default: ~/.cache/autoresearch)
#   OLLAMA_MODELS     Host path for persistent Ollama model weights (default: ~/.ollama/models)
#   DOCKER_IMAGE      Base Docker image (default: nvcr.io/nvidia/pytorch:25.12-py3)
#   SHM_SIZE          Shared memory size (default: 64gb)

set -euo pipefail

# --- Defaults ---
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"
SHARD_CACHE_DIR="${SHARD_CACHE_DIR:-$HOME/.cache/autoresearch}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
SHM_SIZE="${SHM_SIZE:-64gb}"
CONTAINER_NAME="autoresearch-dgx-agent"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash run-dgx-agent.sh [-h|--help]"
            echo ""
            echo "Launches the autonomous autoresearch agent with a local LLM."
            echo ""
            echo "Environment variables:"
            echo "  OLLAMA_MODEL     Model to use (default: qwen3.6:27b)"
            echo "  SHARD_CACHE_DIR  Persistent shard storage (default: ~/.cache/autoresearch)"
            echo "  OLLAMA_MODELS    Persistent model weights (default: ~/.ollama/models)"
            echo "  DOCKER_IMAGE     Docker image (default: nvcr.io/nvidia/pytorch:25.12-py3)"
            echo "  SHM_SIZE         Shared memory (default: 64gb)"
            echo ""
            echo "Tested models:"
            echo "  qwen3.6:27b          ~18GB  Strong code reasoning (default)"
            echo "  gemma4:27b           ~16GB  Strong general + code capability"
            echo "  gemma4:12b           ~7GB   Good capability, more memory headroom"
            echo "  qwen2.5-coder:14b    ~8GB   Purpose-built for code tasks"
            echo "  qwen3:8b             ~5GB   Lightweight option"
            echo ""
            echo "Any Ollama-compatible model works — set OLLAMA_MODEL to the tag."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Pre-flight checks ---
echo "=== DGX Spark Autoresearch Agent Launcher ==="
echo ""

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed."
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon is not running."
    exit 1
fi

# --- Create persistent directories ---
mkdir -p "$SHARD_CACHE_DIR"
mkdir -p "$OLLAMA_MODELS"

echo "Configuration:"
echo "  LLM model:        $OLLAMA_MODEL"
echo "  Docker image:     $DOCKER_IMAGE"
echo "  Shard cache:      $SHARD_CACHE_DIR"
echo "  Ollama models:    $OLLAMA_MODELS"
echo "  Shared memory:    $SHM_SIZE"
echo "  Container name:   $CONTAINER_NAME"
echo ""

# --- Build the in-container setup script ---
SETUP_SCRIPT=$(cat <<'INNEREOF'
#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up autonomous agent environment ==="

# 1. Install Python dependencies
echo "[1/5] Installing Python dependencies..."
pip install -q rustbpe huggingface_hub tiktoken pyarrow requests

# 2. Install Ollama
echo "[2/5] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq zstd >/dev/null 2>&1
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 3. Start Ollama server
echo "[3/5] Starting Ollama server..."
ollama serve &>/dev/null &
OLLAMA_PID=$!
sleep 3

# Wait for Ollama to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "  Ollama server ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Ollama server failed to start."
        exit 1
    fi
    sleep 1
done

# 4. Pull the model (skips if already cached via volume mount)
echo "[4/5] Pulling model: $OLLAMA_MODEL ..."
ollama pull "$OLLAMA_MODEL"

# 5. Install Claude Code
echo "[5/5] Installing Claude Code..."
if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code 2>/dev/null || {
        echo "  Installing Node.js first..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash - &>/dev/null
        apt-get install -y nodejs &>/dev/null
        npm install -g @anthropic-ai/claude-code
    }
fi

# Configure Claude Code to use local Ollama
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_API_KEY="ollama"
export ANTHROPIC_MODEL="$OLLAMA_MODEL"

echo ""
echo "=== Environment ready ==="
echo "  Ollama:      http://localhost:11434"
echo "  Model:       $OLLAMA_MODEL"
echo "  Claude Code: $(claude --version 2>/dev/null || echo 'installed')"
echo ""

# Prepare data
echo "=== Preparing training data ==="
python prepare.py --num-shards 10

echo ""
echo "=== Launching autonomous agent ==="
echo "  The agent will read program.md and begin the experiment loop."
echo "  Press Ctrl+C to stop."
echo ""

# Set up logging directories
mkdir -p logs/transcripts

# Write initial event to event log
python3 -c "
import json, time
from datetime import datetime, timezone
event = {
    'ts': datetime.now(timezone.utc).isoformat(),
    'elapsed_s': time.monotonic(),
    'event': 'orchestrator_start',
    'mode': 'base',
    'tag': 'agent',
    'config': 'run-dgx-agent.sh',
}
with open('logs/events.jsonl', 'a') as f:
    f.write(json.dumps(event) + '\n')
"

echo "  Event log:    logs/events.jsonl"
echo "  Transcript:   logs/transcripts/agent.jsonl"
echo ""
echo "  Monitor in another terminal:"
echo "    bash monitor-game.sh --transcript   (agent thinking + tool calls)"
echo "    bash monitor-game.sh --events       (event stream)"
echo ""

# Launch Claude Code with stream-json output, capturing transcript
claude -p --output-format stream-json "$(cat program.md)" \
    > >(tee logs/transcripts/agent.jsonl) 2>&1
INNEREOF
)

# --- Launch container ---
echo "Starting autonomous agent container..."
docker run -it --rm \
    --gpus all \
    --ipc=host \
    --shm-size "$SHM_SIZE" \
    --oom-score-adj 1000 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name "$CONTAINER_NAME" \
    -v "$(pwd)":/workspace \
    -v "$SHARD_CACHE_DIR":/cache/autoresearch \
    -v "$OLLAMA_MODELS":/root/.ollama/models \
    -e AUTORESEARCH_CACHE_DIR=/cache/autoresearch \
    -e OLLAMA_MODEL="$OLLAMA_MODEL" \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -w /workspace \
    "$DOCKER_IMAGE" \
    bash -c "$SETUP_SCRIPT"
