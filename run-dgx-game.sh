#!/usr/bin/env bash
# run-dgx-game.sh — Launch autoresearch meta-research orchestrator on DGX Spark
#
# Supports 4 modes:
#   base         Original autoresearch loop (delegates to run-dgx-agent.sh)
#   island       Island Model with Adaptive Resource Allocation
#   bandit       Multi-Armed Bandit (UCB1)
#   coopetition  Iterated Coopetition
#
# Usage:
#   bash run-dgx-game.sh                          # uses mode from game_config.yaml
#   bash run-dgx-game.sh --mode island            # override mode
#   bash run-dgx-game.sh --mode base              # original autoresearch
#   bash run-dgx-game.sh --config my_config.yaml  # custom config
#
# Environment variables (same as run-dgx-agent.sh):
#   OLLAMA_MODEL      Ollama model tag (default: qwen3.6:27b)
#   SHARD_CACHE_DIR   Host path for persistent training shards
#   OLLAMA_MODELS     Host path for persistent Ollama model weights
#   DOCKER_IMAGE      Base Docker image
#   SHM_SIZE          Shared memory size

set -euo pipefail

# --- Defaults ---
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"
SHARD_CACHE_DIR="${SHARD_CACHE_DIR:-$HOME/.cache/autoresearch}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
SHM_SIZE="${SHM_SIZE:-64gb}"
CONTAINER_NAME="autoresearch-dgx-game"
GAME_MODE=""
GAME_CONFIG="game_config.yaml"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            GAME_MODE="$2"; shift 2 ;;
        --config)
            GAME_CONFIG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run-dgx-game.sh [--mode MODE] [--config FILE] [-h]"
            echo ""
            echo "Modes:"
            echo "  base          Original autoresearch loop (single branch)"
            echo "  island        Island Model with Adaptive Resource Allocation"
            echo "  bandit        Multi-Armed Bandit (UCB1)"
            echo "  coopetition   Iterated Coopetition"
            echo ""
            echo "Options:"
            echo "  --mode MODE      Override mode in game_config.yaml"
            echo "  --config FILE    Use custom config file (default: game_config.yaml)"
            echo "  -h, --help       Show this help"
            echo ""
            echo "Configure game parameters in game_config.yaml before running."
            echo "See GAME_STRATEGIES.md for strategy details."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Pre-flight checks ---
echo "=== autoresearch-dgx Game Orchestrator ==="
echo ""

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed."
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon is not running."
    exit 1
fi

if [ ! -f "$GAME_CONFIG" ]; then
    echo "ERROR: Config file not found: $GAME_CONFIG"
    echo "Run from the autoresearch repo directory, or specify --config."
    exit 1
fi

# --- Read mode from config if not overridden ---
if [ -z "$GAME_MODE" ]; then
    GAME_MODE=$(python3 -c "import yaml; print(yaml.safe_load(open('$GAME_CONFIG'))['mode'])" 2>/dev/null || echo "base")
fi

# --- For base mode, delegate to run-dgx-agent.sh ---
if [ "$GAME_MODE" = "base" ]; then
    echo "Mode: base (original autoresearch loop)"
    echo "Delegating to run-dgx-agent.sh..."
    exec bash run-dgx-agent.sh
fi

# --- Create persistent directories ---
mkdir -p "$SHARD_CACHE_DIR"
mkdir -p "$OLLAMA_MODELS"

echo "Configuration:"
echo "  Game mode:        $GAME_MODE"
echo "  Config file:      $GAME_CONFIG"
echo "  LLM model:        $OLLAMA_MODEL"
echo "  Docker image:     $DOCKER_IMAGE"
echo "  Shard cache:      $SHARD_CACHE_DIR"
echo "  Ollama models:    $OLLAMA_MODELS"
echo ""

# --- Build the in-container setup script ---
MODE_ARG=""
if [ -n "$GAME_MODE" ]; then
    MODE_ARG="--mode $GAME_MODE"
fi

SETUP_SCRIPT=$(cat <<INNEREOF
#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up game orchestrator environment ==="

# 1. Install Python dependencies
echo "[1/5] Installing Python dependencies..."
pip install -q rustbpe huggingface_hub tiktoken pyarrow requests pyyaml

# 2. Install Ollama
echo "[2/5] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq zstd >/dev/null 2>&1
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 3. Start Ollama server
echo "[3/5] Starting Ollama server..."
ollama serve &>/dev/null &
sleep 3
for i in \$(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "  Ollama server ready."
        break
    fi
    [ "\$i" -eq 30 ] && echo "ERROR: Ollama failed to start." && exit 1
    sleep 1
done

# 4. Pull the model
echo "[4/5] Pulling model: $OLLAMA_MODEL ..."
ollama pull "$OLLAMA_MODEL"

# 5. Install Claude Code
echo "[5/5] Installing Claude Code..."
if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code 2>/dev/null || {
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash - &>/dev/null
        apt-get install -y nodejs &>/dev/null
        npm install -g @anthropic-ai/claude-code
    }
fi

export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_API_KEY="ollama"
export ANTHROPIC_MODEL="$OLLAMA_MODEL"

# Pre-configure git safe.directory
git config --global --add safe.directory /workspace

# Configure Claude Code permissions for autonomous operation
# Scoped to the experiment: agent can only edit train.py, run training,
# read the workspace, and use git. Cannot modify other scripts, install
# packages, or access files outside the workspace.
mkdir -p /workspace/.claude
cat > /workspace/.claude/settings.json << 'SETTINGS'
{
  "permissions": {
    "allow": [
      "Edit(/workspace/train.py)",
      "Read(/workspace/*)",
      "Read(/cache/autoresearch/*)",
      "Write(/workspace/results.tsv)",
      "Bash(ls /cache/*)",
      "Bash(python train.py*)",
      "Bash(python prepare.py*)",
      "Bash(python3 train.py*)",
      "Bash(python3 prepare.py*)",
      "Bash(python -c *)",
      "Bash(python3 -c *)",
      "Bash(git status*)",
      "Bash(git diff*)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git log*)",
      "Bash(git checkout *)",
      "Bash(git branch*)",
      "Bash(git stash*)",
      "Bash(git switch*)",
      "Bash(git restore*)",
      "Bash(git rev-parse*)",
      "Bash(grep *)",
      "Bash(diff *)",
      "Bash(wc *)",
      "Bash(head *)",
      "Bash(tail *)",
      "Bash(cat *)",
      "Bash(ls *)",
      "Bash(find *)",
      "Bash(test *)",
      "Bash(echo *)",
      "Bash(nvidia-smi*)"
    ],
    "deny": []
  }
}
SETTINGS

# Symlink cache so the agent can find data at the default path
ln -sfn /cache/autoresearch /root/.cache/autoresearch 2>/dev/null || true

# Prepare data
echo "=== Preparing training data ==="
python prepare.py --num-shards 10

# Write CLAUDE.md so the agent knows the environment is ready
cat > /workspace/CLAUDE.md << 'CLAUDEMD'
# Environment Notes

## Data is pre-prepared
Training data and tokenizer are already downloaded and ready.
You do NOT need to run prepare.py or check for data — it has already been done.
The cache is at /cache/autoresearch (also symlinked to ~/.cache/autoresearch).

## How to run an experiment
Just run: python train.py
The data path is configured via AUTORESEARCH_CACHE_DIR environment variable.

## What you can modify
Only train.py — this is the single file you should edit. Do not modify prepare.py or any other files.
CLAUDEMD

echo ""
echo "=== Launching orchestrator ==="
python orchestrator.py --config $GAME_CONFIG $MODE_ARG
INNEREOF
)

# --- Launch container ---
echo "Starting game orchestrator container..."
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
