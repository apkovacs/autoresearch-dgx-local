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

# Restore .git ownership to host user on exit (container runs as root,
# which creates root-owned .git/objects that the host user can't write to)
fix_git_ownership() {
    if [ -n "\${HOST_UID:-}" ] && [ -n "\${HOST_GID:-}" ]; then
        chown -R "\$HOST_UID:\$HOST_GID" /workspace/.git 2>/dev/null || true
    fi
}
trap fix_git_ownership EXIT

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

# Pre-configure git (avoids permission prompts and identity errors inside Claude Code)
git config --global --add safe.directory /workspace
git config --global user.email "agent@autoresearch.local"
git config --global user.name "AutoResearch Agent"

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
      "Write(results.tsv)",
      "Write(/workspace/run.log)",
      "Write(run.log)",
      "Bash(ls /cache/*)",
      "Bash(bash run_experiment.sh*)",
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
      "Bash(git reset*)",
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

# Create experiment runner script
cat > /workspace/run_experiment.sh << 'RUNEXP'
#!/usr/bin/env bash
python train.py > run.log 2>&1
exit_code=$?
echo "=== Experiment finished (exit code: $exit_code) ==="
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:" run.log 2>/dev/null || echo "(no metrics found — check run.log for errors)"
exit $exit_code
RUNEXP
chmod +x /workspace/run_experiment.sh

# Write CLAUDE.md so the agent knows the environment is ready
cat > /workspace/CLAUDE.md << 'CLAUDEMD'
# Environment Notes — READ THIS FIRST

## IMPORTANT: Command differences from program.md
- Use `bash run_experiment.sh` to run experiments (NOT `python train.py > run.log 2>&1`)
  This wrapper captures output to run.log and prints key metrics when done.
- Do NOT use output redirection (`>`) in bash commands — it is blocked by the sandbox.
- Use `python train.py` instead of `uv run train.py` (there is no uv in this environment)
- Data is already prepared — do NOT run prepare.py
- The cache is at /cache/autoresearch (also symlinked to ~/.cache/autoresearch)

## What you can modify
Only train.py — this is the single file you should edit.
Use the Edit tool to modify train.py. Do NOT use python3 -c to rewrite files.

## Experiment loop — follow this EXACTLY

For EVERY experiment (including the baseline), do ALL of these steps:

1. Edit train.py with your experimental idea (skip for baseline)
2. `git add train.py && git commit -m "description of change"` (skip for baseline)
3. `bash run_experiment.sh`
4. `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. Get the commit hash: `git rev-parse --short HEAD`
6. **Log the result to results.tsv NOW** — run this command:
   `printf "%s\t%s\t%s\t%s\t%s\n" "COMMIT" "VAL_BPB" "MEM_GB" "STATUS" "DESCRIPTION" >> results.tsv`
   (replace the placeholder values with actuals, e.g.:)
   `printf "%s\t%s\t%s\t%s\t%s\n" "a1b2c3d" "1.879972" "7.6" "keep" "baseline" >> results.tsv`
7. If val_bpb IMPROVED (lower): keep the commit, move on
8. If val_bpb did NOT improve: `git reset --hard HEAD~1` to revert

**AFTER EVERY EXPERIMENT you MUST run the printf command in step 6 to log to results.tsv.**
**To revert failed experiments, MUST use `git reset --hard HEAD~1`.**
Do not manually undo code changes. Do not use python3 -c to edit files.
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
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -w /workspace \
    "$DOCKER_IMAGE" \
    bash -c "$SETUP_SCRIPT"
