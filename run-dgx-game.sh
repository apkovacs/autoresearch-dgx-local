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
#   OLLAMA_KEEP_ALIVE Model unload delay after last request (default: 0, unload immediately)

set -euo pipefail

# --- Defaults ---
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"
SHARD_CACHE_DIR="${SHARD_CACHE_DIR:-$HOME/.cache/autoresearch}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
SHM_SIZE="${SHM_SIZE:-64gb}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}"
CONTAINER_NAME="autoresearch-dgx-local-game"
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
echo "=== autoresearch-dgx-local Game Orchestrator ==="
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

# Install dependencies (skipped if using pre-built image)
if command -v ollama &>/dev/null && command -v claude &>/dev/null && python -c "import rustbpe" 2>/dev/null; then
    echo "[deps] Pre-built image detected — skipping installs"
else
    echo "[1/5] Installing Python dependencies..."
    pip install -q rustbpe huggingface_hub tiktoken pyarrow requests pyyaml

    echo "[2/5] Installing Ollama..."
    if ! command -v ollama &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq zstd >/dev/null 2>&1
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    echo "[4/5] Installing Claude Code..."
    if ! command -v claude &>/dev/null; then
        npm install -g @anthropic-ai/claude-code 2>/dev/null || {
            curl -fsSL https://deb.nodesource.com/setup_22.x | bash - &>/dev/null
            apt-get install -y nodejs &>/dev/null
            npm install -g @anthropic-ai/claude-code
        }
    fi
fi

# Start Ollama server (OLLAMA_KEEP_ALIVE is passed via docker run -e;
# default 0 unloads model between agent turns, freeing ~18GB for training)
echo "[start] Starting Ollama server (keep-alive: \${OLLAMA_KEEP_ALIVE:-0})..."
export OLLAMA_KEEP_ALIVE="\${OLLAMA_KEEP_ALIVE:-0}"
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

# Pull the model (skips if already cached via volume mount)
echo "[start] Pulling model: $OLLAMA_MODEL ..."
ollama pull "$OLLAMA_MODEL"

# Cap output tokens to prevent degenerate repetitive thinking loops
echo "[start] Creating output-capped model alias..."
ollama create "${OLLAMA_MODEL}-capped" -f /dev/stdin <<MODELFILE
FROM $OLLAMA_MODEL
PARAMETER num_predict 16384
MODELFILE
OLLAMA_MODEL="${OLLAMA_MODEL}-capped"
echo "  Using capped model: $OLLAMA_MODEL (num_predict=16384)"

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
      "Edit",
      "Read",
      "Write",
      "Bash(ls /cache/*)",
      "Bash(bash run_experiment.sh*)",
      "Bash(bash log_result.sh*)",
      "Bash(bash revert_train.sh*)",
      "Bash(python prepare.py*)",
      "Bash(python3 prepare.py*)",
      "Bash(python -c *)",
      "Bash(python3 -c *)",
      "Bash(git status*)",
      "Bash(git diff*)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git log --oneline*)",
      "Bash(git checkout *)",
      "Bash(git branch*)",
      "Bash(git stash*)",
      "Bash(git switch*)",
      "Bash(git restore*)",
      "Bash(git reset*)",
      "Bash(git rev-parse*)",
      "Bash(grep *)",
      "Bash(sed *)",
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
    "deny": [
      "Task",
      "TaskCreate",
      "TaskGet",
      "TaskOutput",
      "TaskStop",
      "TaskUpdate",
      "TaskList",
      "Monitor",
      "Agent",
      "AskUserQuestion",
      "WebSearch",
      "WebFetch",
      "CronCreate",
      "CronDelete",
      "CronList",
      "NotebookEdit",
      "PushNotification",
      "EnterPlanMode",
      "ExitPlanMode",
      "EnterWorktree",
      "ExitWorktree",
      "ScheduleWakeup",
      "Skill"
    ]
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
# Wrapper: validates train.py, runs it, and captures output to run.log
# - Syntax-checks before running (catches bad edits instantly)
# - Prints heartbeat every 30s so the Bash tool does not time out
# - Saves .train.py.lastgood after each successful run
# Usage: bash run_experiment.sh
# Takes ~5 minutes. Do NOT poll run.log — just wait for this to finish.

# Step 1: Syntax check — fail fast on bad edits
echo "Checking train.py syntax..."
if ! python -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2> /tmp/syntax_err.txt; then
    echo "=== SYNTAX ERROR in train.py ==="
    cat /tmp/syntax_err.txt
    echo ""
    echo "Fix the error or run: bash revert_train.sh"
    exit 2
fi

# Step 2: Run training in background, heartbeat to keep Bash tool alive
echo "Training started — this takes ~5 minutes. Wait for results below."
python train.py > run.log 2>&1 &
TRAIN_PID=$!

# Heartbeat: print progress every 30s so the tool doesn't time out
SECONDS_ELAPSED=0
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + 30))
    # Show latest progress from run.log (last step line)
    LATEST=$(grep -oP '\d+\.\d+%' run.log 2>/dev/null | tail -1)
    echo "  ... training ${LATEST:-starting} (${SECONDS_ELAPSED}s elapsed)"
done

# Collect exit code
wait $TRAIN_PID
exit_code=$?

# Step 3: Clean run.log — training uses \r for progress, making the file
# appear as one huge line. Replace \r with \n so tail/head work properly.
if [ -f run.log ]; then
    tr '\r' '\n' < run.log | grep -v '^$' > run.log.tmp && mv run.log.tmp run.log
fi

# Step 4: Print results
echo "=== Experiment finished (exit code: $exit_code) ==="
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:" run.log 2>/dev/null || echo "(no metrics found — check run.log for errors)"

# Step 5: Save last-good backup if training produced results
if grep -q "^val_bpb:" run.log 2>/dev/null; then
    cp train.py .train.py.lastgood
fi

exit $exit_code
RUNEXP
chmod +x /workspace/run_experiment.sh

# Create revert script (restores train.py from last-good backup)
cat > /workspace/revert_train.sh << 'REVERT'
#!/usr/bin/env bash
# Restores train.py from the last successful version.
# Usage: bash revert_train.sh
if [ -f .train.py.lastgood ]; then
    cp .train.py.lastgood train.py
    echo "Reverted train.py to last working version."
    echo "Diff from current git HEAD:"
    git diff train.py 2>/dev/null | head -30
else
    echo "No .train.py.lastgood found. Reverting to git HEAD instead."
    git checkout -- train.py
    echo "Reverted train.py to last committed version."
fi
REVERT
chmod +x /workspace/revert_train.sh

# Save initial train.py as the first last-good backup
cp /workspace/train.py /workspace/.train.py.lastgood

# Create results logger script (>> redirect is blocked by Claude Code sandbox)
cat > /workspace/log_result.sh << 'LOGEXP'
#!/usr/bin/env bash
# Wrapper: appends experiment result to results.tsv (with UTC timestamp)
# Usage: bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION
if [ $# -lt 5 ]; then
    echo "Usage: bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION"
    echo "Example: bash log_result.sh a1b2c3d 1.879972 7.6 keep baseline"
    exit 1
fi
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$TS" >> results.tsv
echo "Logged to results.tsv: $1  val_bpb=$2  mem=$3GB  status=$4  $5  [$TS]"
LOGEXP
chmod +x /workspace/log_result.sh

# Write CLAUDE.md so the agent knows the environment is ready
cat > /workspace/CLAUDE.md << 'CLAUDEMD'
# START HERE — Setup is done, begin experimenting immediately

## Step 1: Run the baseline NOW

Use the Read tool to read train.py. Then run:

```
bash run_experiment.sh
```

This takes ~5 minutes and prints heartbeat progress. Wait for it to finish.

## Step 2: Log the baseline result

After it finishes, extract results and log them:
```
grep "^val_bpb:\|^peak_vram_mb:" run.log
git rev-parse --short HEAD
bash log_result.sh COMMIT VAL_BPB MEM_GB keep baseline
```

## Step 3: Begin experiment loop

Repeat this cycle to beat the baseline val_bpb:

1. Use the **Edit** tool to modify train.py (the only file you edit)
2. `git add train.py` then `git commit -m "description"`
3. `bash run_experiment.sh` — wait for it (~5 min)
4. If exit code 2 (syntax error): `bash revert_train.sh`, fix edit, retry
5. `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. `git rev-parse --short HEAD`
7. `bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION`
8. If val_bpb improved: keep. If not: `git reset --hard HEAD~1`

## Rules

- Tools you can use: **Bash**, **Edit**, **Read** — nothing else
- Do NOT use `>` or `>>` redirection — blocked by sandbox
- Do NOT use `python train.py` directly — always use `bash run_experiment.sh`
- Do NOT run prepare.py — data is already prepared
- Do NOT run commands in background — run directly and wait
- Do NOT use `python3 -c` to edit files — use the Edit tool
- Run git commands one at a time (e.g. `git add train.py` then `git commit -m "msg"`), not chained with `&&`

## Recovery

- `bash revert_train.sh` — restore last working train.py
- `bash run_experiment.sh` auto-checks syntax before training
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
    -e OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -w /workspace \
    "$DOCKER_IMAGE" \
    bash -c "$SETUP_SCRIPT"
