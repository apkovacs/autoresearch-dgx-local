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
#   OLLAMA_KEEP_ALIVE Model unload delay after last request (default: 0, unload immediately)
#   OLLAMA_GGUF       Host path to a local GGUF file to import instead of pulling
#                     (for community quants not in the Ollama library, e.g.
#                     DeepSeek V4 Flash Dwarf Star). OLLAMA_MODEL becomes the
#                     name of the imported model.
#   OLLAMA_NUM_CTX    Context window for imported GGUF models (default: 32768)

set -euo pipefail

# --- Defaults ---
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"
SHARD_CACHE_DIR="${SHARD_CACHE_DIR:-$HOME/.cache/autoresearch}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
SHM_SIZE="${SHM_SIZE:-64gb}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}"
OLLAMA_GGUF="${OLLAMA_GGUF:-}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-32768}"
CONTAINER_NAME="autoresearch-dgx-local-agent"
MAX_RESTARTS=3
RESTART_COOLDOWN=10
EXPERIMENTS_PER_SESSION=30
AGENT_MODE="${AGENT_MODE:-guarded}"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            AGENT_MODE="$2"; shift 2 ;;
        --max-restarts)
            MAX_RESTARTS="$2"; shift 2 ;;
        --no-restart)
            MAX_RESTARTS=0; shift ;;
        --experiments-per-session)
            EXPERIMENTS_PER_SESSION="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run-dgx-agent.sh [--mode guarded|minimal] [--max-restarts N] [--no-restart] [--experiments-per-session N] [-h|--help]"
            echo ""
            echo "Launches the autonomous autoresearch agent with a local LLM."
            echo ""
            echo "Options:"
            echo "  --mode MODE                  Agent mode (default: guarded)"
            echo "                                 guarded — behavioral guardrails for local models:"
            echo "                                   output token cap, action-first prompt, narrow"
            echo "                                   permission allowlist, explicit rules"
            echo "                                 minimal — Karpathy's original design for capable"
            echo "                                   models: program.md drives everything, no token"
            echo "                                   cap, all permissions granted, facts-only CLAUDE.md"
            echo "  --max-restarts N             Auto-restart agent up to N times (default: 3)"
            echo "  --no-restart                 Disable auto-restart (exit on first stop)"
            echo "  --experiments-per-session N   Restart for fresh context after N experiments (default: 30)"
            echo "  -h, --help                   Show this help"
            echo ""
            echo "Environment variables:"
            echo "  OLLAMA_MODEL     Model to use (default: qwen3.6:27b)"
            echo "  SHARD_CACHE_DIR  Persistent shard storage (default: ~/.cache/autoresearch)"
            echo "  OLLAMA_MODELS    Persistent model weights (default: ~/.ollama/models)"
            echo "  DOCKER_IMAGE     Docker image (default: nvcr.io/nvidia/pytorch:25.12-py3)"
            echo "  SHM_SIZE         Shared memory (default: 64gb)"
            echo "  OLLAMA_KEEP_ALIVE  Model unload delay (default: 0 = unload immediately)"
            echo "  OLLAMA_GGUF      Path to a local GGUF file to import instead of pulling"
            echo "  OLLAMA_NUM_CTX   Context window for imported GGUF (default: 32768)"
            echo ""
            echo "Custom GGUF example (DeepSeek V4 Flash Dwarf Star quant):"
            echo "  OLLAMA_GGUF=~/models/deepseek-v4-flash-dwarf.gguf bash run-dgx-agent.sh"
            echo ""
            echo "Tested models:"
            echo "  qwen3.6:27b          ~18GB  Strong code reasoning (default)"
            echo "  gemma4:26b           ~18GB  Strong general + code capability"
            echo "  gemma4:e4b           ~10GB  Good capability, more memory headroom"
            echo "  gemma4:e2b           ~7GB   Lightweight edge model"
            echo "  qwen2.5-coder:14b    ~8GB   Purpose-built for code tasks"
            echo "  qwen3:8b             ~5GB   Lightweight option"
            echo ""
            echo "Any Ollama-compatible model works — set OLLAMA_MODEL to the tag."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$AGENT_MODE" != "guarded" ] && [ "$AGENT_MODE" != "minimal" ]; then
    echo "ERROR: --mode must be 'guarded' or 'minimal' (got: $AGENT_MODE)"
    exit 1
fi

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

# --- Custom GGUF import (community quants not in the Ollama library) ---
GGUF_ARGS=()
if [ -n "$OLLAMA_GGUF" ]; then
    if [ ! -f "$OLLAMA_GGUF" ]; then
        echo "ERROR: OLLAMA_GGUF file not found: $OLLAMA_GGUF"
        exit 1
    fi
    # If the user didn't override OLLAMA_MODEL, derive a name from the filename
    if [ "$OLLAMA_MODEL" = "qwen3.6:27b" ]; then
        OLLAMA_MODEL=$(basename "$OLLAMA_GGUF" .gguf | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed 's/-*$//')
        echo "Derived model name from GGUF filename: $OLLAMA_MODEL"
    fi
    GGUF_ARGS=(
        -v "$OLLAMA_GGUF":/gguf/model.gguf:ro
        -e OLLAMA_GGUF_FILE=/gguf/model.gguf
        -e OLLAMA_NUM_CTX="$OLLAMA_NUM_CTX"
    )
fi

# --- Create persistent directories ---
mkdir -p "$SHARD_CACHE_DIR"
mkdir -p "$OLLAMA_MODELS"

echo "Configuration:"
echo "  Agent mode:       $AGENT_MODE"
echo "  LLM model:        $OLLAMA_MODEL"
[ -n "$OLLAMA_GGUF" ] && echo "  Custom GGUF:      $OLLAMA_GGUF (num_ctx=$OLLAMA_NUM_CTX)"
echo "  Docker image:     $DOCKER_IMAGE"
echo "  Shard cache:      $SHARD_CACHE_DIR"
echo "  Ollama models:    $OLLAMA_MODELS"
echo "  Shared memory:    $SHM_SIZE"
echo "  Ollama keep-alive: $OLLAMA_KEEP_ALIVE"
echo "  Container name:   $CONTAINER_NAME"
echo "  Max restarts:     $MAX_RESTARTS"
echo "  Exp/session:      $EXPERIMENTS_PER_SESSION"
echo ""

# --- Build the in-container setup script ---
SETUP_SCRIPT=$(cat <<'INNEREOF'
#!/usr/bin/env bash
set -euo pipefail

# Restore .git ownership to host user on exit (container runs as root,
# which creates root-owned .git/objects that the host user can't write to)
fix_git_ownership() {
    if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
        chown -R "$HOST_UID:$HOST_GID" /workspace/.git 2>/dev/null || true
    fi
}
trap fix_git_ownership EXIT

echo "=== Setting up autonomous agent environment ==="

# Install dependencies (skipped if using pre-built image)
if command -v ollama &>/dev/null && command -v claude &>/dev/null && python -c "import rustbpe" 2>/dev/null; then
    echo "[deps] Pre-built image detected — skipping installs"
else
    echo "[1/5] Installing Python dependencies..."
    pip install -q rustbpe huggingface_hub tiktoken pyarrow requests

    echo "[2/5] Installing Ollama..."
    if ! command -v ollama &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq zstd >/dev/null 2>&1
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    echo "[4/5] Installing Claude Code..."
    if ! command -v claude &>/dev/null; then
        npm install -g @anthropic-ai/claude-code 2>/dev/null || {
            echo "  Installing Node.js first..."
            curl -fsSL https://deb.nodesource.com/setup_22.x | bash - &>/dev/null
            apt-get install -y nodejs &>/dev/null
            npm install -g @anthropic-ai/claude-code
        }
    fi
fi

# Start Ollama server (OLLAMA_KEEP_ALIVE is passed via docker run -e;
# default 0 unloads model between agent turns, freeing ~18GB for training)
echo "[start] Starting Ollama server (keep-alive: ${OLLAMA_KEEP_ALIVE:-0})..."
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}"
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

# Pull or import the model
if [ -n "${OLLAMA_GGUF_FILE:-}" ]; then
    # Custom GGUF import (e.g. DeepSeek V4 Flash Dwarf Star quant).
    # ollama create hashes the file into the blob store; since the model
    # store is a persistent volume mount, this is a one-time cost — later
    # runs detect the existing model and skip the import.
    if ollama show "$OLLAMA_MODEL" &>/dev/null; then
        echo "[start] Custom model already imported: $OLLAMA_MODEL"
    else
        echo "[start] Importing GGUF as $OLLAMA_MODEL (num_ctx=${OLLAMA_NUM_CTX:-32768})..."
        echo "        Large files take several minutes to hash on first import."
        ollama create "$OLLAMA_MODEL" -f /dev/stdin <<GGUFMODELFILE
FROM $OLLAMA_GGUF_FILE
PARAMETER num_ctx ${OLLAMA_NUM_CTX:-32768}
GGUFMODELFILE
    fi
else
    # Pull from the Ollama library (skips if already cached via volume mount)
    echo "[start] Pulling model: $OLLAMA_MODEL ..."
    ollama pull "$OLLAMA_MODEL"
fi

# Cap output tokens to prevent degenerate repetitive thinking loops.
# Some models (especially Gemma 4) can enter infinite reasoning cycles
# where they repeat the same sentence thousands of times, consuming
# the entire output budget without ever emitting a tool call.
# num_predict limits each completion to ~16K tokens — enough for any
# legitimate tool call sequence, short enough to break the loop.
# Minimal mode skips the cap: capable models don't loop, and capping
# would contaminate any measurement of the model's natural behavior.
if [ "${AGENT_MODE:-guarded}" = "guarded" ]; then
    echo "[start] Creating output-capped model alias..."
    ollama create "${OLLAMA_MODEL}-capped" -f /dev/stdin <<MODELFILE
FROM $OLLAMA_MODEL
PARAMETER num_predict 16384
MODELFILE
    OLLAMA_MODEL="${OLLAMA_MODEL}-capped"
    echo "  Using capped model: $OLLAMA_MODEL (num_predict=16384)"
else
    echo "[start] Minimal mode — no output token cap."
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

# Pre-configure git (avoids permission prompts and identity errors inside Claude Code)
git config --global --add safe.directory /workspace
git config --global user.email "agent@autoresearch.local"
git config --global user.name "AutoResearch Agent"

# Configure Claude Code permissions for autonomous operation.
# Guarded mode: narrow allowlist scoped to the experiment — the agent can
# only edit train.py, run training, read the workspace, and use git.
# Minimal mode: no settings file — the agent runs with bypassPermissions
# so we observe the model's natural behavior with zero permission friction
# (the Docker container is the sandbox).
mkdir -p /workspace/.claude
rm -f /workspace/.claude/settings.json
if [ "${AGENT_MODE:-guarded}" = "guarded" ]; then
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
fi

# Symlink cache so the agent can find data at the default path
ln -sfn /cache/autoresearch /root/.cache/autoresearch 2>/dev/null || true

# Prepare data
echo "=== Preparing training data ==="
python prepare.py --num-shards 10

# Pre-create experiment branch and results.tsv so agent can skip setup
RUN_TAG=$(date +%b%d | tr '[:upper:]' '[:lower:]')
BRANCH_NAME="autoresearch/$RUN_TAG"
cd /workspace
git config --global --add safe.directory /workspace
if ! git rev-parse --verify "$BRANCH_NAME" &>/dev/null; then
    echo "Creating experiment branch: $BRANCH_NAME"
    git checkout -b "$BRANCH_NAME"
else
    echo "Switching to existing branch: $BRANCH_NAME"
    git checkout "$BRANCH_NAME"
fi

# Initialize results.tsv if it doesn't exist
if [ ! -f results.tsv ]; then
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\ttimestamp\n' > results.tsv
    echo "Created results.tsv with header"
fi

# Create experiment runner script (handles output redirect so the agent doesn't need to)
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

# Write CLAUDE.md so the agent knows the environment is ready.
# Guarded mode: action-first structure with explicit rules (small models
# respond better to imperative first-action prompts).
# Minimal mode: environment facts only — program.md drives the loop,
# exactly as in Karpathy's original design.
if [ "${AGENT_MODE:-guarded}" = "minimal" ]; then
cat > /workspace/CLAUDE.md << 'MINIMALMD'
# Environment notes

Setup is already complete — do not repeat it:
- Training data is prepared (do not run prepare.py)
- The experiment branch is checked out and results.tsv exists
- Git identity is configured

Helper scripts available in the workspace:
- `bash run_experiment.sh` — syntax-checks train.py, runs training (~5 min),
  prints val_bpb / peak_vram_mb, and heartbeats so the call doesn't time out.
  Use this instead of running train.py directly.
- `bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION` — appends a
  row to results.tsv (append redirection is unavailable in this environment).
- `bash revert_train.sh` — restores the last working train.py

Follow program.md for the research loop.
MINIMALMD
else
cat > /workspace/CLAUDE.md << CLAUDEMD
# START HERE — Setup is done, begin experimenting immediately

## Step 1: Run the baseline NOW

Use the Read tool to read train.py. Then run:

\`\`\`
bash run_experiment.sh
\`\`\`

This takes ~5 minutes and prints heartbeat progress. Wait for it to finish.

## Step 2: Log the baseline result

After it finishes, extract results and log them:
\`\`\`
grep "^val_bpb:\|^peak_vram_mb:" run.log
git rev-parse --short HEAD
bash log_result.sh COMMIT VAL_BPB MEM_GB keep baseline
\`\`\`

## Step 3: Begin experiment loop

Repeat this cycle to beat the baseline val_bpb:

1. Use the **Edit** tool to modify train.py (the only file you edit)
2. \`git add train.py\` then \`git commit -m "description"\`
3. \`bash run_experiment.sh\` — wait for it (~5 min)
4. If exit code 2 (syntax error): \`bash revert_train.sh\`, fix edit, retry
5. \`grep "^val_bpb:\|^peak_vram_mb:" run.log\`
6. \`git rev-parse --short HEAD\`
7. \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\`
8. If val_bpb improved: keep. If not: \`git reset --hard HEAD~1\`

## Rules

- Tools you can use: **Bash**, **Edit**, **Read** — nothing else
- Do NOT use \`>\` or \`>>\` redirection — blocked by sandbox
- Do NOT use \`python train.py\` directly — always use \`bash run_experiment.sh\`
- Do NOT run prepare.py — data is already prepared
- Do NOT run commands in background — run directly and wait
- Do NOT use \`python3 -c\` to edit files — use the Edit tool
- Run git commands one at a time (e.g. \`git add train.py\` then \`git commit -m "msg"\`), not chained with \`&&\`

## Recovery

- \`bash revert_train.sh\` — restore last working train.py
- \`bash run_experiment.sh\` auto-checks syntax before training
CLAUDEMD
fi

echo ""
echo "=== Launching autonomous agent ($AGENT_MODE mode) ==="
echo "  The agent will read program.md and begin the experiment loop."
echo "  Press Ctrl+C to stop."
echo ""

# Set up logging directories
mkdir -p logs/transcripts

# Write initial event to event log
log_event() {
    python3 -c "
import json, time, sys
from datetime import datetime, timezone
event = json.loads(sys.argv[1])
event['ts'] = datetime.now(timezone.utc).isoformat()
event['elapsed_s'] = time.monotonic()
with open('logs/events.jsonl', 'a') as f:
    f.write(json.dumps(event) + '\n')
" "$1"
}

log_event "{\"event\": \"orchestrator_start\", \"mode\": \"base\", \"tag\": \"agent\", \"agent_mode\": \"${AGENT_MODE:-guarded}\", \"config\": \"run-dgx-agent.sh\"}"

# Permission handling per mode:
# - guarded: dontAsk + the narrow allowlist in .claude/settings.json
# - minimal: bypassPermissions — zero permission friction, the Docker
#   container is the sandbox. IS_SANDBOX=1 lets Claude Code accept
#   bypassPermissions while running as root inside the container.
if [ "${AGENT_MODE:-guarded}" = "minimal" ]; then
    PERMISSION_FLAGS=(--permission-mode bypassPermissions)
    export IS_SANDBOX=1
else
    PERMISSION_FLAGS=(--permission-mode dontAsk)
fi

echo "  Event log:    logs/events.jsonl"
echo "  Transcript:   logs/transcripts/agent.jsonl"
echo ""
echo "  Monitor in another terminal:"
echo "    bash monitor-game.sh --transcript   (agent thinking + tool calls)"
echo "    bash monitor-game.sh --events       (event stream)"
echo ""

# --- Agent launch with auto-restart ---
# The agent is instructed to NEVER STOP. Any exit — clean or crash — is
# premature and should trigger a restart, UNLESS the user sent Ctrl+C.
MAX_RESTARTS=${MAX_RESTARTS:-3}
RESTART_COOLDOWN=${RESTART_COOLDOWN:-10}
ATTEMPT=0
USER_STOPPED=false

trap 'USER_STOPPED=true' INT TERM

count_experiments() {
    if [ -f results.tsv ]; then
        tail -n +2 results.tsv | wc -l | tr -d ' '
    else
        echo 0
    fi
}

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    TRANSCRIPT="logs/transcripts/agent_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
    BEFORE_COUNT=$(count_experiments)

    if [ "$ATTEMPT" -gt 1 ]; then
        echo ""
        echo "=== Agent restart (attempt $ATTEMPT/$((MAX_RESTARTS + 1))) ==="

        # Extract best and baseline val_bpb from results.tsv
        BEST_BPB="unknown"
        BASELINE_BPB="unknown"
        HAS_BASELINE=false
        if [ -f results.tsv ]; then
            # Best val_bpb among kept experiments (lowest non-zero value)
            BEST_BPB=$(awk -F'\t' 'NR>1 && $4=="keep" && $2+0>0 {print $2}' results.tsv | sort -n | head -1)
            [ -z "$BEST_BPB" ] && BEST_BPB="unknown"
            # Baseline is the first row with "baseline" in description
            BASELINE_BPB=$(awk -F'\t' 'NR>1 && $5~/baseline/ {print $2; exit}' results.tsv)
            [ -n "$BASELINE_BPB" ] && HAS_BASELINE=true || BASELINE_BPB="not yet run"
        fi

        # Update CLAUDE.md for resume — forward-looking, no history
        if [ "${AGENT_MODE:-guarded}" = "minimal" ]; then
        cat > /workspace/CLAUDE.md << MINRESUMEMD
# Environment notes (resumed session)

Setup is already complete — do not repeat it:
- Training data is prepared (do not run prepare.py)
- The experiment branch is checked out and results.tsv exists

Current state:
- Best val_bpb so far: **$BEST_BPB**
- Baseline val_bpb: $BASELINE_BPB
- Experiments completed: $BEFORE_COUNT
$([ "$HAS_BASELINE" = false ] && echo "- The baseline has NOT been run yet — run it first.")

Helper scripts available in the workspace:
- \`bash run_experiment.sh\` — syntax-checks train.py, runs training (~5 min),
  prints val_bpb / peak_vram_mb, and heartbeats so the call doesn't time out.
  Use this instead of running train.py directly.
- \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\` — appends a
  row to results.tsv (append redirection is unavailable in this environment).
- \`bash revert_train.sh\` — restores the last working train.py

Follow program.md for the research loop. Review results.tsv for what has
been tried already.
MINRESUMEMD
        else
        cat > /workspace/CLAUDE.md << RESUMEMD
# START HERE — Resume experimenting immediately

## Current state
- Best val_bpb so far: **$BEST_BPB** (beat this number)
- Baseline val_bpb: $BASELINE_BPB
- Experiments completed: $BEFORE_COUNT

## Step 1: Read train.py and begin

$([ "$HAS_BASELINE" = false ] && echo "The baseline has NOT been run yet. Run \`bash run_experiment.sh\` first." || echo "Use the Read tool to read train.py, then start experimenting.")

## Experiment loop

1. Use the **Edit** tool to modify train.py (the only file you edit)
2. \`git add train.py\` then \`git commit -m "description"\`
3. \`bash run_experiment.sh\` — wait for it (~5 min)
4. If exit code 2 (syntax error): \`bash revert_train.sh\`, fix edit, retry
5. \`grep "^val_bpb:\|^peak_vram_mb:" run.log\`
6. \`git rev-parse --short HEAD\`
7. \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\`
8. If val_bpb < $BEST_BPB: keep. If not: \`git reset --hard HEAD~1\`

## Rules

- Tools you can use: **Bash**, **Edit**, **Read** — nothing else
- Do NOT use \`>\` or \`>>\` redirection — blocked by sandbox
- Do NOT use \`python train.py\` directly — always use \`bash run_experiment.sh\`
- Do NOT run prepare.py — data is already prepared
- Do NOT run commands in background — run directly and wait
- Do NOT use \`python3 -c\` to edit files — use the Edit tool
- Run git commands one at a time (e.g. \`git add train.py\` then \`git commit -m "msg"\`), not chained with \`&&\`

## Recovery

- \`bash revert_train.sh\` — restore last working train.py
- \`bash run_experiment.sh\` auto-checks syntax before training
RESUMEMD
        fi

        log_event "{\"event\": \"agent_restart\", \"attempt\": $ATTEMPT, \"experiments_before\": $BEFORE_COUNT, \"best_bpb\": \"$BEST_BPB\"}"
        echo "  Experiments so far: $BEFORE_COUNT"
        echo "  Best val_bpb:       $BEST_BPB"
        echo "  Cooling down for ${RESTART_COOLDOWN}s..."
        sleep "$RESTART_COOLDOWN"
    fi

    echo "  Transcript: $TRANSCRIPT"
    echo ""

    # Context compaction watchdog: after EXPERIMENTS_PER_SESSION new
    # experiments, kill the agent so the restart loop launches a fresh
    # session with a clean context window. Polls every 30s.
    EXPERIMENTS_PER_SESSION=${EXPERIMENTS_PER_SESSION:-30}
    WATCHDOG_PID=""
    if [ "$EXPERIMENTS_PER_SESSION" -gt 0 ]; then
        (
            while true; do
                sleep 30
                CURRENT=$(count_experiments)
                NEW=$((CURRENT - BEFORE_COUNT))
                if [ "$NEW" -ge "$EXPERIMENTS_PER_SESSION" ]; then
                    echo ""
                    echo "=== Context compaction: $NEW experiments this session (limit: $EXPERIMENTS_PER_SESSION) ==="
                    echo "Restarting agent for fresh context window..."
                    log_event "{\"event\": \"context_compaction\", \"session_experiments\": $NEW, \"total_experiments\": $CURRENT}"
                    pkill -f "claude -p" 2>/dev/null || true
                    break
                fi
            done
        ) &
        WATCHDOG_PID=$!
    fi

    # Launch Claude Code with stream-json output
    set +e
    claude -p "${PERMISSION_FLAGS[@]}" --verbose --output-format stream-json "$(cat program.md)" \
        2>&1 | python3 stream_formatter.py "$TRANSCRIPT"
    EXIT_CODE=$?
    set -e

    # Clean up watchdog
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true

    AFTER_COUNT=$(count_experiments)
    NEW_EXPERIMENTS=$((AFTER_COUNT - BEFORE_COUNT))

    log_event "{\"event\": \"agent_exit\", \"attempt\": $ATTEMPT, \"exit_code\": $EXIT_CODE, \"new_experiments\": $NEW_EXPERIMENTS, \"total_experiments\": $AFTER_COUNT}"

    echo ""
    echo "=== Agent exited (code $EXIT_CODE, +$NEW_EXPERIMENTS experiments, $AFTER_COUNT total) ==="

    # User pressed Ctrl+C — respect it, don't restart
    if [ "$USER_STOPPED" = true ]; then
        echo "User interrupted — stopping."
        break
    fi

    # Max restarts reached
    if [ "$ATTEMPT" -gt "$MAX_RESTARTS" ]; then
        echo "Max restarts ($MAX_RESTARTS) reached. Stopping."
        log_event "{\"event\": \"agent_max_restarts\", \"attempts\": $ATTEMPT, \"total_experiments\": $AFTER_COUNT}"
        break
    fi

    # Zero-progress detection: if multiple restarts produced no experiments,
    # the model is likely stuck in a degenerate loop (e.g., repetitive
    # thinking, tool confusion). Increase cooldown to avoid burning through
    # restarts instantly.
    TOTAL_EXPERIMENTS=$(count_experiments)
    if [ "$ATTEMPT" -ge 2 ] && [ "$TOTAL_EXPERIMENTS" -eq 0 ]; then
        echo "WARNING: $ATTEMPT attempts with zero experiments completed."
        echo "  The model may be stuck in a degenerate reasoning loop."
        echo "  Consider trying a different model: OLLAMA_MODEL=qwen3.6:27b"
        RESTART_COOLDOWN=30
    fi

    # Agent is told to NEVER STOP, so any exit is premature — restart
    echo "Agent stopped prematurely (should run forever). Restarting..."
    echo "Will restart (attempt $((ATTEMPT + 1))/$((MAX_RESTARTS + 1)))..."
done

echo ""
echo "=== Agent session complete ==="
FINAL_COUNT=$(count_experiments)
echo "  Total experiments logged: $FINAL_COUNT"
echo "  Restart attempts used:    $ATTEMPT/$((MAX_RESTARTS + 1))"
echo "  Transcripts:              logs/transcripts/"
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
    ${GGUF_ARGS[@]+"${GGUF_ARGS[@]}"} \
    -e AUTORESEARCH_CACHE_DIR=/cache/autoresearch \
    -e OLLAMA_MODEL="$OLLAMA_MODEL" \
    -e OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e MAX_RESTARTS="$MAX_RESTARTS" \
    -e RESTART_COOLDOWN="$RESTART_COOLDOWN" \
    -e EXPERIMENTS_PER_SESSION="$EXPERIMENTS_PER_SESSION" \
    -e AGENT_MODE="$AGENT_MODE" \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -w /workspace \
    "$DOCKER_IMAGE" \
    bash -c "$SETUP_SCRIPT"
