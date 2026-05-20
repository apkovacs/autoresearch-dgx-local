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
CONTAINER_NAME="autoresearch-dgx-local-agent"
MAX_RESTARTS=3
RESTART_COOLDOWN=10
EXPERIMENTS_PER_SESSION=30

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-restarts)
            MAX_RESTARTS="$2"; shift 2 ;;
        --no-restart)
            MAX_RESTARTS=0; shift ;;
        --experiments-per-session)
            EXPERIMENTS_PER_SESSION="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run-dgx-agent.sh [--max-restarts N] [--no-restart] [--experiments-per-session N] [-h|--help]"
            echo ""
            echo "Launches the autonomous autoresearch agent with a local LLM."
            echo ""
            echo "Options:"
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

# Start Ollama server
echo "[start] Starting Ollama server..."
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

# Pull the model (skips if already cached via volume mount)
echo "[start] Pulling model: $OLLAMA_MODEL ..."
ollama pull "$OLLAMA_MODEL"

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
# - Saves .train.py.lastgood after each successful run
# Usage: bash run_experiment.sh

# Step 1: Syntax check — fail fast on bad edits
echo "Checking train.py syntax..."
if ! python -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2> /tmp/syntax_err.txt; then
    echo "=== SYNTAX ERROR in train.py ==="
    cat /tmp/syntax_err.txt
    echo ""
    echo "Fix the error or run: bash revert_train.sh"
    exit 2
fi

# Step 2: Run training
python train.py > run.log 2>&1
exit_code=$?
echo "=== Experiment finished (exit code: $exit_code) ==="
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:" run.log 2>/dev/null || echo "(no metrics found — check run.log for errors)"

# Step 3: Save last-good backup if training produced results
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
# This overrides the Setup section of program.md
cat > /workspace/CLAUDE.md << CLAUDEMD
# Environment Notes — READ THIS FIRST

## Setup is ALREADY DONE — skip to experimenting
- Branch: $BRANCH_NAME (already checked out)
- Data: pre-downloaded at /cache/autoresearch (symlinked to ~/.cache/autoresearch)
- Tokenizer: pre-trained
- results.tsv: created with header row

Do NOT run the Setup section of program.md. Go directly to the Experimentation loop.

## IMPORTANT: Rules
- Only use these tools: **Bash**, **Edit**, **Read**. Do NOT use Task, Monitor, TaskCreate, Agent, or any other tools.
- Use \`bash run_experiment.sh\` to run experiments (NOT \`python train.py > run.log 2>&1\`)
- Use \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\` to log results (timestamp added automatically)
- Do NOT use output redirection (\`>\` or \`>>\`) in bash commands — it is blocked by the sandbox.
- Do NOT run experiments in the background. Run them directly with Bash and wait for completion.
- Data is already prepared — do NOT run prepare.py

## Available commands
Wrapper scripts (use these, not raw commands):
- \`bash run_experiment.sh\` — syntax-check + train + save backup
- \`bash log_result.sh COMMIT BPB MEM STATUS DESC\` — log to results.tsv
- \`bash revert_train.sh\` — restore last working train.py

Git (only these forms work):
- \`git add train.py && git commit -m "msg"\` — commit changes
- \`git diff train.py\` — see uncommitted changes
- \`git log --oneline -5\` — recent commits (MUST use --oneline)
- \`git reset --hard HEAD~1\` — revert last commit
- \`git rev-parse --short HEAD\` — get commit hash
- \`git status\` — check working tree

Reading files:
- \`grep "^val_bpb:\|^peak_vram_mb:" run.log\` — get experiment results
- \`tail -n 50 run.log\` — read error output from failed runs

## What you can modify
Only train.py — this is the single file you should edit.
Use the Edit tool to modify train.py. Do NOT use python3 -c to rewrite files.

## Safety nets
- \`bash run_experiment.sh\` **automatically syntax-checks** train.py before running. If there is a syntax error, it will tell you immediately (exit code 2) without wasting training time.
- If you break train.py, run \`bash revert_train.sh\` to restore the last working version.
- After each successful training run, train.py is backed up automatically.

## Experiment loop — follow this EXACTLY

For EVERY experiment (including the baseline), do ALL of these steps:

1. Edit train.py with your experimental idea (skip for baseline)
2. \`git add train.py && git commit -m "description of change"\` (skip for baseline)
3. \`bash run_experiment.sh\`
4. If exit code is 2 (syntax error): run \`bash revert_train.sh\`, then fix your edit and retry
5. \`grep "^val_bpb:\|^peak_vram_mb:" run.log\`
6. Get the commit hash: \`git rev-parse --short HEAD\`
7. **Log the result to results.tsv NOW** — run this command:
   \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\`
   Example: \`bash log_result.sh a1b2c3d 1.879972 7.6 keep baseline\`
8. If val_bpb IMPROVED (lower): keep the commit, move on
9. If val_bpb did NOT improve: \`git reset --hard HEAD~1\` to revert

**AFTER EVERY EXPERIMENT you MUST run \`bash log_result.sh\` (step 7) to log to results.tsv.**
**To revert failed experiments, MUST use \`git reset --hard HEAD~1\`.**
**If train.py is broken, run \`bash revert_train.sh\` to restore the last working version.**
Do not manually undo code changes. Do not use python3 -c to edit files.

## Start now
1. Read train.py
2. Run the baseline: \`bash run_experiment.sh\`
3. Log the baseline to results.tsv (step 7 above)
4. Begin experimenting
CLAUDEMD

echo ""
echo "=== Launching autonomous agent ==="
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

log_event '{"event": "orchestrator_start", "mode": "base", "tag": "agent", "config": "run-dgx-agent.sh"}'

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
        cat > /workspace/CLAUDE.md << RESUMEMD
# Environment Notes — READ THIS FIRST

## Current state
- Baseline val_bpb: $BASELINE_BPB
- Best val_bpb so far: $BEST_BPB (this is the number to beat)
- Experiments completed: $BEFORE_COUNT
- train.py reflects the best version so far — read it and improve from here

$([ "$HAS_BASELINE" = false ] && echo "The baseline has NOT been run yet. Run it first: \`bash run_experiment.sh\`")
Do NOT re-run the baseline if it has already been run (check above).

## IMPORTANT: Rules
- Only use these tools: **Bash**, **Edit**, **Read**. Do NOT use Task, Monitor, TaskCreate, Agent, or any other tools.
- Do NOT use output redirection (\`>\` or \`>>\`) in bash commands — it is blocked by the sandbox.
- Do NOT run experiments in the background. Run them directly with Bash and wait for completion.
- Data is already prepared — do NOT run prepare.py

## Available commands
Wrapper scripts (use these, not raw commands):
- \`bash run_experiment.sh\` — syntax-check + train + save backup
- \`bash log_result.sh COMMIT BPB MEM STATUS DESC\` — log to results.tsv
- \`bash revert_train.sh\` — restore last working train.py

Git (only these forms work):
- \`git add train.py && git commit -m "msg"\` — commit changes
- \`git diff train.py\` — see uncommitted changes
- \`git log --oneline -5\` — recent commits (MUST use --oneline)
- \`git reset --hard HEAD~1\` — revert last commit
- \`git rev-parse --short HEAD\` — get commit hash
- \`git status\` — check working tree

Reading files:
- \`grep "^val_bpb:\|^peak_vram_mb:" run.log\` — get experiment results
- \`tail -n 50 run.log\` — read error output from failed runs

## What you can modify
Only train.py — this is the single file you should edit.
Use the Edit tool to modify train.py. Do NOT use python3 -c to rewrite files.

## Safety nets
- \`bash run_experiment.sh\` **automatically syntax-checks** train.py before running. If there is a syntax error, it will tell you immediately (exit code 2) without wasting training time.
- If you break train.py, run \`bash revert_train.sh\` to restore the last working version.
- After each successful training run, train.py is backed up automatically.

## Experiment loop — follow this EXACTLY

1. Read train.py — understand the current architecture and hyperparameters
2. Edit train.py with your experimental idea
3. \`git add train.py && git commit -m "description of change"\`
4. \`bash run_experiment.sh\`
5. If exit code is 2 (syntax error): run \`bash revert_train.sh\`, then fix your edit and retry
6. \`grep "^val_bpb:\|^peak_vram_mb:" run.log\`
7. Get the commit hash: \`git rev-parse --short HEAD\`
8. **Log the result**: \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\`
9. If val_bpb IMPROVED (lower than $BEST_BPB): keep the commit, move on
10. If val_bpb did NOT improve: \`git reset --hard HEAD~1\` to revert

**AFTER EVERY EXPERIMENT you MUST run \`bash log_result.sh\` (step 8) to log to results.tsv.**
**To revert failed experiments, MUST use \`git reset --hard HEAD~1\`.**
**If train.py is broken, run \`bash revert_train.sh\` to restore the last working version.**

## Start now
1. Read train.py
2. Begin experimenting — beat val_bpb $BEST_BPB
RESUMEMD

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
    claude -p --permission-mode dontAsk --verbose --output-format stream-json "$(cat program.md)" \
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
    -e AUTORESEARCH_CACHE_DIR=/cache/autoresearch \
    -e OLLAMA_MODEL="$OLLAMA_MODEL" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e MAX_RESTARTS="$MAX_RESTARTS" \
    -e RESTART_COOLDOWN="$RESTART_COOLDOWN" \
    -e EXPERIMENTS_PER_SESSION="$EXPERIMENTS_PER_SESSION" \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -w /workspace \
    "$DOCKER_IMAGE" \
    bash -c "$SETUP_SCRIPT"
