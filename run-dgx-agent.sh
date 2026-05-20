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
      "Edit(/workspace/train.py)",
      "Read(/workspace/*)",
      "Read(/cache/autoresearch/*)",
      "Write(/workspace/results.tsv)",
      "Write(results.tsv)",
      "Write(/workspace/run.log)",
      "Write(run.log)",
      "Bash(ls /cache/*)",
      "Bash(bash run_experiment.sh*)",
      "Bash(bash log_result.sh*)",
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
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > results.tsv
    echo "Created results.tsv with header"
fi

# Create experiment runner script (handles output redirect so the agent doesn't need to)
cat > /workspace/run_experiment.sh << 'RUNEXP'
#!/usr/bin/env bash
# Wrapper: runs train.py and captures output to run.log
# Usage: bash run_experiment.sh
python train.py > run.log 2>&1
exit_code=$?
echo "=== Experiment finished (exit code: $exit_code) ==="
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:" run.log 2>/dev/null || echo "(no metrics found — check run.log for errors)"
exit $exit_code
RUNEXP
chmod +x /workspace/run_experiment.sh

# Create results logger script (>> redirect is blocked by Claude Code sandbox)
cat > /workspace/log_result.sh << 'LOGEXP'
#!/usr/bin/env bash
# Wrapper: appends experiment result to results.tsv
# Usage: bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION
if [ $# -lt 5 ]; then
    echo "Usage: bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION"
    echo "Example: bash log_result.sh a1b2c3d 1.879972 7.6 keep baseline"
    exit 1
fi
printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" >> results.tsv
echo "Logged to results.tsv: $1  val_bpb=$2  mem=$3GB  status=$4  $5"
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

## IMPORTANT: Command differences from program.md
- Use \`bash run_experiment.sh\` to run experiments (NOT \`python train.py > run.log 2>&1\`)
- Use \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\` to log results
- Do NOT use output redirection (\`>\` or \`>>\`) in bash commands — it is blocked by the sandbox.
- Use \`python train.py\` instead of \`uv run train.py\` (there is no uv in this environment)
- Data is already prepared — do NOT run prepare.py

## What you can modify
Only train.py — this is the single file you should edit.
Use the Edit tool to modify train.py. Do NOT use python3 -c to rewrite files.

## Experiment loop — follow this EXACTLY

For EVERY experiment (including the baseline), do ALL of these steps:

1. Edit train.py with your experimental idea (skip for baseline)
2. \`git add train.py && git commit -m "description of change"\` (skip for baseline)
3. \`bash run_experiment.sh\`
4. \`grep "^val_bpb:\|^peak_vram_mb:" run.log\`
5. Get the commit hash: \`git rev-parse --short HEAD\`
6. **Log the result to results.tsv NOW** — run this command:
   \`bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION\`
   Example: \`bash log_result.sh a1b2c3d 1.879972 7.6 keep baseline\`
7. If val_bpb IMPROVED (lower): keep the commit, move on
8. If val_bpb did NOT improve: \`git reset --hard HEAD~1\` to revert

**AFTER EVERY EXPERIMENT you MUST run \`bash log_result.sh\` (step 6) to log to results.tsv.**
**To revert failed experiments, MUST use \`git reset --hard HEAD~1\`.**
Do not manually undo code changes. Do not use python3 -c to edit files.

## Start now
1. Read train.py
2. Run the baseline: \`bash run_experiment.sh\`
3. Log the baseline to results.tsv (step 6 above)
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

# Launch Claude Code with stream-json output
# stream_formatter.py saves full JSON to the transcript file while
# printing formatted output to the terminal (agent activity + training progress)
claude -p --verbose --output-format stream-json "$(cat program.md)" \
    2>&1 | python3 stream_formatter.py logs/transcripts/agent.jsonl
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
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -w /workspace \
    "$DOCKER_IMAGE" \
    bash -c "$SETUP_SCRIPT"
