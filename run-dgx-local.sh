#!/usr/bin/env bash
# run-dgx-local.sh — Hypothesis generator mode for local models on DGX Spark
#
# Deterministic experiment loop using raw Ollama API calls. No Claude Code,
# no agent framework, no permissions system. The LLM's only job is to propose
# edits — everything else (git, training, logging, keep/revert) is handled
# by this script.
#
# Usage:
#   bash run-dgx-local.sh                              # default (Qwen3.6 27B)
#   OLLAMA_MODEL=gemma4:26b bash run-dgx-local.sh      # use Gemma 4
#   bash run-dgx-local.sh --max-experiments 100         # run 100 experiments
#
# Environment variables:
#   OLLAMA_MODEL      Ollama model tag (default: qwen3.6:27b)
#   SHARD_CACHE_DIR   Host path for persistent training shards (default: ~/.cache/autoresearch)
#   OLLAMA_MODELS     Host path for persistent Ollama model weights (default: ~/.ollama/models)
#   DOCKER_IMAGE      Base Docker image (default: nvcr.io/nvidia/pytorch:25.12-py3)
#   SHM_SIZE          Shared memory size (default: 64gb)
#   OLLAMA_KEEP_ALIVE Model unload delay after last request (default: 0, unload immediately)
#   OLLAMA_GGUF       Host path to a local GGUF file to import instead of pulling
#   OLLAMA_NUM_CTX    Context window for imported GGUF models (default: 32768)
#   INFERENCE_BACKEND ollama (default) or openai — set to openai to use an
#                     external OpenAI-compatible server (llama-server, vLLM,
#                     ds4) instead of in-container Ollama, e.g. for engines
#                     with speculative decoding support
#   INFERENCE_URL     Base URL of the external server as seen from inside the
#                     container (e.g. http://host.docker.internal:8080/v1)

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
INFERENCE_BACKEND="${INFERENCE_BACKEND:-ollama}"
INFERENCE_URL="${INFERENCE_URL:-}"
CONTAINER_NAME="autoresearch-dgx-local-hyp"
MAX_EXPERIMENTS=50

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-experiments)
            MAX_EXPERIMENTS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run-dgx-local.sh [--max-experiments N] [-h|--help]"
            echo ""
            echo "Hypothesis generator mode: deterministic experiment loop with raw Ollama API."
            echo "The LLM proposes edits; everything else is automated. No Claude Code needed."
            echo ""
            echo "Options:"
            echo "  --max-experiments N   Run up to N experiments (default: 50)"
            echo "  -h, --help            Show this help"
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
            echo "  INFERENCE_BACKEND  ollama (default) or openai — use an external"
            echo "                     OpenAI-compatible server instead of in-container Ollama"
            echo "  INFERENCE_URL      External server URL as seen from the container"
            echo "                     (e.g. http://host.docker.internal:8080/v1)"
            echo ""
            echo "External backend example (llama-server with speculative decoding):"
            echo "  INFERENCE_BACKEND=openai INFERENCE_URL=http://host.docker.internal:8080/v1 \\"
            echo "      bash run-dgx-local.sh"
            echo ""
            echo "Tested models:"
            echo "  qwen3.6:27b          ~18GB  Strong code reasoning (default)"
            echo "  gemma4:26b           ~18GB  Strong general + code capability"
            echo "  gemma4:e4b           ~10GB  Good capability, more memory headroom"
            echo "  qwen2.5-coder:14b    ~8GB   Purpose-built for code tasks"
            echo "  qwen3:8b             ~5GB   Lightweight option"
            echo ""
            echo "Compared to run-dgx-agent.sh (full agent mode):"
            echo "  - No Claude Code or Node.js required"
            echo "  - LLM only proposes edits (structured JSON), never runs commands"
            echo "  - Higher reliability: no permission denials, no tool confusion"
            echo "  - Lower flexibility: can't do multi-step reasoning or inspect logs"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Pre-flight checks ---
echo "=== DGX Spark Autoresearch — Hypothesis Generator Mode ==="
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
echo "  Mode:             hypothesis generator (deterministic loop)"
echo "  Backend:          $INFERENCE_BACKEND${INFERENCE_URL:+ ($INFERENCE_URL)}"
echo "  LLM model:        $OLLAMA_MODEL"
[ -n "$OLLAMA_GGUF" ] && echo "  Custom GGUF:      $OLLAMA_GGUF (num_ctx=$OLLAMA_NUM_CTX)"
echo "  Docker image:     $DOCKER_IMAGE"
echo "  Shard cache:      $SHARD_CACHE_DIR"
echo "  Ollama models:    $OLLAMA_MODELS"
echo "  Shared memory:    $SHM_SIZE"
echo "  Ollama keep-alive: $OLLAMA_KEEP_ALIVE"
echo "  Container name:   $CONTAINER_NAME"
echo "  Max experiments:  $MAX_EXPERIMENTS"
echo ""

# --- Build the in-container setup + loop script ---
SETUP_SCRIPT=$(cat <<'INNEREOF'
#!/usr/bin/env bash
set -euo pipefail

# Restore .git ownership to host user on exit
fix_git_ownership() {
    if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
        chown -R "$HOST_UID:$HOST_GID" /workspace/.git 2>/dev/null || true
    fi
}
trap fix_git_ownership EXIT

echo "=== Setting up hypothesis generator environment ==="

# Install dependencies (skipped if using pre-built image)
if command -v ollama &>/dev/null && python -c "import rustbpe" 2>/dev/null; then
    echo "[deps] Pre-built image detected — skipping installs"
else
    echo "[1/3] Installing Python dependencies..."
    pip install -q rustbpe huggingface_hub tiktoken pyarrow requests

    if [ "${INFERENCE_BACKEND:-ollama}" = "ollama" ]; then
        echo "[2/3] Installing Ollama..."
        if ! command -v ollama &>/dev/null; then
            apt-get update -qq && apt-get install -y -qq zstd >/dev/null 2>&1
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    else
        echo "[2/3] External backend ($INFERENCE_BACKEND) — skipping Ollama install"
    fi

    echo "[3/3] No Claude Code needed — hypothesis generator uses raw API calls"
fi

if [ "${INFERENCE_BACKEND:-ollama}" = "ollama" ]; then
    # Start Ollama server
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
        # Custom GGUF import — one-time cost, model store is a persistent mount
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
        echo "[start] Pulling model: $OLLAMA_MODEL ..."
        ollama pull "$OLLAMA_MODEL"
    fi
    BACKEND_DESC="Ollama (http://localhost:11434)"
else
    # External OpenAI-compatible server (llama-server, vLLM, ds4).
    # The server manages its own model loading — we just verify it's up.
    echo "[start] Using external $INFERENCE_BACKEND backend: $INFERENCE_URL"
    if ! curl -s "${INFERENCE_URL%/}/models" &>/dev/null; then
        echo "WARNING: No response from $INFERENCE_URL — proceeding anyway."
        echo "  (From inside the container, a host server is usually"
        echo "   http://host.docker.internal:<port>/v1)"
    fi
    BACKEND_DESC="$INFERENCE_BACKEND ($INFERENCE_URL)"
fi

echo ""
echo "=== Environment ready ==="
echo "  Backend: $BACKEND_DESC"
echo "  Model:   $OLLAMA_MODEL"
echo ""

# Pre-configure git
git config --global --add safe.directory /workspace
git config --global user.email "agent@autoresearch.local"
git config --global user.name "AutoResearch HypGen"

# Symlink cache
ln -sfn /cache/autoresearch /root/.cache/autoresearch 2>/dev/null || true

# Prepare data
echo "=== Preparing training data ==="
python prepare.py --num-shards 10

# Create experiment branch
RUN_TAG=$(date +%b%d | tr '[:upper:]' '[:lower:]')
BRANCH_NAME="autoresearch/hyp-$RUN_TAG"
cd /workspace
if ! git rev-parse --verify "$BRANCH_NAME" &>/dev/null; then
    echo "Creating experiment branch: $BRANCH_NAME"
    git checkout -b "$BRANCH_NAME"
else
    echo "Switching to existing branch: $BRANCH_NAME"
    git checkout "$BRANCH_NAME"
fi

# Initialize results.tsv
if [ ! -f results.tsv ]; then
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\ttimestamp\n' > results.tsv
    echo "Created results.tsv with header"
fi

# Create experiment runner script (identical to run-dgx-agent.sh version)
cat > /workspace/run_experiment.sh << 'RUNEXP'
#!/usr/bin/env bash
# Wrapper: validates train.py, runs it, and captures output to run.log
echo "Checking train.py syntax..."
if ! python -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2> /tmp/syntax_err.txt; then
    echo "=== SYNTAX ERROR in train.py ==="
    cat /tmp/syntax_err.txt
    echo ""
    echo "Fix the error or run: bash revert_train.sh"
    exit 2
fi

echo "Training started — this takes ~5 minutes. Wait for results below."
python train.py > run.log 2>&1 &
TRAIN_PID=$!

SECONDS_ELAPSED=0
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + 30))
    LATEST=$(grep -oP '\d+\.\d+%' run.log 2>/dev/null | tail -1)
    echo "  ... training ${LATEST:-starting} (${SECONDS_ELAPSED}s elapsed)"
done

wait $TRAIN_PID
exit_code=$?

if [ -f run.log ]; then
    tr '\r' '\n' < run.log | grep -v '^$' > run.log.tmp && mv run.log.tmp run.log
fi

echo "=== Experiment finished (exit code: $exit_code) ==="
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:" run.log 2>/dev/null || echo "(no metrics found — check run.log for errors)"

if grep -q "^val_bpb:" run.log 2>/dev/null; then
    cp train.py .train.py.lastgood
fi

exit $exit_code
RUNEXP
chmod +x /workspace/run_experiment.sh

# Create revert script
cat > /workspace/revert_train.sh << 'REVERT'
#!/usr/bin/env bash
if [ -f .train.py.lastgood ]; then
    cp .train.py.lastgood train.py
    echo "Reverted train.py to last working version."
else
    echo "No .train.py.lastgood found. Reverting to git HEAD instead."
    git checkout -- train.py
    echo "Reverted train.py to last committed version."
fi
REVERT
chmod +x /workspace/revert_train.sh

# Create results logger script
cat > /workspace/log_result.sh << 'LOGEXP'
#!/usr/bin/env bash
if [ $# -lt 5 ]; then
    echo "Usage: bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION"
    exit 1
fi
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$TS" >> results.tsv
echo "Logged: $1  val_bpb=$2  mem=$3GB  status=$4  $5"
LOGEXP
chmod +x /workspace/log_result.sh

# Save initial train.py as first backup
cp /workspace/train.py /workspace/.train.py.lastgood

# Event logging helper
mkdir -p logs
log_event() {
    python3 -c "
import json, time, sys
from datetime import datetime, timezone
event = json.loads(sys.argv[1])
event['ts'] = datetime.now(timezone.utc).isoformat()
with open('logs/events.jsonl', 'a') as f:
    f.write(json.dumps(event) + '\n')
" "$1"
}

log_event '{"event": "hypgen_start", "mode": "hypothesis_generator", "model": "'"$OLLAMA_MODEL"'"}'

# =========================================================================
# DETERMINISTIC EXPERIMENT LOOP
# =========================================================================

echo ""
echo "=== Running baseline ==="
bash run_experiment.sh
BASELINE_EXIT=$?

if [ "$BASELINE_EXIT" -ne 0 ]; then
    echo "ERROR: Baseline training failed (exit code $BASELINE_EXIT)"
    echo "Check run.log for errors. Cannot proceed without a baseline."
    exit 1
fi

BASELINE_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
BASELINE_MEM_MB=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
BASELINE_MEM_GB=$(python3 -c "print(f'{float(\"${BASELINE_MEM_MB}\") / 1024:.1f}')")
COMMIT=$(git rev-parse --short HEAD)
bash log_result.sh "$COMMIT" "$BASELINE_BPB" "$BASELINE_MEM_GB" keep baseline
BEST_BPB="$BASELINE_BPB"

log_event "{\"event\": \"hypgen_baseline\", \"val_bpb\": $BASELINE_BPB, \"memory_gb\": $BASELINE_MEM_GB}"

echo ""
echo "=== Baseline complete ==="
echo "  val_bpb:  $BASELINE_BPB"
echo "  memory:   ${BASELINE_MEM_GB} GB"
echo "  Target:   beat $BEST_BPB"
echo ""
echo "=== Starting experiment loop ($MAX_EXPERIMENTS experiments max) ==="
echo ""

EXPERIMENT=0
CONSECUTIVE_FAILURES=0
KEPT=0
DISCARDED=0
ERRORS=0

while [ "$EXPERIMENT" -lt "$MAX_EXPERIMENTS" ]; do
    EXPERIMENT=$((EXPERIMENT + 1))
    echo "--- Experiment $EXPERIMENT/$MAX_EXPERIMENTS (best: $BEST_BPB, kept: $KEPT, discarded: $DISCARDED, errors: $ERRORS) ---"

    # Step 1: Propose edit via Ollama API
    echo "  [propose] Calling $OLLAMA_MODEL..."
    PROPOSAL=$(python3 hypothesis_generator.py propose \
        --model "$OLLAMA_MODEL" \
        --max-results 10 \
        --temperature 0.7 2>/dev/null) || true

    # Check for proposal errors
    if [ -z "$PROPOSAL" ]; then
        echo "  [propose] Empty response from model. Retrying..."
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        ERRORS=$((ERRORS + 1))
        if [ "$CONSECUTIVE_FAILURES" -ge 5 ]; then
            echo "  5 consecutive failures. Stopping."
            break
        fi
        continue
    fi

    HAS_ERROR=$(echo "$PROPOSAL" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if 'error' in d else 'no')" 2>/dev/null || echo "yes")
    if [ "$HAS_ERROR" = "yes" ]; then
        ERROR_MSG=$(echo "$PROPOSAL" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null || echo "parse error")
        echo "  [propose] Failed: $ERROR_MSG"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        ERRORS=$((ERRORS + 1))
        if [ "$CONSECUTIVE_FAILURES" -ge 5 ]; then
            echo "  5 consecutive failures. Stopping."
            break
        fi
        continue
    fi

    DESCRIPTION=$(echo "$PROPOSAL" | python3 -c "import sys,json; print(json.load(sys.stdin)['description'])" 2>/dev/null || echo "unknown edit")
    ELAPSED=$(echo "$PROPOSAL" | python3 -c "import sys,json; print(json.load(sys.stdin).get('_meta',{}).get('elapsed_s','?'))" 2>/dev/null || echo "?")
    echo "  [propose] \"$DESCRIPTION\" (${ELAPSED}s)"

    # Check for baseline/empty edits proposal
    EDIT_COUNT=$(echo "$PROPOSAL" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('edits',[])))" 2>/dev/null || echo "0")
    if [ "$EDIT_COUNT" = "0" ]; then
        echo "  [propose] No edits proposed (model suggested baseline). Skipping."
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        continue
    fi

    # Step 2: Apply edits
    echo "  [apply] Applying $EDIT_COUNT edit(s)..."
    APPLY_OUTPUT=$(python3 hypothesis_generator.py apply --edits-json "$PROPOSAL" --train-py train.py 2>&1)
    APPLY_EXIT=$?
    if [ "$APPLY_EXIT" -ne 0 ]; then
        echo "  [apply] Failed: $APPLY_OUTPUT"
        git checkout -- train.py
        bash log_result.sh "-------" "0.000000" "0.0" crash "edit failed: $DESCRIPTION"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        ERRORS=$((ERRORS + 1))
        log_event "{\"event\": \"hypgen_edit_failed\", \"experiment\": $EXPERIMENT, \"description\": \"$DESCRIPTION\"}"
        continue
    fi
    echo "  [apply] $APPLY_OUTPUT"

    # Step 3: Syntax check (belt-and-suspenders — run_experiment.sh also checks)
    if ! python -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2>/dev/null; then
        echo "  [syntax] FAILED — reverting"
        git checkout -- train.py
        bash log_result.sh "-------" "0.000000" "0.0" crash "syntax error: $DESCRIPTION"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        ERRORS=$((ERRORS + 1))
        log_event "{\"event\": \"hypgen_syntax_error\", \"experiment\": $EXPERIMENT, \"description\": \"$DESCRIPTION\"}"
        continue
    fi

    # Step 4: Git commit
    git add train.py
    git commit -q -m "$DESCRIPTION"
    COMMIT=$(git rev-parse --short HEAD)
    echo "  [git] Committed: $COMMIT"

    # Step 5: Train
    echo "  [train] Starting training (~5 min)..."
    bash run_experiment.sh
    TRAIN_EXIT=$?

    # Step 6: Parse results
    VAL_BPB=$(grep "^val_bpb:" run.log 2>/dev/null | awk '{print $2}')
    if [ -z "$VAL_BPB" ] || [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "  [train] Failed (exit $TRAIN_EXIT) — reverting"
        git reset --hard HEAD~1
        bash log_result.sh "$COMMIT" "0.000000" "0.0" crash "$DESCRIPTION"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        ERRORS=$((ERRORS + 1))
        log_event "{\"event\": \"hypgen_train_crash\", \"experiment\": $EXPERIMENT, \"description\": \"$DESCRIPTION\"}"
        continue
    fi

    PEAK_MEM_MB=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
    MEM_GB=$(python3 -c "print(f'{float(\"${PEAK_MEM_MB}\") / 1024:.1f}')")
    CONSECUTIVE_FAILURES=0

    # Step 7: Keep or revert
    IMPROVED=$(python3 -c "print('yes' if float('$VAL_BPB') < float('$BEST_BPB') else 'no')")
    if [ "$IMPROVED" = "yes" ]; then
        STATUS="keep"
        PREV_BEST="$BEST_BPB"
        BEST_BPB="$VAL_BPB"
        cp train.py .train.py.lastgood
        KEPT=$((KEPT + 1))
        echo "  [result] IMPROVED: val_bpb $VAL_BPB (was $PREV_BEST) — keeping"
        log_event "{\"event\": \"hypgen_improved\", \"experiment\": $EXPERIMENT, \"val_bpb\": $VAL_BPB, \"prev_best\": $PREV_BEST, \"description\": \"$DESCRIPTION\"}"
    else
        STATUS="discard"
        git reset --hard HEAD~1
        DISCARDED=$((DISCARDED + 1))
        echo "  [result] No improvement: val_bpb $VAL_BPB >= $BEST_BPB — reverting"
        log_event "{\"event\": \"hypgen_discarded\", \"experiment\": $EXPERIMENT, \"val_bpb\": $VAL_BPB, \"best_bpb\": $BEST_BPB, \"description\": \"$DESCRIPTION\"}"
    fi
    bash log_result.sh "$COMMIT" "$VAL_BPB" "$MEM_GB" "$STATUS" "$DESCRIPTION"

    echo ""
done

# =========================================================================
# SUMMARY
# =========================================================================

echo ""
echo "=== Experiment loop complete ==="
TOTAL=$((KEPT + DISCARDED + ERRORS))
echo "  Experiments:  $TOTAL / $MAX_EXPERIMENTS"
echo "  Kept:         $KEPT"
echo "  Discarded:    $DISCARDED"
echo "  Errors:       $ERRORS"
echo "  Best val_bpb: $BEST_BPB (baseline: $BASELINE_BPB)"
if [ "$KEPT" -gt 0 ]; then
    IMPROVEMENT=$(python3 -c "print(f'{(float(\"$BASELINE_BPB\") - float(\"$BEST_BPB\")):.6f}')")
    echo "  Improvement:  $IMPROVEMENT (${KEPT} successful edits)"
fi
echo ""
echo "  Results:      results.tsv"
echo "  Event log:    logs/events.jsonl"

log_event "{\"event\": \"hypgen_complete\", \"total\": $TOTAL, \"kept\": $KEPT, \"discarded\": $DISCARDED, \"errors\": $ERRORS, \"best_bpb\": $BEST_BPB, \"baseline_bpb\": $BASELINE_BPB}"
INNEREOF
)

# --- Launch container ---
echo "Starting hypothesis generator container..."
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
    --add-host=host.docker.internal:host-gateway \
    -e INFERENCE_BACKEND="$INFERENCE_BACKEND" \
    -e INFERENCE_URL="$INFERENCE_URL" \
    -e AUTORESEARCH_CACHE_DIR=/cache/autoresearch \
    -e OLLAMA_MODEL="$OLLAMA_MODEL" \
    -e OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e MAX_EXPERIMENTS="$MAX_EXPERIMENTS" \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -w /workspace \
    "$DOCKER_IMAGE" \
    bash -c "$SETUP_SCRIPT"
