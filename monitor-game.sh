#!/usr/bin/env bash
# monitor-game.sh — Live dashboard for the game orchestrator
#
# Shows current state, leaderboard, recent events, and optionally
# tails an agent transcript with formatted thinking/tool/response output.
#
# Usage:
#   bash monitor-game.sh                  # dashboard mode (default)
#   bash monitor-game.sh --events         # tail formatted event log
#   bash monitor-game.sh --transcript     # tail latest agent transcript (formatted)
#   bash monitor-game.sh --transcript-raw # tail latest transcript (raw JSON)

set -euo pipefail

MODE="dashboard"
CONTAINER_NAME="autoresearch-dgx-local-game"
EVENT_LOG="logs/events.jsonl"
TRANSCRIPT_DIR="logs/transcripts"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --events) MODE="events"; shift ;;
        --transcript) MODE="transcript"; shift ;;
        --transcript-raw) MODE="transcript-raw"; shift ;;
        --status) MODE="status"; shift ;;
        -h|--help)
            echo "Usage: bash monitor-game.sh [--events|--transcript|--transcript-raw|--status] [-h]"
            echo ""
            echo "Modes:"
            echo "  (default)         Live dashboard with leaderboard and recent events"
            echo "  --events          Tail the event log (formatted)"
            echo "  --transcript      Tail the latest agent transcript (formatted)"
            echo "  --transcript-raw  Tail the latest agent transcript (raw stream-json)"
            echo "  --status          One-shot snapshot: experiments, GPU, agent activity"
            echo ""
            echo "The dashboard reads from logs/ inside the container."
            echo "Since the workspace is volume-mounted, you can also read"
            echo "these files directly from the host."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Check if running in container or on host ---
DETECTED_CONTAINER=""
if [ -f "$EVENT_LOG" ]; then
    # Event log exists locally (inside container or host with volume mount)
    RUN_PREFIX=""
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    RUN_PREFIX="docker exec $CONTAINER_NAME"
    DETECTED_CONTAINER="$CONTAINER_NAME"
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^autoresearch-dgx-local-agent$"; then
    RUN_PREFIX="docker exec autoresearch-dgx-local-agent"
    DETECTED_CONTAINER="autoresearch-dgx-local-agent"
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^autoresearch-dgx-local$"; then
    RUN_PREFIX="docker exec autoresearch-dgx-local"
    DETECTED_CONTAINER="autoresearch-dgx-local"
else
    echo "ERROR: No event log found and no autoresearch container is running."
    echo ""
    echo "Start one of:"
    echo "  bash run-dgx-game.sh --mode <mode>   (game orchestrator)"
    echo "  bash run-dgx-agent.sh                (single-branch agent)"
    echo ""
    echo "Note: The single-branch agent (run-dgx-agent.sh) does not produce"
    echo "an event log. Use 'docker logs -f autoresearch-dgx-local-agent' to monitor it,"
    echo "or switch to run-dgx-game.sh --mode base for full observability."
    exit 1
fi

run_cmd() {
    if [ -z "$RUN_PREFIX" ]; then
        eval "$@"
    else
        $RUN_PREFIX bash -c "$*"
    fi
}

# --- Locate helper scripts ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# stream_formatter.py (for --transcript mode)
if [ -f "$SCRIPT_DIR/stream_formatter.py" ]; then
    FORMATTER="$SCRIPT_DIR/stream_formatter.py"
elif [ -f "/workspace/stream_formatter.py" ]; then
    FORMATTER="/workspace/stream_formatter.py"
elif [ -f "stream_formatter.py" ]; then
    FORMATTER="stream_formatter.py"
else
    FORMATTER=""
fi

# dashboard_renderer.py (for dashboard mode)
if [ -f "$SCRIPT_DIR/dashboard_renderer.py" ]; then
    DASHBOARD_RENDERER="$SCRIPT_DIR/dashboard_renderer.py"
elif [ -f "/workspace/dashboard_renderer.py" ]; then
    DASHBOARD_RENDERER="/workspace/dashboard_renderer.py"
elif [ -f "dashboard_renderer.py" ]; then
    DASHBOARD_RENDERER="dashboard_renderer.py"
else
    DASHBOARD_RENDERER=""
fi

# --- Find latest transcript ---
find_latest_transcript() {
    run_cmd "ls -t $TRANSCRIPT_DIR/*.jsonl 2>/dev/null | head -1" 2>/dev/null || true
}

wait_for_transcript() {
    echo "No transcripts found yet. Waiting..."
    local latest=""
    while true; do
        latest=$(find_latest_transcript)
        [ -n "$latest" ] && echo "$latest" && return
        sleep 5
    done
}

# --- Mode: status snapshot ---
if [ "$MODE" = "status" ]; then
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║              autoresearch-dgx-local Status Snapshot                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Experiments
    echo "  ── Experiments ──"
    if run_cmd "test -f results.tsv" 2>/dev/null; then
        TOTAL=$(run_cmd "wc -l < results.tsv" 2>/dev/null | tr -d ' ')
        TOTAL=$((TOTAL - 1))  # subtract header
        if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
            echo "  Completed: $TOTAL"
            echo ""
            run_cmd "cat results.tsv" 2>/dev/null | column -t -s$'\t' | while IFS= read -r line; do
                echo "    $line"
            done
        else
            echo "  No experiments completed yet."
        fi
    else
        echo "  results.tsv not found."
    fi
    echo ""

    # Latest result / active training
    echo "  ── Training ──"
    if run_cmd "pgrep -af 'python train.py'" 2>/dev/null | grep -q train; then
        echo "  Status: RUNNING"
        # train.py uses \r for in-place updates, so convert to \n before grepping
        LAST_STEP=$(run_cmd "tr '\r' '\n' < run.log 2>/dev/null | grep 'step [0-9]' | tail -1" 2>/dev/null)
        if [ -n "$LAST_STEP" ]; then
            # Trim extra whitespace
            LAST_STEP=$(echo "$LAST_STEP" | sed 's/  */ /g')
            echo "  $LAST_STEP"
        fi
    else
        LAST_BPB=$(run_cmd "grep 'val_bpb:' run.log 2>/dev/null | tail -1" 2>/dev/null)
        if [ -n "$LAST_BPB" ]; then
            echo "  Status: idle (last run finished)"
            echo "  $LAST_BPB"
        else
            echo "  Status: idle"
        fi
    fi
    echo ""

    # GPU
    echo "  ── GPU ──"
    run_cmd "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null | \
        while IFS=', ' read -r idx util mem_used mem_total temp; do
            pct=$((mem_used * 100 / mem_total))
            echo "  GPU $idx: ${util}% util | ${mem_used}/${mem_total} MiB (${pct}%) | ${temp}°C"
        done || echo "  (nvidia-smi not available)"
    echo ""

    # Git state
    echo "  ── Git ──"
    BRANCH=$(run_cmd "git -C /workspace rev-parse --abbrev-ref HEAD" 2>/dev/null)
    [ -n "$BRANCH" ] && echo "  Branch: $BRANCH"
    echo "  Recent commits:"
    run_cmd "git -C /workspace log --oneline -5" 2>/dev/null | while IFS= read -r line; do
        echo "    $line"
    done
    echo ""

    # Agent process
    echo "  ── Agent ──"
    if run_cmd "pgrep -af claude" 2>/dev/null | grep -q claude; then
        echo "  Claude Code: running"
    else
        echo "  Claude Code: not running"
    fi

    # Transcript info
    LATEST_TRANSCRIPT=$(find_latest_transcript)
    if [ -n "$LATEST_TRANSCRIPT" ]; then
        LINES=$(run_cmd "wc -l < $LATEST_TRANSCRIPT" 2>/dev/null | tr -d ' ')
        echo "  Transcript: $LATEST_TRANSCRIPT ($LINES events)"
    fi

    exit 0
fi

# --- Mode: raw event tail ---
if [ "$MODE" = "events" ]; then
    echo "=== Tailing event log (Ctrl+C to stop) ==="
    run_cmd "tail -f $EVENT_LOG" | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        ts = e['ts'][11:19]
        evt = e['event']
        if evt == 'branch_progress':
            print(f'  {ts}  {evt:<20} {e[\"branch\"]:>12}  {e[\"completed\"]}/{e[\"target\"]}  [{e[\"elapsed_s\"]:.0f}s]')
        elif evt == 'branch_start':
            print(f'  {ts}  {evt:<20} {e[\"branch\"]:>12}  target={e[\"target_experiments\"]}  focus={e.get(\"focus\",\"\")}')
        elif evt == 'branch_end':
            print(f'  {ts}  {evt:<20} {e[\"branch\"]:>12}  completed={e[\"completed\"]}  [{e[\"duration_s\"]:.0f}s]')
        elif evt == 'round_start':
            print(f'\\n  {ts}  {evt:<20} round={e[\"round\"]}')
        elif evt == 'round_end':
            print(f'  {ts}  {evt:<20} round={e[\"round\"]}  [{e[\"duration_s\"]:.0f}s]')
        elif evt == 'migration':
            print(f'  {ts}  {evt:<20} {e[\"source\"]} -> {e[\"dest\"]}  params={e[\"params\"]}')
        elif evt == 'adoption':
            print(f'  {ts}  {evt:<20} {e[\"loser\"]} adopts from {e[\"winner\"]}  params={e[\"params\"]}')
        elif evt == 'leaderboard':
            best = e.get('global_best_bpb')
            best_b = e.get('global_best_branch', '?')
            print(f'  {ts}  {evt:<20} best={best}  ({best_b})')
        elif evt == 'error':
            print(f'  {ts}  ERROR: {e.get(\"message\",\"\")}')
        else:
            print(f'  {ts}  {evt:<20} {json.dumps({k:v for k,v in e.items() if k not in (\"ts\",\"elapsed_s\",\"event\")}, default=str)}')
    except (json.JSONDecodeError, KeyError):
        print(line.rstrip())
" 2>/dev/null || run_cmd "tail -f $EVENT_LOG"
    exit 0
fi

# --- Mode: formatted transcript ---
if [ "$MODE" = "transcript" ]; then
    echo "=== Tailing latest agent transcript (Ctrl+C to stop) ==="
    LATEST=$(find_latest_transcript)
    [ -z "$LATEST" ] && LATEST=$(wait_for_transcript)
    echo "Following: $LATEST"
    echo "  Showing: thinking | tool calls | agent responses | training progress"
    echo "---"

    # Also tail run.log for live training progress (runs in background)
    # train.py uses \r for in-place step updates, so we pipe through
    # tr to convert \r to \n, then filter for progress lines.
    TRAIN_LOG_PID=""
    start_training_tail() {
        local log_path="$1"
        (run_cmd "tail -f $log_path 2>/dev/null" | tr '\r' '\n' | \
            grep --line-buffered -E "step [0-9]|val_bpb:|training_seconds:|compil" | \
            while IFS= read -r line; do
                case "$line" in
                    *val_bpb*)
                        printf "\n\033[32;1m  [result] %s\033[0m\n" "$line"
                        ;;
                    *training_seconds*)
                        printf "  [result] %s\n" "$line"
                        ;;
                    *step\ [0-9]*)
                        # Compact: show the step line trimmed
                        clean=$(echo "$line" | sed 's/  */ /g' | cut -c1-80)
                        printf "\r\033[2m  [training] %s\033[0m" "$clean"
                        ;;
                    *ompil*)
                        printf "\r\033[2m  [training] compiling...\033[0m"
                        ;;
                esac
            done) &
        TRAIN_LOG_PID=$!
    }

    # Determine run.log path
    if [ -z "$RUN_PREFIX" ]; then
        RUN_LOG="run.log"
    else
        RUN_LOG="/workspace/run.log"
    fi
    start_training_tail "$RUN_LOG"

    # Clean up background tail on exit
    trap '[ -n "$TRAIN_LOG_PID" ] && kill $TRAIN_LOG_PID 2>/dev/null; exit' INT TERM EXIT

    if [ -n "$FORMATTER" ]; then
        run_cmd "tail -f $LATEST" | python3 "$FORMATTER"
    else
        echo "(stream_formatter.py not found — showing raw JSON)"
        run_cmd "tail -f $LATEST"
    fi
    exit 0
fi

# --- Mode: raw transcript ---
if [ "$MODE" = "transcript-raw" ]; then
    echo "=== Tailing latest agent transcript - raw (Ctrl+C to stop) ==="
    LATEST=$(find_latest_transcript)
    [ -z "$LATEST" ] && LATEST=$(wait_for_transcript)
    echo "Following: $LATEST"
    echo "---"
    run_cmd "tail -f $LATEST"
    exit 0
fi

# --- Mode: dashboard ---
draw_dashboard() {
    clear

    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║              autoresearch-dgx-local Game Monitor                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    if run_cmd "test -f $EVENT_LOG" 2>/dev/null; then
        if [ -n "$DASHBOARD_RENDERER" ]; then
            STATE=$(run_cmd "python3 $DASHBOARD_RENDERER $EVENT_LOG" 2>/dev/null || echo "  (waiting for events...)")
        else
            STATE="  (dashboard_renderer.py not found)"
        fi
        echo "$STATE"
    else
        echo "  Waiting for orchestrator to start..."
    fi

    # Latest transcript
    LATEST_TRANSCRIPT=$(run_cmd "ls -t $TRANSCRIPT_DIR/*.jsonl 2>/dev/null | head -1" 2>/dev/null || true)
    if [ -n "$LATEST_TRANSCRIPT" ]; then
        echo ""
        echo "  Latest transcript: $LATEST_TRANSCRIPT"
    fi

    echo ""
    echo "  Refreshing every 5s. Ctrl+C to exit."
    echo ""
    echo "  Other views:"
    echo "    bash monitor-game.sh --status            one-shot snapshot (experiments, GPU, git)"
    echo "    bash monitor-game.sh --events            orchestrator event stream"
    echo "    bash monitor-game.sh --transcript         agent thinking + tool calls + training"
    echo "    bash monitor-game.sh --transcript-raw     raw stream-json from agent"
}

while true; do
    draw_dashboard
    sleep 5
done
