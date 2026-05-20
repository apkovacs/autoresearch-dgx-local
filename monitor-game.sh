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
CONTAINER_NAME="autoresearch-dgx-game"
EVENT_LOG="logs/events.jsonl"
TRANSCRIPT_DIR="logs/transcripts"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --events) MODE="events"; shift ;;
        --transcript) MODE="transcript"; shift ;;
        --transcript-raw) MODE="transcript-raw"; shift ;;
        -h|--help)
            echo "Usage: bash monitor-game.sh [--events|--transcript|--transcript-raw] [-h]"
            echo ""
            echo "Modes:"
            echo "  (default)         Live dashboard with leaderboard and recent events"
            echo "  --events          Tail the event log (formatted)"
            echo "  --transcript      Tail the latest agent transcript (formatted)"
            echo "  --transcript-raw  Tail the latest agent transcript (raw stream-json)"
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
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^autoresearch-dgx-agent$"; then
    RUN_PREFIX="docker exec autoresearch-dgx-agent"
    DETECTED_CONTAINER="autoresearch-dgx-agent"
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^autoresearch-dgx$"; then
    RUN_PREFIX="docker exec autoresearch-dgx"
    DETECTED_CONTAINER="autoresearch-dgx"
else
    echo "ERROR: No event log found and no autoresearch container is running."
    echo ""
    echo "Start one of:"
    echo "  bash run-dgx-game.sh --mode <mode>   (game orchestrator)"
    echo "  bash run-dgx-agent.sh                (single-branch agent)"
    echo ""
    echo "Note: The single-branch agent (run-dgx-agent.sh) does not produce"
    echo "an event log. Use 'docker logs -f autoresearch-dgx-agent' to monitor it,"
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
    # Shows compact progress: step count, loss, val_bpb, tokens/sec
    TRAIN_LOG_PID=""
    start_training_tail() {
        local log_path="$1"
        (run_cmd "tail -f $log_path 2>/dev/null" | \
            grep --line-buffered -E "^step |^val_bpb:|^training_seconds:|compil" | \
            while IFS= read -r line; do
                # Compact: show step lines as progress, key metrics in full
                case "$line" in
                    step*)
                        # Extract step number and show compact progress
                        step_num=$(echo "$line" | grep -oP 'step \K\d+')
                        loss=$(echo "$line" | grep -oP 'loss \K[0-9.]+')
                        tps=$(echo "$line" | grep -oP '[0-9.]+ tokens/sec' | head -1)
                        printf "\r\033[2m  [training] step %-5s loss %-8s %s\033[0m" \
                            "$step_num" "$loss" "$tps"
                        ;;
                    val_bpb*)
                        printf "\n\033[32;1m  [result] %s\033[0m\n" "$line"
                        ;;
                    training_seconds*)
                        printf "  [result] %s\n" "$line"
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
    echo "║              autoresearch-dgx Game Monitor                         ║"
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
    echo "    bash monitor-game.sh --events          orchestrator event stream"
    echo "    bash monitor-game.sh --transcript       agent thinking + tool calls + responses"
    echo "    bash monitor-game.sh --transcript-raw   raw stream-json from agent"
}

while true; do
    draw_dashboard
    sleep 5
done
