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

# --- Locate stream_formatter.py ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/stream_formatter.py" ]; then
    FORMATTER="$SCRIPT_DIR/stream_formatter.py"
elif [ -f "/workspace/stream_formatter.py" ]; then
    FORMATTER="/workspace/stream_formatter.py"
elif [ -f "stream_formatter.py" ]; then
    FORMATTER="stream_formatter.py"
else
    FORMATTER=""
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
    echo "  Showing: thinking | tool calls | agent responses"
    echo "---"
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
        STATE=$(run_cmd "python3 -c \"
import json, sys

events = []
with open('$EVENT_LOG') as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass

if not events:
    print('  No events yet.')
    sys.exit(0)

current_round = 0
mode = '?'
for e in events:
    if e['event'] == 'orchestrator_start':
        mode = e.get('mode', '?')
    if e['event'] == 'round_start':
        current_round = e.get('round', 0)

last = events[-1]
activity = last['event']
branch = last.get('branch', '')
if activity == 'branch_progress':
    status = f'Running: {branch} ({last[\"completed\"]}/{last[\"target\"]} experiments, {last[\"elapsed_s\"]:.0f}s)'
elif activity == 'branch_start':
    status = f'Starting: {branch} (target: {last[\"target_experiments\"]} experiments)'
elif activity == 'branch_end':
    status = f'Finished: {branch} ({last[\"completed\"]} experiments in {last[\"duration_s\"]:.0f}s)'
elif activity == 'round_end':
    status = f'Round {last[\"round\"]} complete ({last[\"duration_s\"]:.0f}s)'
elif activity == 'migration':
    status = f'Migration: {last[\"source\"]} -> {last[\"dest\"]}'
elif activity == 'adoption':
    status = f'Adoption: {last[\"loser\"]} adopts from {last[\"winner\"]}'
else:
    status = activity

print(f'  Mode:     {mode}')
print(f'  Round:    {current_round}')
print(f'  Status:   {status}')
print()

lb_events = [e for e in events if e['event'] == 'leaderboard']
if lb_events:
    lb = lb_events[-1]
    entries = lb.get('entries', {})
    best_branch = lb.get('global_best_branch')
    best_bpb = lb.get('global_best_bpb')

    print(f'  {\"Branch\":<16} {\"Best BPB\":>10} {\"Experiments\":>12} {\"Kept\":>6} {\"Imp/Exp\":>10}')
    print(f'  {\"-\"*16} {\"-\"*10} {\"-\"*12} {\"-\"*6} {\"-\"*10}')
    for name in sorted(entries.keys()):
        entry = entries[name]
        bpb = entry.get('best_val_bpb')
        bpb_s = f'{bpb:.6f}' if bpb else 'N/A'
        marker = ' *' if name == best_branch else ''
        imp = entry.get('improvement_rate', 0)
        imp_s = f'{imp:.6f}' if imp else '—'
        print(f'  {name:<16} {bpb_s:>10} {entry[\"total_experiments\"]:>12} {entry[\"kept_experiments\"]:>6} {imp_s:>10}{marker}')

    if best_bpb:
        print(f'')
        print(f'  Global best: {best_bpb:.6f} ({best_branch})')
    print()

recent = [e for e in events[-20:] if e['event'] != 'branch_progress'][-8:]
if recent:
    print(f'  Recent events:')
    for e in recent:
        ts = e['ts'][11:19]
        evt = e['event']
        if evt == 'branch_start':
            print(f'    {ts}  {e.get(\"branch\",\"\"):<12} started (target: {e[\"target_experiments\"]})')
        elif evt == 'branch_end':
            print(f'    {ts}  {e[\"branch\"]:<12} finished ({e[\"completed\"]} exp, {e[\"duration_s\"]:.0f}s)')
        elif evt == 'round_start':
            print(f'    {ts}  Round {e[\"round\"]} started')
        elif evt == 'round_end':
            print(f'    {ts}  Round {e[\"round\"]} ended ({e[\"duration_s\"]:.0f}s)')
        elif evt == 'migration':
            print(f'    {ts}  Migration: {e[\"source\"]} -> {e[\"dest\"]} [{e[\"params\"]}]')
        elif evt == 'adoption':
            print(f'    {ts}  Adoption: {e[\"loser\"]} <- {e[\"winner\"]} [{e[\"params\"]}]')
        elif evt == 'error':
            print(f'    {ts}  ERROR: {e.get(\"message\",\"\")}')
        else:
            print(f'    {ts}  {evt}')
\"" 2>/dev/null || echo "  (waiting for events...)")
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
