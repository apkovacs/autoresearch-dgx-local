#!/usr/bin/env bash
# monitor-dgx.sh — Monitor autoresearch training on DGX Spark
#
# Shows Docker container stats and GPU utilization in real-time.
# Works with both run-dgx.sh and run-dgx-agent.sh containers.
#
# Usage:
#   bash monitor-dgx.sh                    # auto-detect container
#   bash monitor-dgx.sh <container-name>   # monitor specific container

set -euo pipefail

CONTAINER="${1:-}"

# Auto-detect container if not specified
if [ -z "$CONTAINER" ]; then
    if docker ps --format '{{.Names}}' | grep -q "autoresearch-dgx-agent"; then
        CONTAINER="autoresearch-dgx-agent"
    elif docker ps --format '{{.Names}}' | grep -q "autoresearch-dgx"; then
        CONTAINER="autoresearch-dgx"
    else
        echo "No autoresearch container found running."
        echo "Start one with: bash run-dgx.sh  or  bash run-dgx-agent.sh"
        exit 1
    fi
fi

echo "Monitoring container: $CONTAINER"
echo "Press Ctrl+C to stop."
echo ""

while true; do
    clear
    echo "=== $(date) === Container: $CONTAINER ==="
    echo ""
    echo "--- Docker Stats ---"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" "$CONTAINER" 2>/dev/null || echo "(container not running)"
    echo ""
    echo "--- GPU Status ---"
    docker exec "$CONTAINER" nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "(unable to query GPU)"
    echo ""
    echo "--- Training Log (last 5 lines) ---"
    docker logs --tail 5 "$CONTAINER" 2>/dev/null || echo "(no logs)"
    sleep 2
done
