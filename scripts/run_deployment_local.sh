#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deployment mode in ONE terminal: starts the SuperLink and one SuperNode per
# client in the background, submits the run, then shuts everything down.
# It does exactly what you would do by hand in several terminals (see
# docs/02_running_flower.md); the logs of each component go to logs/.
#
# Usage:
#   bash scripts/run_deployment_local.sh [PARTITIONS_DIR] [RUN_CONFIG_OVERRIDES]
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/_common.sh"

PARTITIONS="$(abspath "${1:-$DEFAULT_PARTITIONS}")"
EXTRA_CONFIG="${2:-}"
check_partitions "$PARTITIONS"
NUM_CLIENTS="$(num_clients_of "$PARTITIONS")"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    echo ">>> Stopping SuperNodes and SuperLink"
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ">>> [terminal 1] SuperLink           -> logs/superlink.log"
bash "$SCRIPTS_DIR/start_superlink.sh" > "$LOG_DIR/superlink.log" 2>&1 &
PIDS+=($!)
sleep 4

for ((i = 0; i < NUM_CLIENTS; i++)); do
    echo ">>> [terminal $((i + 2))] SuperNode $i         -> logs/supernode_$i.log"
    bash "$SCRIPTS_DIR/start_supernode.sh" "$i" "$PARTITIONS" > "$LOG_DIR/supernode_$i.log" 2>&1 &
    PIDS+=($!)
done
sleep 3

echo ">>> [terminal $((NUM_CLIENTS + 2))] flwr run"
bash "$SCRIPTS_DIR/submit_run.sh" "$PARTITIONS" "$EXTRA_CONFIG"
