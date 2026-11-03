#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the whole federation (server + all clients) on this machine with
# Flower's Simulation Runtime – ONE command, ONE terminal.
#
# Usage:
#   bash scripts/run_simulation.sh [PARTITIONS_DIR] [RUN_CONFIG_OVERRIDES]
#
# Examples:
#   bash scripts/run_simulation.sh
#   bash scripts/run_simulation.sh data/partitions/iid
#   bash scripts/run_simulation.sh data/partitions/dirichlet "num-server-rounds=10 strategy='fedprox'"
#
# The number of simulated clients is read from <PARTITIONS_DIR>/meta.json.
# Results are written to outputs/sim-<partition-name>-<timestamp>/.
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PARTITIONS="$(abspath "${1:-$DEFAULT_PARTITIONS}")"
EXTRA_CONFIG="${2:-}"
check_partitions "$PARTITIONS"
NUM_CLIENTS="$(num_clients_of "$PARTITIONS")"
# Results folder (SAVE_DIR=... to choose it, as experiments/run_seeds.py does)
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/outputs/sim-$(basename "$PARTITIONS")-$(date +%Y%m%d-%H%M%S)}"

# `flwr run .` starts a background "managed" SuperLink and keeps it alive.
# Restart it so that it uses THIS virtual environment and settings.
if pkill -f "[f]lower-superlink.*--simulation" 2>/dev/null; then
    echo ">>> Restarting Flower's local simulation SuperLink"
    sleep 2
fi

echo ">>> Simulating $NUM_CLIENTS clients on $PARTITIONS"
cd "$APP_DIR"
flwr run . --stream \
    --federation-config "num-supernodes=$NUM_CLIENTS" \
    --run-config "partitions-dir='$PARTITIONS' save-dir='$SAVE_DIR' min-nodes=$NUM_CLIENTS $EXTRA_CONFIG"

# `flwr run` does not fail when the ServerApp crashes: check that results exist
if [ ! -f "$SAVE_DIR/metrics.json" ]; then
    echo ">>> The run FAILED - read the error (Traceback) above." >&2
    exit 1
fi
echo
echo ">>> Done. Model and metrics: $SAVE_DIR"
