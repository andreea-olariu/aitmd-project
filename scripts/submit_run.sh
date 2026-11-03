#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LAST TERMINAL – submit the Flower app to the running SuperLink.
# Start the SuperLink and ALL SuperNodes first.
#
# Usage:
#   bash scripts/submit_run.sh [PARTITIONS_DIR] [RUN_CONFIG_OVERRIDES]
#
# Examples:
#   bash scripts/submit_run.sh
#   bash scripts/submit_run.sh data/partitions/iid "num-server-rounds=10"
#
# PARTITIONS_DIR is needed by the server for the global test set.
# The server waits until <num_clients in meta.json> SuperNodes are connected.
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PARTITIONS="$(abspath "${1:-$DEFAULT_PARTITIONS}")"
EXTRA_CONFIG="${2:-}"
check_partitions "$PARTITIONS"
NUM_CLIENTS="$(num_clients_of "$PARTITIONS")"
SAVE_DIR="$REPO_ROOT/outputs/deploy-$(basename "$PARTITIONS")-$(date +%Y%m%d-%H%M%S)"

ensure_connection

echo ">>> Submitting run to '$CONNECTION_NAME' (waiting for $NUM_CLIENTS SuperNodes)"
cd "$APP_DIR"
flwr run . "$CONNECTION_NAME" --stream \
    --run-config "partitions-dir='$PARTITIONS' save-dir='$SAVE_DIR' min-nodes=$NUM_CLIENTS $EXTRA_CONFIG"

# `flwr run` does not fail when the ServerApp crashes: check that results exist
if [ ! -f "$SAVE_DIR/metrics.json" ]; then
    echo ">>> The run FAILED - read the error (Traceback) above." >&2
    echo "    (If the SuperLink runs on another machine, the results are saved there.)" >&2
    exit 1
fi
echo
echo ">>> Done. Model and metrics: $SAVE_DIR"
