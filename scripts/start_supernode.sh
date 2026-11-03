#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# TERMINALS 2, 3, 4, ... – one client (SuperNode) per terminal.
#
# Usage:
#   bash scripts/start_supernode.sh CLIENT_ID [PARTITIONS_DIR] [SUPERLINK_ADDRESS]
#
# Examples:
#   bash scripts/start_supernode.sh 0
#   bash scripts/start_supernode.sh 1 data/partitions/iid
#   bash scripts/start_supernode.sh 2 data/partitions/iid 192.168.1.10:8000
#   EXTRA_NODE_CONFIG="malicious=true" bash scripts/start_supernode.sh 2   # an attacker
#
# Client CLIENT_ID trains on <PARTITIONS_DIR>/client_<CLIENT_ID>.npz – i.e. each
# SuperNode only knows the path to ITS OWN data, like a real hospital would.
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ $# -lt 1 ]; then
    sed -n '5,14p' "$0"
    exit 1
fi
CLIENT_ID="$1"
PARTITIONS="$(abspath "${2:-$DEFAULT_PARTITIONS}")"
SUPERLINK="${3:-$SUPERLINK_ADDRESS}"
DATA_PATH="$PARTITIONS/client_$CLIENT_ID.npz"

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: $DATA_PATH does not exist (check CLIENT_ID and PARTITIONS_DIR)" >&2
    exit 1
fi

# Each SuperNode on the same machine needs its own local port (9094, 9095, ...)
PORT=$((9094 + CLIENT_ID))

# Optional extra node config, e.g. EXTRA_NODE_CONFIG="malicious=true" (docs/04)
NODE_CONFIG="data-path='$DATA_PATH'${EXTRA_NODE_CONFIG:+ $EXTRA_NODE_CONFIG}"

echo ">>> Starting SuperNode $CLIENT_ID -> SuperLink $SUPERLINK (node config: $NODE_CONFIG)"
exec flower-supernode \
    --insecure \
    --superlink "$SUPERLINK" \
    --port "$PORT" \
    --node-config "$NODE_CONFIG"
