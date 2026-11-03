#!/usr/bin/env bash
# Shared settings for the run scripts. Sourced, never executed directly.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/fl_app"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
DEFAULT_PARTITIONS="$REPO_ROOT/data/partitions/dirichlet"

# Name of the SuperLink connection used for deployment runs (~/.flwr/config.toml)
CONNECTION_NAME="local-deployment"
SUPERLINK_ADDRESS="127.0.0.1:8000"

# Activate the virtual environment (created by scripts/setup_env.sh)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "ERROR: no virtual environment in $VENV_DIR. Run: bash scripts/setup_env.sh" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Everything is already installed in the venv: stop Flower from building a
# separate environment (and downloading PyTorch again) for every run.
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
# Silence a harmless Ray warning printed in simulation mode
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Turn a (possibly relative) path into an absolute one. Flower runs the
# apps from its own working directory, so relative paths would break.
abspath() {
    python -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# Fail early with a helpful message if the partitions were not created yet
check_partitions() {
    local dir="$1"
    if [ ! -f "$dir/test.npz" ] || [ ! -f "$dir/meta.json" ]; then
        echo "ERROR: no partitions found in $dir" >&2
        echo "Create them first, e.g.:" >&2
        echo "    python data_splitting/split_data.py --dataset dermamnist --num-clients 3 --strategy dirichlet" >&2
        exit 1
    fi
}

# Number of clients stored in meta.json by split_data.py
num_clients_of() {
    python -c 'import json, sys; print(json.load(open(sys.argv[1]))["num_clients"])' "$1/meta.json"
}

# Add the deployment connection to ~/.flwr/config.toml (only once)
ensure_connection() {
    local config_file
    config_file="$(flwr config list 2>/dev/null | sed -n 's/^Flower Config file: //p')"
    config_file="${config_file:-$HOME/.flwr/config.toml}"
    if ! grep -q "^\[superlink\.$CONNECTION_NAME\]" "$config_file" 2>/dev/null; then
        echo ">>> Adding connection '$CONNECTION_NAME' ($SUPERLINK_ADDRESS) to $config_file"
        mkdir -p "$(dirname "$config_file")"
        printf '\n[superlink.%s]\naddress = "%s"\ninsecure = true\n' \
            "$CONNECTION_NAME" "$SUPERLINK_ADDRESS" >> "$config_file"
    fi
}
