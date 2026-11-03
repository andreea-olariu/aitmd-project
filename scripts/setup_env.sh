#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Creates a Python virtual environment in ./.venv and installs everything
# from requirements.txt (Flower, PyTorch, MONAI, ...) plus the Flower app in
# fl_app/ (editable mode).
#
# Usage (from the repository root):
#   bash scripts/setup_env.sh              # default PyTorch build (CUDA on Linux)
#   bash scripts/setup_env.sh --cpu        # CPU-only PyTorch (much smaller download)
#   PYTHON=python3.12 bash scripts/setup_env.sh   # choose the interpreter
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
TORCH_INDEX=""

# Pick the first interpreter that actually runs and is 3.11-3.13. Trying them
# matters on Windows, where 'python3' is often a Microsoft Store stub that
# exists but only prints "Python was not found".
find_python() {
    local cand v
    for cand in python3.13 python3.12 python3.11 python3 python; do
        if "$cand" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))' >/dev/null 2>&1; then
            echo "$cand"; return
        fi
    done
    # Windows: ask the py launcher for the full path of a matching version
    if command -v py >/dev/null 2>&1; then
        for v in 3.13 3.12 3.11; do
            if cand=$(py -"$v" -c 'import sys; sys.stdout.write(sys.executable)' 2>/dev/null); then
                echo "$cand"; return
            fi
        done
    fi
    echo python3   # nothing suitable; the check below reports the error
}
PYTHON="${PYTHON:-$(find_python)}"

for arg in "$@"; do
    case "$arg" in
        --cpu) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# --- 1. Check the Python version (Flower needs >= 3.11) --------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.11+ or set PYTHON=/path/to/python" >&2
    exit 1
fi
"$PYTHON" - <<'EOF'
import sys
if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    sys.exit(f"ERROR: Python 3.11-3.13 required, found {sys.version.split()[0]}. "
             "Set PYTHON=python3.12 (or similar) and re-run.")
print(f"Using Python {sys.version.split()[0]}")
EOF

# --- 2. Create the virtual environment ------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating virtual environment in $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo ">>> Re-using existing virtual environment in $VENV_DIR"
fi
# Linux/macOS venvs use bin/, Windows venvs (Git Bash) use Scripts/
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    ACTIVATE="$VENV_DIR/Scripts/activate"
else
    ACTIVATE="$VENV_DIR/bin/activate"
fi
# shellcheck disable=SC1090
source "$ACTIVATE"
python -m pip install --upgrade pip setuptools wheel

# --- 3. Install PyTorch first (so the CPU/CUDA choice is respected) -------
TORCH_PKGS=$(grep -E '^(torch|torchvision)==' "$REPO_ROOT/requirements.txt")
if [ -n "$TORCH_INDEX" ]; then
    echo ">>> Installing CPU-only PyTorch"
    # shellcheck disable=SC2086
    pip install $TORCH_PKGS --index-url "$TORCH_INDEX"
else
    echo ">>> Installing PyTorch (default build)"
    # shellcheck disable=SC2086
    pip install $TORCH_PKGS
fi

# --- 4. Install the remaining requirements and the Flower app -------------
echo ">>> Installing requirements.txt"
pip install -r "$REPO_ROOT/requirements.txt"
echo ">>> Installing the Flower app (fl_app/) in editable mode"
pip install -e "$REPO_ROOT/fl_app"

# --- 5. Sanity check -------------------------------------------------------
python - <<'EOF'
import flwr, torch, torchvision, monai, numpy
print("\n=== Environment ready ===")
print(f"flwr        {flwr.__version__}")
print(f"torch       {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"torchvision {torchvision.__version__}")
print(f"monai       {monai.__version__}")
print(f"numpy       {numpy.__version__}")
EOF

echo
echo "Activate the environment in every new terminal with:"
echo "    source \"$ACTIVATE\""
