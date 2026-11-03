#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deployment mode with ONE WINDOW PER COMPONENT, opened for you:
#   * "SuperLink"        the server side
#   * "SuperNode 0..N-1" one client each (N = num_clients in meta.json)
#   * "Submit run"       flwr run, followed by a menu to run again
#
# Unlike run_deployment_local.sh, nothing is shut down after the run: every
# window shows a menu when its component stops, so you can re-run pieces
# individually, e.g. submit again with other settings, or restart client 2 as
# an attacker. Ctrl+C in a window stops only that component.
#
# Usage:
#   bash scripts/run_deployment_terminals.sh [PARTITIONS_DIR] [RUN_CONFIG_OVERRIDES]
#
# Examples:
#   bash scripts/run_deployment_terminals.sh
#   bash scripts/run_deployment_terminals.sh data/partitions/iid "num-server-rounds=10"
#   FL_TERMINAL=tmux bash scripts/run_deployment_terminals.sh   # no GUI (SSH, WSL)
#
# Terminal programs (auto-detected, or force one with FL_TERMINAL=...):
#   gnome-terminal | konsole | xfce4-terminal | xterm | macos | tmux
# Stop everything: [q] in every window, or bash scripts/stop_flower.sh
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/_common.sh"

PARTITIONS="$(abspath "${1:-$DEFAULT_PARTITIONS}")"
EXTRA_CONFIG="${2:-}"
check_partitions "$PARTITIONS"
NUM_CLIENTS="$(num_clients_of "$PARTITIONS")"
TMUX_SESSION="aitdm-fl"

# A deployment SuperLink / SuperNodes that are already running would block the
# ports. (The hidden SuperLink of simulation runs, --simulation, does not.)
if pgrep -u "$(id -u)" -af "^[^ ]*python[0-9.]* [^ ]*/flower-(superlink|supernode)( |$)" \
        | grep -qv -- "--simulation"; then
    echo "ERROR: Flower components are already running. Stop them first:" >&2
    echo "    bash scripts/stop_flower.sh" >&2
    exit 1
fi
ensure_connection  # once here, so the windows do not race to edit the config

detect_terminal() {
    if [ "$(uname)" = Darwin ]; then
        echo macos
        return
    fi
    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        local t
        for t in gnome-terminal konsole xfce4-terminal xterm; do
            if command -v "$t" >/dev/null; then
                echo "$t"
                return
            fi
        done
    fi
    if command -v tmux >/dev/null; then
        echo tmux
        return
    fi
    echo none
}

TERMINAL="${FL_TERMINAL:-$(detect_terminal)}"
if [ "$TERMINAL" = none ]; then
    cat >&2 <<'EOF'
ERROR: no supported terminal program found.
  * Desktop Linux / WSL2 with a GUI:  sudo apt install xterm
  * No GUI (SSH, WSL without GUI):    sudo apt install tmux
  * Or start the components by hand, one terminal each (docs/02_running_flower.md),
    or use: bash scripts/run_deployment_local.sh
EOF
    exit 1
fi

# open_window TITLE ARG... : run "bash _terminal_loop.sh ARG..." in a new window
open_window() {
    local title="$1"
    shift
    local cmd
    cmd="$(printf '%q ' bash "$SCRIPTS_DIR/_terminal_loop.sh" "$@")"
    # Keep the window open if the loop itself fails, so the error can be read
    cmd="cd $(printf '%q' "$REPO_ROOT") && $cmd || read -r -p 'Press Enter to close this window'"

    case "$TERMINAL" in
        gnome-terminal) gnome-terminal --title="$title" -- bash -c "$cmd" ;;
        konsole) konsole --new-tab -p tabtitle="$title" -e bash -c "$cmd" >/dev/null 2>&1 & ;;
        xfce4-terminal) xfce4-terminal --title="$title" -x bash -c "$cmd" ;;
        xterm) xterm -T "$title" -geometry 110x30 -e bash -c "$cmd" & ;;
        macos)
            local escaped="${cmd//\\/\\\\}"
            escaped="${escaped//\"/\\\"}"
            osascript -e "tell application \"Terminal\" to do script \"$escaped\"" >/dev/null
            ;;
        tmux)
            if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
                tmux new-window -t "$TMUX_SESSION" -n "$title" "bash -c $(printf '%q' "$cmd")"
            else
                tmux new-session -d -s "$TMUX_SESSION" -n "$title" "bash -c $(printf '%q' "$cmd")"
            fi
            ;;
        *)
            echo "ERROR: unknown FL_TERMINAL '$TERMINAL'" >&2
            exit 1
            ;;
    esac
}

if [ "$TERMINAL" = tmux ] && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$TMUX_SESSION' already exists. Close it first:" >&2
    echo "    tmux kill-session -t $TMUX_SESSION" >&2
    exit 1
fi

echo ">>> Opening $((NUM_CLIENTS + 2)) windows with $TERMINAL"
echo "    SuperLink | SuperNode 0..$((NUM_CLIENTS - 1)) | Submit run"
open_window "SuperLink" superlink
for ((i = 0; i < NUM_CLIENTS; i++)); do
    open_window "SuperNode $i" supernode "$i" "$PARTITIONS"
done
open_window "Submit run" submit "$PARTITIONS" "$EXTRA_CONFIG"

cat <<EOF

>>> Done. The SuperNodes and the submit window wait for the SuperLink, then the
    first run starts by itself. When a component stops, its window shows a menu:
      Submit run    [Enter] same run again | [e] edit the overrides | [q] close
      SuperNode i   [Enter] restart | [m] restart as attacker | [h] as honest | [q] close
      SuperLink     [Enter] restart | [q] close
    Ctrl+C in a window stops only that component.
    Stop everything: [q] in every window, or  bash scripts/stop_flower.sh
EOF

if [ "$TERMINAL" = tmux ]; then
    echo
    echo "    tmux: switch windows with Ctrl+b n / Ctrl+b p (or Ctrl+b w for a list),"
    echo "    detach with Ctrl+b d, come back with: tmux attach -t $TMUX_SESSION"
    if [ -t 0 ] && [ -z "${TMUX:-}" ]; then
        exec tmux attach -t "$TMUX_SESSION"
    fi
fi
