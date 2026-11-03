#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs ONE Flower component in its own terminal window and, when it stops,
# offers to run it again. Started by run_deployment_terminals.sh, not by hand.
#
#   bash scripts/_terminal_loop.sh superlink
#   bash scripts/_terminal_loop.sh supernode CLIENT_ID PARTITIONS_DIR
#   bash scripts/_terminal_loop.sh submit PARTITIONS_DIR [RUN_CONFIG_OVERRIDES]
#
# Ctrl+C stops the component but keeps the window open with the menu.
# ---------------------------------------------------------------------------
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/_common.sh"

ROLE="${1:?role missing}"
shift

# Ctrl+C goes to the running component; this loop survives and shows the menu
INTERRUPTED=0
trap 'INTERRUPTED=1; echo' INT

set_title() { printf '\033]0;%s\007' "$1"; }

# SuperNodes and the submit step need a running SuperLink.
# Returns 1 if the user pressed Ctrl+C while waiting.
wait_for_superlink() {
    local host="${SUPERLINK_ADDRESS%:*}" port="${SUPERLINK_ADDRESS##*:}" shown=0
    INTERRUPTED=0
    until python -c 'import socket, sys; socket.create_connection((sys.argv[1], int(sys.argv[2])), 1)' \
            "$host" "$port" 2>/dev/null; do
        [ "$INTERRUPTED" = 1 ] && return 1
        if [ "$shown" = 0 ]; then
            echo ">>> Waiting for the SuperLink on $SUPERLINK_ADDRESS ... (Ctrl+C for the menu)"
            shown=1
        fi
        sleep 1
    done
    [ "$INTERRUPTED" = 0 ]
}

# Read the menu choice into $choice. Ctrl+C at the prompt asks again; end of
# input (window closed) quits.
ask() {
    local status
    while true; do
        read -r -p "> " choice
        status=$?
        [ "$status" = 0 ] && return
        [ "$status" -gt 128 ] || exit 0
    done
}

banner() {
    echo
    echo "======================================================================"
    echo " $1"
    echo "======================================================================"
}

case "$ROLE" in
superlink)
    set_title "SuperLink"
    while true; do
        bash "$SCRIPTS_DIR/start_superlink.sh"
        banner "SuperLink stopped (exit code $?)"
        echo "  [Enter] start it again    [q] close this window"
        ask
        [ "$choice" = q ] && break
    done
    ;;

supernode)
    CLIENT_ID="${1:?client id missing}"
    PARTITIONS="${2:?partitions dir missing}"
    EXTRA_NODE_CONFIG=""
    while true; do
        if [ -n "$EXTRA_NODE_CONFIG" ]; then
            set_title "SuperNode $CLIENT_ID (ATTACKER)"
        else
            set_title "SuperNode $CLIENT_ID"
        fi
        if wait_for_superlink; then
            started=$SECONDS
            EXTRA_NODE_CONFIG="$EXTRA_NODE_CONFIG" \
                bash "$SCRIPTS_DIR/start_supernode.sh" "$CLIENT_ID" "$PARTITIONS"
            status=$?
            # The SuperNode stopped by itself (no Ctrl+C here) after running for
            # a while: the SuperLink went away, e.g. it was restarted and forgot
            # all nodes. Flower then exits with 0, or 130 because it stops itself
            # with SIGINT. Reconnect automatically. Quick failures (wrong data
            # path, ...) still show the menu, so they cannot loop forever.
            if [ "$INTERRUPTED" = 0 ] && { [ "$status" = 0 ] || [ "$status" = 130 ]; } \
                    && [ $((SECONDS - started)) -ge 5 ]; then
                echo ">>> SuperNode $CLIENT_ID lost the SuperLink. Reconnecting in 3 s" \
                     "(Ctrl+C for the menu instead) ..."
                sleep 3
                [ "$INTERRUPTED" = 0 ] && continue
            fi
            banner "SuperNode $CLIENT_ID stopped (exit code $status)"
        else
            banner "SuperNode $CLIENT_ID: stopped waiting for the SuperLink"
        fi
        echo "  [Enter] start it again (same settings)"
        echo "  [m]     start it again as an ATTACKER (malicious=true, see docs/04)"
        echo "  [h]     start it again as an honest client"
        echo "  [q]     close this window"
        ask
        case "$choice" in
            q) break ;;
            m) EXTRA_NODE_CONFIG="malicious=true" ;;
            h) EXTRA_NODE_CONFIG="" ;;
        esac
    done
    ;;

submit)
    PARTITIONS="${1:?partitions dir missing}"
    OVERRIDES="${2:-}"
    set_title "Submit run (flwr run)"
    run_number=1
    while true; do
        if wait_for_superlink; then
            banner "Run $run_number | overrides: ${OVERRIDES:-<none>}"
            bash "$SCRIPTS_DIR/submit_run.sh" "$PARTITIONS" "$OVERRIDES"
            banner "Run $run_number finished (exit code $?)"
            run_number=$((run_number + 1))
        else
            banner "Stopped waiting for the SuperLink"
        fi
        echo "  [Enter] submit the same run again"
        echo "  [e]     edit the run-config overrides, then submit"
        echo "          (e.g. num-server-rounds=10 strategy='krum' malicious-clients='2')"
        echo "  [q]     close this window (the SuperLink and SuperNodes keep running)"
        ask
        case "$choice" in
            q) break ;;
            e) read -r -e -i "$OVERRIDES" -p "overrides> " OVERRIDES ;;
        esac
    done
    ;;

*)
    echo "Unknown role '$ROLE' (superlink | supernode | submit)" >&2
    exit 1
    ;;
esac
