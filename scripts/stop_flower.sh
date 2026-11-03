#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stop every Flower process of the current user: SuperLinks, SuperNodes and
# the background "managed" SuperLink that `flwr run .` starts for simulations.
# Use it when a port is "already in use" or a run seems stuck.
# ---------------------------------------------------------------------------
found=0
for component in supernode superlink superexec; do
    # Only match the actual Python processes ("python .../bin/flower-<component> ...")
    if pkill -u "$(id -u)" -f "^[^ ]*python[0-9.]* [^ ]*/flower-$component( |$)"; then
        echo "Stopped flower-$component"
        found=1
    fi
done
[ "$found" -eq 0 ] && echo "No Flower processes running."
exit 0
