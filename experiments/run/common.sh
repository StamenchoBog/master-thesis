#!/bin/bash
# Shared settings, sourced by every step. Arguments: naive|sisa SEED [--wan]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ARM=${1:-}
SEED=${2:-}
case "$ARM" in
  naive | sisa) ;;
  *) echo "usage: $0 naive|sisa SEED [--wan]" >&2; exit 1 ;;
esac

PI=admin@rasp5node.local
HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
export SEED

if [ "$ARM" = naive ]; then CLIENT_MODE=standard; else CLIENT_MODE=sisa; fi

# shellcheck disable=SC2034  # PROFILE is used by the step scripts
if [ "${3:-}" = --wan ]; then
  RUN=results/msc/runs/wan_${ARM}_seed$SEED
  PROFILE="--profile wan"
  FLEET_PORT=19092
else
  RUN=results/msc/runs/${ARM}_seed$SEED
  PROFILE=""
  FLEET_PORT=9092
fi

# Run a command in the Pi's repo checkout; the edge compose file always needs HOST_IP.
edge() {
  ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED FLEET_PORT=$FLEET_PORT CLIENT_MODE=$CLIENT_MODE $*"
}

# Phase marker for the power log (host) and the telemetry (Pi).
mark() {
  echo "$(date +%s) $1" >> "$RUN/phases.log"
  ssh "$PI" "echo $1 > /dev/shm/run_marker"
}
