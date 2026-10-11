#!/bin/bash
# Sourced by every step. Arguments: naive|sisa SEED [--wan]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source experiments/lib.sh

ARM=${1:-}
SEED=${2:-}
case "$ARM" in
  naive | sisa) ;;
  *) echo "usage: $0 naive|sisa SEED [--wan]" >&2; exit 1 ;;
esac

HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
MODEL_WIDTH=${MODEL_WIDTH:-128}
export SEED MODEL_WIDTH

if [ "$ARM" = naive ]; then CLIENT_MODE=standard; else CLIENT_MODE=sisa; fi

# shellcheck disable=SC2034  # PROFILE is used by the step scripts
if [ "${3:-}" = --wan ]; then
  RUN=results/msc/campaign/wan_${ARM}_seed$SEED
  PROFILE="--profile wan"
  FLEET_PORT=19092
else
  RUN=results/msc/campaign/${ARM}_seed$SEED
  PROFILE=""
  FLEET_PORT=9092
fi
if [ "$MODEL_WIDTH" != 128 ]; then
  RUN=results/msc/model_size/validation/w${MODEL_WIDTH}_${ARM}_seed$SEED
fi

edge() {
  ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED FLEET_PORT=$FLEET_PORT CLIENT_MODE=$CLIENT_MODE MODEL_WIDTH=$MODEL_WIDTH $*"
}

# Phase marker for the power log (host) and the telemetry (Pi).
mark() {
  echo "$(date +%s) $1" >> "$RUN/phases.log"
  ssh "$PI" "echo $1 > /dev/shm/run_marker"
}
