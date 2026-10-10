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
# 128 is the campaign model; other widths are the model-size validation runs (Addendum A).
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

# Run a command in the Pi's repo checkout; the edge compose file always needs HOST_IP.
edge() {
  ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED FLEET_PORT=$FLEET_PORT CLIENT_MODE=$CLIENT_MODE MODEL_WIDTH=$MODEL_WIDTH $*"
}

# Phase marker for the power log (host) and the telemetry (Pi).
mark() {
  echo "$(date +%s) $1" >> "$RUN/phases.log"
  ssh "$PI" "echo $1 > /dev/shm/run_marker"
}

# Block until the power log is fresh, so no measured phase starts without the meter. While
# waiting, a Mac notification and a sound every minute ask for a replug of the FNB58 cable.
wait_for_meter() {
  local f=$1 last=0
  while [ ! -f "$f" ] || [ $(( $(date +%s) - $(stat -f %m "$f") )) -gt 15 ]; do
    if [ $(( $(date +%s) - last )) -ge 60 ]; then
      echo "$(date +%H:%M:%S) power meter not streaming, replug the FNB58 USB cable; waiting" >&2
      osascript -e 'display notification "Replug the FNB58 USB cable" with title "Power meter stopped" sound name "Glass"' 2>/dev/null || true
      afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &
      last=$(date +%s)
    fi
    sleep 5
  done
}
