#!/bin/bash
# Block until the Pi's SoC has been at or below THRESHOLD_C for STABLE_S
# consecutive seconds. Every measured run must pass this gate so thermal
# history from the previous run cannot confound the next one.
#
# Usage: ./cooldown_gate.sh [THRESHOLD_C] [STABLE_S]
set -euo pipefail

THRESHOLD_C=${1:-40}
STABLE_S=${2:-120}
PI=${PI_HOST:-admin@rasp5node.local}
KEY=${SSH_KEY:-$HOME/.ssh/rasp5node}
POLL_S=10

started=$(date +%s)
stable_since=""

echo "Cooldown gate: waiting for SoC <= ${THRESHOLD_C}C stable for ${STABLE_S}s..."
while true; do
  temp_milli=$(ssh -i "$KEY" -o IdentitiesOnly=yes "$PI" 'cat /sys/class/thermal/thermal_zone0/temp')
  temp=$((temp_milli / 1000))
  now=$(date +%s)
  if [ "$temp" -le "$THRESHOLD_C" ]; then
    [ -z "$stable_since" ] && stable_since=$now
    if [ $((now - stable_since)) -ge "$STABLE_S" ]; then
      echo "Cooldown complete: ${temp}C (waited $((now - started))s total)"
      exit 0
    fi
  else
    stable_since=""
  fi
  echo "  $(date +%H:%M:%S) SoC ${temp}C"
  sleep "$POLL_S"
done
