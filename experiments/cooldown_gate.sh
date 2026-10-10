#!/bin/bash
# Wait until the Pi's SoC temperature has settled at its idle level. The Pi runs without
# a fan, so it never gets really cold; every run should start from the same stable state.
# Stable means the temperature stays within TOLERANCE_C for STABLE_S seconds, below CEILING_C.
#
# Usage: ./cooldown_gate.sh [TOLERANCE_C] [STABLE_S] [CEILING_C]
set -euo pipefail

TOLERANCE_C=${1:-2}
STABLE_S=${2:-120}
CEILING_C=${3:-80}
PI=${PI_HOST:-admin@rasp5node.local}
KEY=${SSH_KEY:-$HOME/.ssh/rasp5node}
POLL_S=15

started=$(date +%s)
win_start=""
win_min=0
win_max=0

echo "Cooldown gate: waiting for a stable plateau (range <= ${TOLERANCE_C}C for ${STABLE_S}s, below ${CEILING_C}C)..."
while true; do
  temp_milli=$(ssh -i "$KEY" -o IdentitiesOnly=yes "$PI" 'cat /sys/class/thermal/thermal_zone0/temp')
  temp=$((temp_milli / 1000))
  now=$(date +%s)

  # Restart the window when the spread gets larger than the tolerance.
  if [ -z "$win_start" ] || [ $((temp - win_min)) -gt "$TOLERANCE_C" ] || [ $((win_max - temp)) -gt "$TOLERANCE_C" ]; then
    win_start=$now; win_min=$temp; win_max=$temp
  else
    [ "$temp" -lt "$win_min" ] && win_min=$temp
    [ "$temp" -gt "$win_max" ] && win_max=$temp
  fi

  held=$((now - win_start))
  echo "  $(date +%H:%M:%S) SoC ${temp}C  (window ${win_min}-${win_max}C, held ${held}s)"

  if [ "$held" -ge "$STABLE_S" ] && [ "$win_max" -le "$CEILING_C" ]; then
    echo "Cooldown complete: plateau at ~${temp}C (waited $((now - started))s total)"
    exit 0
  fi
  sleep "$POLL_S"
done
