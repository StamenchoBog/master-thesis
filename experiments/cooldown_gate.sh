#!/bin/bash
# Wait until the Pi's SoC temperature stays within TOLERANCE_C for STABLE_S seconds, below
# CEILING_C, so every run starts from the same idle state (the Pi has no fan).
#
#   experiments/cooldown_gate.sh [TOLERANCE_C] [STABLE_S] [CEILING_C]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TOLERANCE_C=${1:-2}
STABLE_S=${2:-120}
CEILING_C=${3:-80}

started=$(date +%s)
win_start=""
win_min=0
win_max=0

echo "Cooldown gate: waiting for a range <= ${TOLERANCE_C}C over ${STABLE_S}s, below ${CEILING_C}C"
while true; do
  temp=$(( $(ssh "$PI" cat /sys/class/thermal/thermal_zone0/temp) / 1000 ))
  now=$(date +%s)
  if [ -z "$win_start" ] || [ $((temp - win_min)) -gt "$TOLERANCE_C" ] || [ $((win_max - temp)) -gt "$TOLERANCE_C" ]; then
    win_start=$now; win_min=$temp; win_max=$temp
  else
    [ "$temp" -lt "$win_min" ] && win_min=$temp
    [ "$temp" -gt "$win_max" ] && win_max=$temp
  fi
  held=$((now - win_start))
  echo "  $(date +%H:%M:%S) SoC ${temp}C (window ${win_min}-${win_max}C, held ${held}s)"
  if [ "$held" -ge "$STABLE_S" ] && [ "$win_max" -le "$CEILING_C" ]; then
    echo "Cooldown complete: ~${temp}C after $((now - started))s"
    exit 0
  fi
  sleep 15
done
