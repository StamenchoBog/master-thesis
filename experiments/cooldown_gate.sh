#!/bin/bash
# Block until the Pi's SoC temperature has plateaued — i.e. it has stopped
# cooling and settled at its passive idle floor. The fan is intentionally
# unplugged (passive-cooling experiment), so the Pi never reaches a low absolute
# temperature; what matters for run-to-run comparability is that every measured
# run starts from the same *stable* thermal state, not a specific number.
#
# "Stable" = the temperature range over a sliding STABLE_S-second window stays
# within TOLERANCE_C. A CEILING_C guard refuses to proceed if the plateau is
# abnormally hot (e.g. the SoC is stuck near the throttle limit).
#
# Usage: ./cooldown_gate.sh [TOLERANCE_C] [STABLE_S] [CEILING_C]
set -euo pipefail

TOLERANCE_C=${1:-2}       # max spread within the window to call it a plateau
STABLE_S=${2:-120}        # how long the plateau must hold
CEILING_C=${3:-80}        # don't start a run above this even if stable (soft-limit guard)
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

  # (Re)start the window on the first reading or whenever the spread breaks tolerance.
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
