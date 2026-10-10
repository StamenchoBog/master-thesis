#!/bin/bash
# Validation pair for the model-size cost model (experiments/protocol.md, Addendum A): the
# full campaign protocol (experiments/run/01-06) at one width, seed 42, naive then SISA.
# Results go to results/msc/model_size/validation/ (see experiments/run/common.sh).
#
#   experiments/model_size/validate.sh                     # width 384, ~2.5 h
#   MODEL_WIDTH=1280 experiments/model_size/validate.sh    # width 1280, ~3.5 h
set -euo pipefail
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -ims "$0" "$@"; fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PI=admin@rasp5node.local
export MODEL_WIDTH=${MODEL_WIDTH:-384}
SEED=42

# Keep sudo fresh so logger restarts and the cleanup never wait for a password.
sudo -v
while kill -0 $$ 2>/dev/null; do sudo -n true; sleep 60; done 2>/dev/null &
experiments/meter_alarm.sh 'results/msc/model_size/validation/*/power_fnb58.csv' $$ &
alarm=$!
trap 'kill $alarm 2>/dev/null || true' EXIT

# Cool to 72 C, then reboot until the throttle status is clean. A reboot sometimes catches a
# short under-voltage at boot, which sets a sticky bit; another reboot usually fixes it.
pi_ready() {
  until [ "$(ssh "$PI" vcgencmd measure_temp | grep -o '[0-9]*' | head -1)" -le 72 ]; do sleep 30; done
  for attempt in 1 2 3; do
    ssh "$PI" "sudo reboot" 2>/dev/null || true
    sleep 30
    until ssh -o ConnectTimeout=5 "$PI" true 2>/dev/null; do sleep 10; done
    sleep 25
    status=$(ssh "$PI" vcgencmd get_throttled)
    [ "$status" = "throttled=0x0" ] && return 0
    echo "Pi status $status after reboot $attempt, retrying" >&2
  done
  echo "Pi did not come up with a clean status; check the power and rerun." >&2
  exit 1
}

for arm in naive sisa; do
  run=results/msc/model_size/validation/w${MODEL_WIDTH}_${arm}_seed$SEED
  if [ -e "$run/recovery_manifest.json" ]; then echo "$arm already done"; continue; fi
  echo "=== $arm: preparing the Pi ($(date +%H:%M))"
  pi_ready
  for step in 01_prepare 02_instruments 03_train 04_recover 05_rejoin 06_collect; do
    echo "--- $arm $step ($(date +%H:%M))"
    "experiments/run/$step.sh" "$arm" "$SEED"
  done
done
echo "Validation pair done ($(date +%H:%M))."
