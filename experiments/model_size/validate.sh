#!/bin/bash
# Cost-model validation: the campaign steps (experiments/run/01-06) at one model width,
# seed 42, naive then SISA. Results go to results/msc/model_size/validation/.
#
#   experiments/model_size/validate.sh                                    # width 384
#   MODEL_WIDTH=1280 experiments/model_size/validate.sh
#   RECOVERY_ONLY=1 MODEL_WIDTH=4096 experiments/model_size/validate.sh   # Phase 3 only
#
# RECOVERY_ONLY skips Phases 1 and 4: make_checkpoints.py writes the checkpoints SISA rolls
# back to, and only the recovery is measured.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
stay_awake "$@"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export MODEL_WIDTH=${MODEL_WIDTH:-384}
SEED=42

keep_sudo
experiments/meter_alarm.sh 'results/msc/model_size/validation/*/power_fnb58.csv' $$ &
alarm=$!
trap 'kill $alarm 2>/dev/null || true' EXIT

make_checkpoints() {
  [ "$1" = sisa ] || return 0
  ssh "$PI" "mkdir -p master-thesis/experiments/model_size"
  scp -q experiments/model_size/make_checkpoints.py "$PI:master-thesis/experiments/model_size/"
  ssh "$PI" "cd master-thesis && HOST_IP=127.0.0.1 SEED=$SEED MODEL_WIDTH=$MODEL_WIDTH docker compose -f docker-compose.edge.yml run --rm --no-deps -v \$PWD:/app -w /app --entrypoint python superexec-clientapp-4 -m experiments.model_size.make_checkpoints"
}

if [ -n "${RECOVERY_ONLY:-}" ]; then
  # Writing the checkpoints heats the Pi, so the cool-down and reboot come after it.
  steps="01_prepare make_checkpoints pi_ready 02_instruments 04_recover 06_collect"
else
  steps="pi_ready 01_prepare 02_instruments 03_train 04_recover 05_rejoin 06_collect"
fi

for arm in naive sisa; do
  run=results/msc/model_size/validation/w${MODEL_WIDTH}_${arm}_seed$SEED
  if [ -e "$run/recovery_manifest.json" ]; then echo "$arm already done"; continue; fi
  for step in $steps; do
    echo "--- $arm $step ($(date +%H:%M))"
    case $step in
      pi_ready) pi_ready ;;
      make_checkpoints) make_checkpoints "$arm" ;;
      *) "experiments/run/$step.sh" "$arm" "$SEED" ;;
    esac
  done
done
echo "Validation pair done ($(date +%H:%M))."
