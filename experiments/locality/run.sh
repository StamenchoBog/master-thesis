#!/bin/bash
# Poisoning-locality sweep (experiments/protocol.md, Addendum B), run from the host.
#
#   experiments/locality/run.sh            # SISA Phase 1, then 30 recoveries, ~3 h, asks for sudo
#
# 1. One SISA Phase 1 on the Pi (10 rounds, no poison, seed 42); its 250 checkpoints are
#    copied aside.
# 2. Each recovery restores that copy, waits for the cooldown gate and runs sisa_recover
#    (or naive_retrain) on one placement from experiments/locality/placements.py.
set -euo pipefail
# Keep the Mac awake: sleep can reset the FNB58.
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -ims "$0" "$@"; fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PI=admin@rasp5node.local
HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
SEED=42
OUT=results/msc/locality
CKPT=msc-experiment/checkpoints
BASE=msc-experiment/locality_base
# Counterbalanced order (by expected replay, reversed, interleaved); "naive" is naive
# retraining on the k1_l3 placement.
ORDERS=(
  "k1_l3_full k1_l4 k1_l3 k1_l2 k1_l1 k1_l0 k2_l3 k3_l3 k5_l3 naive"
  "naive k5_l3 k3_l3 k2_l3 k1_l0 k1_l1 k1_l2 k1_l3 k1_l4 k1_l3_full"
  "k1_l2 k5_l3 k1_l4 naive k1_l0 k3_l3 k1_l3_full k1_l1 k2_l3 k1_l3"
)
mkdir -p "$OUT"
export SEED MODEL_WIDTH=128

edge() { ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED MODEL_WIDTH=128 $*"; }
now() { python3 -c "import time; print(f'{time.time():.3f}')"; }
log() { echo "$(now) $1" >> "$OUT/phases.log"; }
mark() { log "$1"; ssh "$PI" "echo $1 > /dev/shm/run_marker"; }

wait_for_meter() {  # same as experiments/run/common.sh
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

# sudo is asked once, here; keep it fresh until this script exits so logger restarts and
# the cleanup never wait for a password.
sudo -v
while kill -0 $$ 2>/dev/null; do sudo -n true; sleep 60; done 2>/dev/null &

# Cool to 72 C, then reboot until the throttle status is clean (as in validate.sh).
echo "$(date +%H:%M) waiting for the Pi to cool to 72 C, then rebooting it (can take 30 min)"
until [ "$(ssh "$PI" vcgencmd measure_temp | grep -o '[0-9]*' | head -1)" -le 72 ]; do sleep 30; done
for attempt in 1 2 3 4; do
  [ "$attempt" = 4 ] && { echo "Pi did not come up with a clean status; check the power and rerun." >&2; exit 1; }
  ssh "$PI" "sudo reboot" 2>/dev/null || true
  sleep 30
  until ssh -o ConnectTimeout=5 "$PI" true 2>/dev/null; do sleep 10; done
  sleep 25
  status=$(ssh "$PI" vcgencmd get_throttled)
  [ "$status" = "throttled=0x0" ] && break
  echo "Pi status $status after reboot $attempt, retrying" >&2
done

# Data: seed-42 caches for the federation, the nine placements for the recoveries.
python3 experiments/prepare_edge_data.py --seed "$SEED" > /dev/null
python3 experiments/locality/placements.py > "$OUT/placements.txt"
scp -q data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json "$PI:master-thesis/data/.cache/msc/"
ssh "$PI" "mkdir -p master-thesis/data/.cache/locality"
scp -rq data/.cache/locality/* "$PI:master-thesis/data/.cache/locality/"
scp -q edge_nodes/model.py "$PI:master-thesis/edge_nodes/"
scp -q docker-compose.edge.yml "$PI:master-thesis/"

ssh "$PI" "sudo systemctl stop msc-monitor 2>/dev/null; sudo systemctl reset-failed msc-monitor 2>/dev/null; sudo systemd-run --unit=msc-monitor --working-directory=/home/admin /home/admin/msc-experiment/monitor.sh"
experiments/meter_alarm.sh 'results/msc/locality/power_fnb58.csv' $$ &
alarm=$!
# A fresh logger process after every loss of the meter (on macOS only a new process gets
# it back); samples go to one file, the logger's messages to logger.log.
py=$(command -v python3)
# shellcheck disable=SC2024  # the log files should belong to us, not root
sudo bash -c "while true; do '$py' -u experiments/fnirsi_logger.py; sleep 2; done" \
  > "$OUT/power_fnb58.csv" 2> "$OUT/logger.log" &
logger_pid=$!
trap 'sudo pkill -f "while true; do .*fnirsi_logger" 2>/dev/null; sudo pkill -f fnirsi_logger.py 2>/dev/null; sudo kill $logger_pid 2>/dev/null; kill $alarm 2>/dev/null; ssh "$PI" "sudo systemctl stop msc-monitor" || true' EXIT
for _ in $(seq 1 18); do  # up to 90 s for the restart loop to get the meter streaming
  sleep 5
  [ "$(wc -l < "$OUT/power_fnb58.csv")" -ge 100 ] && break
done
wait_for_meter "$OUT/power_fnb58.csv"  # after that, ask for a replug until it streams
host_t=$(now); pi_t=$(ssh "$PI" date +%s.%N)
echo "clock_offset_pi_minus_host $(echo "$pi_t - $host_t" | bc)" >> "$OUT/phases.log"

# 1. SISA Phase 1 without poison, unless its checkpoints are already set aside.
if ! ssh "$PI" "test -e $BASE/shard4_round10_slice4.pt"; then
  ssh "$PI" "sudo rm -rf $CKPT/* $BASE"
  mark phase1
  NUM_ROUNDS=10 RESULTS_SUFFIX=_locality docker compose -f docker-compose.host.yml up -d > /dev/null 2>&1
  edge CLIENT_MODE=sisa POISON_MODE=off docker compose -f docker-compose.edge.yml up -d > /dev/null 2>&1
  docker compose -f docker-compose.host.yml run --rm runner > "$OUT/phase1_runner.log" 2>&1
  mark idle
  docker compose -f docker-compose.host.yml down > /dev/null 2>&1
  edge docker compose -f docker-compose.edge.yml down > /dev/null 2>&1
  rm -rf results/msc/global_checkpoints
  mv results/msc/fedavg_locality.json "$OUT/phase1_results.json"
  ssh "$PI" "sudo cp -a $CKPT $BASE"
fi

# 2. Recoveries.
for rep in 1 2 3; do
  for cfg in ${ORDERS[$((rep - 1))]}; do
    if [ -e "$OUT/${cfg}_r${rep}.json" ]; then continue; fi
    if [ "$cfg" = naive ]; then module=naive_retrain; place=k1_l3; else module=sisa_recover; place=$cfg; fi
    # The container runs as app (uid 49999) and must be able to overwrite the restored files.
    ssh "$PI" "sudo rm -rf $CKPT && sudo cp -a $BASE $CKPT && sudo chown -R 49999:49999 $CKPT && sudo rm -f /dev/shm/recovery_manifest.json; sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
    experiments/cooldown_gate.sh
    wait_for_meter "$OUT/power_fnb58.csv"
    mark "recover_${cfg}_r${rep}"
    # shellcheck disable=SC2016  # $PWD is expanded on the Pi
    edge docker compose -f docker-compose.edge.yml run --rm --no-deps -v '$PWD:/app' -w /app \
      -e CACHE_DIR=/data/.cache/locality/$place -e NUM_ROUNDS=10 \
      --entrypoint python superexec-clientapp-4 -m "edge_nodes.$module" > "$OUT/${cfg}_r${rep}.log" 2>&1
    mark idle
    scp -q "$PI:/dev/shm/recovery_manifest.json" "$OUT/${cfg}_r${rep}.json"
  done
done

scp -q "$PI:/dev/shm/hardware_telemetry_*.csv" "$OUT/"
echo "Done. Results in $OUT"
