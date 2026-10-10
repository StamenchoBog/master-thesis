#!/bin/bash
# Model-size micro-benchmark (experiments/protocol.md, Addendum A), run from the host.
#
#   experiments/model_size/run.sh            # 3 repetitions x 4 widths, ~3 h, asks for sudo
#
# Per measurement: cooldown gate, benchmark on the Pi, copy its JSON back. Telemetry and
# the FNB58 logger run for the whole session; phases.log has the host timestamps of every
# block and the Pi-host clock offset, so the power log can be cut per block.
set -euo pipefail
# Keep the Mac awake: sleep can reset the FNB58.
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -ims "$0" "$@"; fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PI=admin@rasp5node.local
HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
# Counterbalanced order, so a slow drift over the session can't line up with model size.
ORDERS=("128 384 1280 4096" "4096 1280 384 128" "384 4096 128 1280")
WIDTHS_ALL=(128 384 1280 4096)
OUT=results/msc/model_size
mkdir -p "$OUT"

edge() { ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP $*"; }

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
now() { python3 -c "import time; print(f'{time.time():.3f}')"; }
log() { echo "$(now) $1" >> "$OUT/phases.log"; }

throttled=$(ssh "$PI" vcgencmd get_throttled)
if [ "$throttled" != "throttled=0x0" ]; then
  echo "$throttled, reboot the Pi and run this again." >&2; exit 1
fi

# The benchmark runs from the Pi's checkout, so it needs this script and the model with the
# width option (the campaign model at the default width of 128).
ssh "$PI" "mkdir -p master-thesis/experiments/model_size"
scp -q experiments/model_size/benchmark.py "$PI:master-thesis/experiments/model_size/"
scp -q edge_nodes/model.py "$PI:master-thesis/edge_nodes/"
scp -q docker-compose.edge.yml "$PI:master-thesis/"
python3 experiments/prepare_edge_data.py --seed 42 > /dev/null
scp -q data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json "$PI:master-thesis/data/.cache/msc/"

ssh "$PI" "sudo systemctl stop msc-monitor 2>/dev/null; sudo systemctl reset-failed msc-monitor 2>/dev/null; sudo systemd-run --unit=msc-monitor --working-directory=/home/admin /home/admin/msc-experiment/monitor.sh"
sudo -v
# Keep sudo fresh so logger restarts and the cleanup never wait for a password.
while kill -0 $$ 2>/dev/null; do sudo -n true; sleep 60; done 2>/dev/null &
experiments/meter_alarm.sh 'results/msc/model_size/power_fnb58.csv' $$ &
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
if [ "$(wc -l < "$OUT/power_fnb58.csv")" -lt 100 ]; then
  echo "The power logger isn't writing samples. Replug the FNB58 and run this again." >&2; exit 1
fi
host_t=$(now); pi_t=$(ssh "$PI" date +%s.%N)
echo "clock_offset_pi_minus_host $(echo "$pi_t - $host_t" | bc)" >> "$OUT/phases.log"

for rep in 1 2 3; do
  order=${ORDERS[$((rep - 1))]}
  for w in $order; do
    if [ -e "$OUT/model_size_w${w}_r${rep}.json" ]; then continue; fi
    ssh "$PI" "sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
    experiments/cooldown_gate.sh
    wait_for_meter "$OUT/power_fnb58.csv"
    log "start w$w r$rep"
    # shellcheck disable=SC2016  # $PWD is expanded on the Pi
    edge docker compose -f docker-compose.edge.yml run --rm --no-deps -v '$PWD:/app' -w /app \
      --entrypoint python superexec-clientapp-4 -m experiments.model_size.benchmark \
      --width "$w" --rep "$rep"
    log "end w$w r$rep"
    scp -q "$PI:/dev/shm/model_size_w${w}_r${rep}.json" "$OUT/"
  done
done

# Checkpoint-only power block per size (fixed before the session), so checkpoint energy is
# measured even if the meter dropped during a block above.
for w in "${WIDTHS_ALL[@]}"; do
  if [ -e "$OUT/model_size_ckptonly_w${w}.json" ]; then continue; fi
  experiments/cooldown_gate.sh
  wait_for_meter "$OUT/power_fnb58.csv"
  log "start ckptonly w$w"
  # shellcheck disable=SC2016  # $PWD is expanded on the Pi
  edge docker compose -f docker-compose.edge.yml run --rm --no-deps -v '$PWD:/app' -w /app \
    --entrypoint python superexec-clientapp-4 -m experiments.model_size.benchmark \
    --width "$w" --rep 0 --ckpt-only
  log "end ckptonly w$w"
  scp -q "$PI:/dev/shm/model_size_ckptonly_w${w}.json" "$OUT/"
done

scp -q "$PI:/dev/shm/hardware_telemetry_*.csv" "$OUT/"
echo "Done. Results in $OUT"
