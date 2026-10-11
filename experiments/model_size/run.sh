#!/bin/bash
# Model-size benchmark: 3 repetitions x 4 widths in counterbalanced order, then one
# checkpoint-only block per width. Telemetry and the power logger run for the whole session;
# phases.log holds the host time of every block and the Pi-host clock offset.
#
#   experiments/model_size/run.sh    # ~3 h, asks for sudo
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
stay_awake "$@"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
ORDERS=("128 384 1280 4096" "4096 1280 384 128" "384 4096 128 1280")
OUT=results/msc/model_size
mkdir -p "$OUT"

edge() { ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP $*"; }
now() { python3 -c "import time; print(f'{time.time():.3f}')"; }
log() { echo "$(now) $1" >> "$OUT/phases.log"; }

block() {  # name, benchmark arguments...
  local name=$1; shift
  experiments/cooldown_gate.sh
  wait_for_meter "$OUT/power_fnb58.csv"
  log "start $name"
  # shellcheck disable=SC2016  # $PWD is expanded on the Pi
  edge docker compose -f docker-compose.edge.yml run --rm --no-deps -v '$PWD:/app' -w /app \
    --entrypoint python superexec-clientapp-4 -m experiments.model_size.benchmark "$@"
  log "end $name"
}

require_clean_pi
ssh "$PI" "mkdir -p master-thesis/experiments/model_size"
scp -q experiments/model_size/benchmark.py "$PI:master-thesis/experiments/model_size/"
scp -q edge_nodes/model.py "$PI:master-thesis/edge_nodes/"
scp -q docker-compose.edge.yml "$PI:master-thesis/"
python3 experiments/prepare_edge_data.py --seed 42 > /dev/null
scp -q data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json "$PI:master-thesis/data/.cache/msc/"

keep_sudo
start_monitor
experiments/meter_alarm.sh "$OUT/power_fnb58.csv" $$ &
alarm=$!
trap 'stop_logger; kill $alarm 2>/dev/null; stop_monitor' EXIT
start_logger "$OUT/power_fnb58.csv" "$OUT/logger.log"
host_t=$(now); pi_t=$(ssh "$PI" date +%s.%N)
echo "clock_offset_pi_minus_host $(echo "$pi_t - $host_t" | bc)" >> "$OUT/phases.log"

for rep in 1 2 3; do
  for w in ${ORDERS[$((rep - 1))]}; do
    [ -e "$OUT/model_size_w${w}_r${rep}.json" ] && continue
    drop_caches
    block "w$w r$rep" --width "$w" --rep "$rep"
    scp -q "$PI:/dev/shm/model_size_w${w}_r${rep}.json" "$OUT/"
  done
done

for w in 128 384 1280 4096; do
  [ -e "$OUT/model_size_ckptonly_w${w}.json" ] && continue
  block "ckptonly w$w" --width "$w" --rep 0 --ckpt-only
  scp -q "$PI:/dev/shm/model_size_ckptonly_w${w}.json" "$OUT/"
done

scp -q "$PI:/dev/shm/hardware_telemetry_*.csv" "$OUT/"
echo "Done. Results in $OUT"
