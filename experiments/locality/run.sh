#!/bin/bash
# Poison-placement study: one clean SISA Phase 1 on the Pi, its 250 checkpoints set aside,
# then 30 recoveries (placements from experiments/locality/placements.py), each from a fresh
# copy of those checkpoints.
#
#   experiments/locality/run.sh    # ~3 h, asks for sudo
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
stay_awake "$@"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
SEED=42
OUT=results/msc/locality
CKPT=msc-experiment/checkpoints
BASE=msc-experiment/locality_base
# Counterbalanced; "naive" is naive retraining on the k1_l3 placement.
ORDERS=(
  "k1_l3_full k1_l4 k1_l3 k1_l2 k1_l1 k1_l0 k2_l3 k3_l3 k5_l3 naive"
  "naive k5_l3 k3_l3 k2_l3 k1_l0 k1_l1 k1_l2 k1_l3 k1_l4 k1_l3_full"
  "k1_l2 k5_l3 k1_l4 naive k1_l0 k3_l3 k1_l3_full k1_l1 k2_l3 k1_l3"
)
mkdir -p "$OUT"
export SEED MODEL_WIDTH=128

edge() { ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED MODEL_WIDTH=128 $*"; }
now() { python3 -c "import time; print(f'{time.time():.3f}')"; }
mark() { echo "$(now) $1" >> "$OUT/phases.log"; ssh "$PI" "echo $1 > /dev/shm/run_marker"; }

keep_sudo
pi_ready

python3 experiments/prepare_edge_data.py --seed "$SEED" > /dev/null
python3 experiments/locality/placements.py > "$OUT/placements.txt"
scp -q data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json "$PI:master-thesis/data/.cache/msc/"
ssh "$PI" "mkdir -p master-thesis/data/.cache/locality"
scp -rq data/.cache/locality/* "$PI:master-thesis/data/.cache/locality/"
scp -q edge_nodes/model.py "$PI:master-thesis/edge_nodes/"
scp -q docker-compose.edge.yml "$PI:master-thesis/"

start_monitor
experiments/meter_alarm.sh "$OUT/power_fnb58.csv" $$ &
alarm=$!
trap 'stop_logger; kill $alarm 2>/dev/null; stop_monitor' EXIT
start_logger "$OUT/power_fnb58.csv" "$OUT/logger.log"
host_t=$(now); pi_t=$(ssh "$PI" date +%s.%N)
echo "clock_offset_pi_minus_host $(echo "$pi_t - $host_t" | bc)" >> "$OUT/phases.log"

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

for rep in 1 2 3; do
  for cfg in ${ORDERS[$((rep - 1))]}; do
    [ -e "$OUT/${cfg}_r${rep}.json" ] && continue
    if [ "$cfg" = naive ]; then module=naive_retrain; place=k1_l3; else module=sisa_recover; place=$cfg; fi
    # The container runs as uid 49999 and must be able to overwrite the restored files.
    ssh "$PI" "sudo rm -rf $CKPT && sudo cp -a $BASE $CKPT && sudo chown -R 49999:49999 $CKPT && sudo rm -f /dev/shm/recovery_manifest.json"
    drop_caches
    experiments/cooldown_gate.sh
    wait_for_meter "$OUT/power_fnb58.csv"
    mark "recover_${cfg}_r${rep}"
    # shellcheck disable=SC2016  # $PWD is expanded on the Pi
    edge docker compose -f docker-compose.edge.yml run --rm --no-deps -v '$PWD:/app' -w /app \
      -e CACHE_DIR=/data/.cache/locality/"$place" -e NUM_ROUNDS=10 \
      --entrypoint python superexec-clientapp-4 -m "edge_nodes.$module" > "$OUT/${cfg}_r${rep}.log" 2>&1
    mark idle
    scp -q "$PI:/dev/shm/recovery_manifest.json" "$OUT/${cfg}_r${rep}.json"
  done
done

scp -q "$PI:/dev/shm/hardware_telemetry_*.csv" "$OUT/"
echo "Done. Results in $OUT"
