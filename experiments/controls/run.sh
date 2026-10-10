#!/bin/bash
# Control runs for the utility analysis (H5), same hybrid topology as the campaign with the
# Pi as node 4. Nothing physical is measured, so no power meter, telemetry or cooldown.
#
#   clean_naive  - no poison, standard client on the Pi, 15 rounds (reference, arm A)
#   clean_sisa   - no poison, SISA client on the Pi, 15 rounds (reference, arm B)
#   poisoned_sisa - poisoned Phase 1 (10 rounds) + SISA recovery on the Pi, keeping the
#                  constituent checkpoints for the ensemble-vs-average comparison
#
# 15 rounds = Phase 1 (10) + Phase 4 (5) of the main protocol.
# Usage: experiments/controls/run.sh SEED [REF_ROUNDS] [P1_ROUNDS]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SEED=${1:?usage: $0 SEED [REF_ROUNDS] [P1_ROUNDS]}
REF_ROUNDS=${2:-15}
P1_ROUNDS=${3:-10}
OUT=${OUT:-results/msc/controls}
PI=admin@rasp5node.local
PI_CKPT=msc-experiment/checkpoints
HOST_IP=${HOST_IP:-$(ipconfig getifaddr en0)}
export SEED

edge() {  # CLIENT_MODE POISON_MODE command...
  local mode=$1 poison=$2; shift 2
  ssh "$PI" "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED CLIENT_MODE=$mode POISON_MODE=$poison $*"
}
host() { docker compose -f docker-compose.host.yml "$@"; }

stop_all() {
  host down > /dev/null 2>&1 || true
  edge standard off docker compose -f docker-compose.edge.yml down > /dev/null 2>&1 || true
}

federate() {  # name client_mode poison_mode rounds
  local dir="$OUT/$1_seed$SEED"
  if [ -e "$dir/global_checkpoints" ]; then echo "[$1] seed $SEED already done, skipping"; return 1; fi
  mkdir -p "$dir"
  ssh "$PI" "sudo rm -rf $PI_CKPT/* /dev/shm/sisa_timings.jsonl /dev/shm/recovery_manifest.json"
  rm -rf results/msc/global_checkpoints
  echo "[$1] seed $SEED: $4 rounds, CLIENT_MODE=$2 POISON_MODE=$3 ($(date +%H:%M))"
  NUM_ROUNDS=$4 RESULTS_SUFFIX=_ref host up -d > /dev/null 2>&1
  edge "$2" "$3" docker compose -f docker-compose.edge.yml up -d > /dev/null 2>&1
  host run --rm runner > "$dir/runner.log" 2>&1
  stop_all
  mv results/msc/fedavg_ref.json "$dir/results.json"
  mv results/msc/global_checkpoints "$dir/global_checkpoints"
}

fetch_constituents() {  # dest_dir round
  mkdir -p "$1"
  for s in 0 1 2 3 4; do scp -q "$PI:$PI_CKPT/shard${s}_round$2_slice4.pt" "$1/"; done
}

# Data for this seed; the Pi gets its partition and must report the same seed back.
python3 experiments/prepare_edge_data.py --seed "$SEED" > /dev/null
scp -q data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json "$PI:master-thesis/data/.cache/msc/"
pi_seed=$(ssh "$PI" "python3 -c \"import json; print(json.load(open('master-thesis/data/.cache/msc/manifest.json'))['seed'])\"")
[ "$pi_seed" = "$SEED" ] || { echo "Pi has the cache for seed $pi_seed, expected $SEED" >&2; exit 1; }
stop_all

federate clean_naive standard off "$REF_ROUNDS" || true

if federate clean_sisa sisa off "$REF_ROUNDS"; then
  fetch_constituents "$OUT/clean_sisa_seed$SEED/constituents" "$REF_ROUNDS"
fi

if federate poisoned_sisa sisa flip "$P1_ROUNDS"; then
  dir="$OUT/poisoned_sisa_seed$SEED"
  # Recovery overwrites the affected shard's checkpoints, so keep the Phase-1 ones first.
  fetch_constituents "$dir/phase1_constituents" "$P1_ROUNDS"
  echo "[poisoned_sisa] SISA recovery on the Pi"
  # shellcheck disable=SC2016  # $PWD is expanded on the Pi
  edge sisa off docker compose -f docker-compose.edge.yml run --rm --no-deps -v '$PWD:/app' -w /app \
    -e NUM_ROUNDS="$P1_ROUNDS" --entrypoint python superexec-clientapp-4 -m edge_nodes.sisa_recover \
    > "$dir/recover.log" 2>&1
  fetch_constituents "$dir/constituents" "$P1_ROUNDS"
  scp -q "$PI:$PI_CKPT/recovered_model.pt" "$PI:/dev/shm/recovery_manifest.json" "$dir/"
  stop_all
fi
echo "seed $SEED done ($(date +%H:%M))"
