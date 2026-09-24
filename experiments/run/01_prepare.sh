#!/bin/bash
# Build the data for this seed, copy the Pi's partition over and reset the Pi.
source "$(dirname "$0")/common.sh"

if [ -e "$RUN/phases.log" ]; then
  echo "$RUN already has a run in it, remove it first." >&2
  exit 1
fi

python3 experiments/prepare_edge_data.py --seed "$SEED"
scp data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json "$PI:master-thesis/data/.cache/msc/"

# A cache from another seed scatters the poison and SISA ends up retraining everything.
pi_seed=$(ssh "$PI" "python3 -c \"import json; print(json.load(open('master-thesis/data/.cache/msc/manifest.json'))['seed'])\"")
if [ "$pi_seed" != "$SEED" ]; then
  echo "The Pi has the cache for seed $pi_seed, expected $SEED." >&2
  exit 1
fi

ssh "$PI" "sudo rm -rf msc-experiment/checkpoints/* /dev/shm/sisa_timings.jsonl /dev/shm/recovery_manifest.json /dev/shm/hardware_telemetry_*.csv"
rm -rf results/msc/global_checkpoints
mkdir -p "$RUN"
