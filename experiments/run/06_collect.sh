#!/bin/bash
# Stop everything and move the run's files into its results folder.
# shellcheck disable=SC2086  # $PROFILE is empty or "--profile wan"
source "$(dirname "$0")/common.sh"

sudo kill "$(cat "$RUN/.logger.pid")"
rm "$RUN/.logger.pid"
edge docker compose -f docker-compose.edge.yml down
ssh "$PI" "sudo systemctl stop msc-monitor"
docker compose -f docker-compose.host.yml $PROFILE down

scp "$PI:/dev/shm/hardware_telemetry_*.csv" "$PI:/dev/shm/recovery_manifest.json" \
  "$PI:msc-experiment/checkpoints/recovered_model.pt" "$RUN/"
if [ "$ARM" = sisa ]; then
  scp "$PI:/dev/shm/sisa_timings.jsonl" "$RUN/"
fi

mv results/msc/fedavg_p1.json "$RUN/results_phase1.json"
mv results/msc/fedavg_p4.json "$RUN/results_phase4.json"
mv results/msc/global_checkpoints "$RUN/phase4_checkpoints"
rm -f results/msc/resume_from.npz
