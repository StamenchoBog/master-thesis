#!/bin/bash
# Stop everything and move the run's files into its results folder.
# shellcheck disable=SC2086  # $PROFILE is empty or "--profile wan"
source "$(dirname "$0")/common.sh"

stop_logger
edge docker compose -f docker-compose.edge.yml down
stop_monitor
docker compose -f docker-compose.host.yml $PROFILE down

scp "$PI:/dev/shm/hardware_telemetry_*.csv" "$PI:/dev/shm/recovery_manifest.json" \
  "$PI:msc-experiment/checkpoints/recovered_model.pt" "$RUN/"
if [ "$ARM" = sisa ]; then
  scp "$PI:/dev/shm/sisa_timings.jsonl" "$RUN/" 2>/dev/null || true
fi

# A recovery-only run (validate.sh with RECOVERY_ONLY) has no Phase 1 or 4.
[ -e results/msc/fedavg_p1.json ] && mv results/msc/fedavg_p1.json "$RUN/results_phase1.json"
[ -e results/msc/fedavg_p4.json ] && mv results/msc/fedavg_p4.json "$RUN/results_phase4.json"
[ -e results/msc/global_checkpoints ] && mv results/msc/global_checkpoints "$RUN/phase4_checkpoints"
rm -f results/msc/resume_from.npz
