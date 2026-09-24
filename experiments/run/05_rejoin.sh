#!/bin/bash
# Phase 4: 5 rejoin rounds, starting from the Phase-1 global model.
# shellcheck disable=SC2086  # $PROFILE is empty or "--profile wan"
source "$(dirname "$0")/common.sh"

cp "$RUN/phase1_checkpoints/round_10.npz" results/msc/resume_from.npz
NUM_ROUNDS=5 RESULTS_SUFFIX=_p4 INIT_FROM_CHECKPOINT=/results/resume_from.npz \
  docker compose -f docker-compose.host.yml $PROFILE up -d superexec-serverapp
edge POISON_MODE=drop docker compose -f docker-compose.edge.yml up -d superexec-clientapp-4

mark phase4
docker compose -f docker-compose.host.yml run --rm runner
mark "done"
