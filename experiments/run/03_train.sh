#!/bin/bash
# Phase 1: 10 federated rounds with the Pi's poisoned labels active.
# shellcheck disable=SC2086  # $PROFILE is empty or "--profile wan"
source "$(dirname "$0")/common.sh"

mark phase1
NUM_ROUNDS=10 RESULTS_SUFFIX=_p1 docker compose -f docker-compose.host.yml $PROFILE up -d
edge POISON_MODE=flip docker compose -f docker-compose.edge.yml up -d
docker compose -f docker-compose.host.yml run --rm runner
mark idle

mv results/msc/global_checkpoints "$RUN/phase1_checkpoints"
