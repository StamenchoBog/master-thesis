#!/bin/bash
# Phase 3: recovery on the Pi. This is the measured window.
source "$(dirname "$0")/common.sh"

if [ "$ARM" = naive ]; then module=naive_retrain; else module=sisa_recover; fi

edge docker compose -f docker-compose.edge.yml stop superexec-clientapp-4
mark phase3
# shellcheck disable=SC2016  # $PWD is expanded on the Pi
edge docker compose -f docker-compose.edge.yml run --rm -v '$PWD:/app' -w /app \
  --entrypoint python superexec-clientapp-4 -m "edge_nodes.$module"
mark idle
