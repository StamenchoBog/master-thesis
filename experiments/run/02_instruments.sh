#!/bin/bash
# Start from a clean, cool Pi, then start telemetry and the power logger.
source "$(dirname "$0")/common.sh"

require_clean_pi
drop_caches
experiments/cooldown_gate.sh
start_monitor
ssh "$PI" "systemctl is-active msc-monitor"
sudo -v
start_logger "$RUN/power_fnb58.csv" "$RUN/logger.log"
