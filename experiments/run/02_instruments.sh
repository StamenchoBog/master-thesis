#!/bin/bash
# Make sure the Pi starts clean and cool, then start telemetry and the power logger.
source "$(dirname "$0")/common.sh"

throttled=$(ssh "$PI" vcgencmd get_throttled)
if [ "$throttled" != "throttled=0x0" ]; then
  echo "$throttled, reboot the Pi (ssh $PI sudo reboot) and run this again." >&2
  exit 1
fi

ssh "$PI" "sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
experiments/cooldown_gate.sh

# Always reset-failed + systemd-run: `systemctl start` on this unit silently does nothing.
ssh "$PI" "sudo systemctl reset-failed msc-monitor 2>/dev/null; sudo systemd-run --unit=msc-monitor --working-directory=/home/admin /home/admin/msc-experiment/monitor.sh"
ssh "$PI" "systemctl is-active msc-monitor"

sudo -v
# shellcheck disable=SC2024  # the log file should belong to us, not root
sudo python3 -u experiments/fnirsi_logger.py > "$RUN/power_fnb58.csv" &
echo $! > "$RUN/.logger.pid"
