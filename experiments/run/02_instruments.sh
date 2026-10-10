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

# Always stop + reset-failed + systemd-run: `systemctl start` on this unit silently does nothing,
# and a unit left loaded by an aborted attempt would block systemd-run.
ssh "$PI" "sudo systemctl stop msc-monitor 2>/dev/null; sudo systemctl reset-failed msc-monitor 2>/dev/null; sudo systemd-run --unit=msc-monitor --working-directory=/home/admin /home/admin/msc-experiment/monitor.sh"
ssh "$PI" "systemctl is-active msc-monitor"

sudo -v
# Restart loop: a fresh logger after every loss of the meter (see experiments/protocol.md).
py=$(command -v python3)
# shellcheck disable=SC2024  # the log files should belong to us, not root
sudo bash -c "while true; do '$py' -u experiments/fnirsi_logger.py; sleep 2; done" \
  > "$RUN/power_fnb58.csv" 2> "$RUN/logger.log" &
echo $! > "$RUN/.logger.pid"

# Don't start Phase 1 unless the meter is actually streaming. Give the restart loop up to
# 90 s: right after a previous logger is stopped the meter sometimes sends nothing at first.
for _ in $(seq 1 18); do
  sleep 5
  [ "$(wc -l < "$RUN/power_fnb58.csv")" -ge 100 ] && break
done
if [ "$(wc -l < "$RUN/power_fnb58.csv")" -lt 100 ]; then
  echo "The power logger isn't writing samples. Replug the FNB58 USB cable and run this step again." >&2
  sudo pkill -f "while true; do .*fnirsi_logger" 2>/dev/null || true
  sudo pkill -f fnirsi_logger.py 2>/dev/null || true
  rm -f "$RUN/.logger.pid" "$RUN/power_fnb58.csv"
  ssh "$PI" "sudo systemctl stop msc-monitor; sudo rm -f /dev/shm/hardware_telemetry_*.csv"
  exit 1
fi
