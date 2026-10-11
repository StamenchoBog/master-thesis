#!/bin/bash
# Shared helpers for the host-side scripts. Source it after cd to the repo root.
# shellcheck disable=SC2034  # PI is used by the scripts that source this file
PI=admin@rasp5node.local

# Usage: stay_awake "$@" (a sleeping Mac can reset the FNB58)
stay_awake() {
  [ -n "${CAFFEINATED:-}" ] || exec env CAFFEINATED=1 caffeinate -ims "$0" "$@"
}

# Ask for sudo once and keep it valid while the calling script runs.
keep_sudo() {
  sudo -v
  while kill -0 $$ 2>/dev/null; do sudo -n true; sleep 60; done 2>/dev/null &
}

notify_replug() {
  echo "$(date +%H:%M:%S) power meter not streaming, replug the FNB58 USB cable" >&2
  osascript -e 'display notification "Replug the FNB58 USB cable" with title "Power meter stopped" sound name "Glass"' 2>/dev/null || true
  afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &
}

# The newest line must be a sample at most 15 s old. The file's mtime is not enough: every
# logger restart writes a header, even when no samples follow.
meter_fresh() {
  local t
  t=$(tail -1 "$1" 2>/dev/null | awk '{print $1}')
  [[ $t =~ ^[0-9]+\.[0-9]+$ ]] || return 1
  python3 -c "import sys, time; sys.exit(0 if time.time() - $t < 15 else 1)"
}

wait_for_meter() {
  local last=0
  until meter_fresh "$1"; do
    if [ $(( $(date +%s) - last )) -ge 60 ]; then notify_replug; last=$(date +%s); fi
    sleep 5
  done
}

# Usage: start_logger CSV LOG
# On macOS only a new process gets a lost meter back, hence the restart loop.
start_logger() {
  local py
  py=$(command -v python3)
  # shellcheck disable=SC2024  # the log files should belong to us, not root
  sudo bash -c "while true; do '$py' -u experiments/fnirsi_logger.py; sleep 2; done" > "$1" 2> "$2" &
  for _ in $(seq 1 18); do
    sleep 5
    [ "$(wc -l < "$1")" -ge 100 ] && break
  done
  wait_for_meter "$1"
}

stop_logger() {
  sudo pkill -f "[w]hile true; do .*fnirsi_logger" 2>/dev/null || true
  sudo pkill -f "[f]nirsi_logger.py" 2>/dev/null || true
}

# `systemctl start` silently does nothing on this unit, and a unit left by an aborted run
# blocks systemd-run, hence stop + reset-failed + systemd-run.
start_monitor() {
  ssh "$PI" "sudo systemctl stop msc-monitor 2>/dev/null; sudo systemctl reset-failed msc-monitor 2>/dev/null; sudo systemd-run --unit=msc-monitor --working-directory=/home/admin /home/admin/msc-experiment/monitor.sh"
}

stop_monitor() {
  ssh "$PI" "sudo systemctl stop msc-monitor" || true
}

pi_temp() {
  ssh "$PI" vcgencmd measure_temp | grep -o '[0-9]*' | head -1
}

# Reboot until the throttle status is clean. Cool to 70 C first: booting heats the Pi by a few
# degrees, which from 72 C can reach the 80 C soft limit. A boot can also catch an under-voltage.
pi_ready() {
  local attempt status
  for attempt in 1 2 3 4 5 6; do
    echo "$(date +%H:%M) cooling the Pi to 70 C before reboot $attempt"
    until [ "$(pi_temp)" -le 70 ]; do sleep 30; done
    ssh "$PI" "sudo reboot" 2>/dev/null || true
    sleep 30
    until ssh -o ConnectTimeout=5 "$PI" true 2>/dev/null; do sleep 10; done
    sleep 25
    status=$(ssh "$PI" vcgencmd get_throttled)
    [ "$status" = "throttled=0x0" ] && return 0
    echo "Pi status $status after reboot $attempt, retrying" >&2
  done
  echo "Pi did not come up with a clean status; check the power and rerun." >&2
  exit 1
}

require_clean_pi() {
  local status
  status=$(ssh "$PI" vcgencmd get_throttled)
  [ "$status" = "throttled=0x0" ] || { echo "$status, reboot the Pi and run this again." >&2; exit 1; }
}

drop_caches() {
  ssh "$PI" "sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
}
