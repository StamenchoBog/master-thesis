#!/bin/bash
# Background alarm while a session runs: if the power logger is running but this session's
# newest power log has not changed for 60 s, notify and play a sound every minute.
#   experiments/meter_alarm.sh 'results/msc/model_size/validation/*/power_fnb58.csv' $$ &
# The second argument is the session's PID; the alarm stops when that process is gone.
pattern=$1
parent=$2
since=$(mktemp "${TMPDIR:-/tmp}/meter_alarm.XXXX")  # only power logs written after this count
trap 'rm -f "$since"' EXIT
while kill -0 "$parent" 2>/dev/null; do
  # [f] keeps pgrep from matching its own command line or a watcher that quotes the pattern.
  if pgrep -f "[f]nirsi_logger\.py" >/dev/null; then
    # shellcheck disable=SC2086  # the pattern is meant to glob
    f=$(ls -t $pattern 2>/dev/null | head -1)
    if [ -n "$f" ] && [ "$f" -nt "$since" ] && [ $(( $(date +%s) - $(stat -f %m "$f") )) -gt 60 ]; then
      osascript -e 'display notification "Replug the FNB58 USB cable" with title "Power meter stopped" sound name "Glass"' 2>/dev/null || true
      afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
      sleep 50
    fi
  fi
  sleep 10
done
