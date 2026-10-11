#!/bin/bash
# While PARENT_PID runs: if the logger is running but the newest power log matching PATTERN
# has not changed for 60 s, notify and play a sound once a minute.
#
#   experiments/meter_alarm.sh 'results/msc/model_size/validation/*/power_fnb58.csv' $$ &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
pattern=$1
parent=$2
since=$(mktemp "${TMPDIR:-/tmp}/meter_alarm.XXXX")  # only logs written after this count
trap 'rm -f "$since"' EXIT
while kill -0 "$parent" 2>/dev/null; do
  if pgrep -f "[f]nirsi_logger\.py" >/dev/null; then
    # shellcheck disable=SC2086,SC2012  # the pattern is meant to glob
    f=$(ls -t $pattern 2>/dev/null | head -1)
    if [ -n "$f" ] && [ "$f" -nt "$since" ] && [ $(( $(date +%s) - $(stat -f %m "$f") )) -gt 60 ]; then
      notify_replug
      sleep 50
    fi
  fi
  sleep 10
done
