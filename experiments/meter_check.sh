#!/bin/bash
# FNB58 reliability check: stream while the Pi alternates 2 min idle / 3 min full load,
# then list every gap in the power log and the logger's reconnects.
#
#   experiments/meter_check.sh                # 45 min, needs sudo for the logger
#   MINUTES=180 experiments/meter_check.sh    # longer check after a change to the setup
set -euo pipefail
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -ims "$0" "$@"; fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PI=admin@rasp5node.local
OUT=data/.cache/meter_check_$(date +%Y%m%d_%H%M)
MINUTES=${MINUTES:-45}
mkdir -p "$OUT"

sudo -v
# Keep the sudo timestamp fresh so the cleanup at the end never waits for a password.
while kill -0 $$ 2>/dev/null; do sudo -n true; sleep 60; done 2>/dev/null &
# shellcheck disable=SC2024  # the log files should belong to us, not root
py=$(command -v python3)
# Same restart loop as the runs: a fresh logger process after every loss of the meter.
sudo bash -c "while true; do '$py' -u experiments/fnirsi_logger.py; sleep 2; done" \
  > "$OUT/power.csv" 2> "$OUT/logger.log" &
logger_pid=$!
trap 'sudo pkill -f "while true; do .*fnirsi_logger" 2>/dev/null; sudo pkill -f fnirsi_logger.py 2>/dev/null; sudo kill $logger_pid 2>/dev/null || true; ssh "$PI" "pkill -f \"[w]hile True\"" || true' EXIT
for _ in $(seq 1 18); do  # up to 90 s for the restart loop to get the meter streaming
  sleep 5
  [ "$(wc -l < "$OUT/power.csv")" -ge 100 ] && break
done
[ "$(wc -l < "$OUT/power.csv")" -ge 100 ] || { echo "Logger not streaming. Replug the FNB58 and run this again." >&2; exit 1; }

pmset -g ps | head -1 > "$OUT/mac_power_source.txt"
end=$(( $(date +%s) + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "$(date +%H:%M:%S) idle"; sleep 120
  echo "$(date +%H:%M:%S) load"
  ssh "$PI" 'for c in 1 2 3 4; do timeout 180 python3 -c "while True: pass" & done; wait'
done

python3 - "$OUT" <<'E'
import sys
out = sys.argv[1]
ts = []
for line in open(f"{out}/power.csv"):
    p = line.split()
    try:
        ts.append(float(p[0]))
    except (ValueError, IndexError):
        pass
gaps = [(a, b - a) for a, b in zip(ts, ts[1:]) if b - a > 1.0]
span = ts[-1] - ts[0]
lost = sum(g for _, g in gaps)
print(f"{len(ts)} samples over {span / 60:.1f} min, coverage {1 - lost / span:.2%}")
for a, g in gaps:
    import time
    print(f"  gap at {time.strftime('%H:%M:%S', time.localtime(a))}: {g:.1f} s")
print("logger events:")
for line in open(f"{out}/logger.log"):
    if "lost" in line or "reconnected" in line or "silent" in line or "Aborting" in line or "fresh restart" in line:
        print("  " + line.strip())
E
echo "Saved in $OUT"
