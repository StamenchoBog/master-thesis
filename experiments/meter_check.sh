#!/bin/bash
# FNB58 check: log power while the Pi alternates 2 min idle / 3 min full load, then list the
# gaps in the log and the logger's restarts.
#
#   experiments/meter_check.sh                # 45 min, asks for sudo
#   MINUTES=180 experiments/meter_check.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
stay_awake "$@"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT=data/.cache/meter_check_$(date +%Y%m%d_%H%M)
MINUTES=${MINUTES:-45}
mkdir -p "$OUT"

keep_sudo
trap 'stop_logger; ssh "$PI" "pkill -f \"[w]hile True\"" || true' EXIT
start_logger "$OUT/power.csv" "$OUT/logger.log"

end=$(( $(date +%s) + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "$(date +%H:%M:%S) idle"; sleep 120
  echo "$(date +%H:%M:%S) load"
  ssh "$PI" 'for c in 1 2 3 4; do timeout 180 python3 -c "while True: pass" & done; wait'
done

python3 - "$OUT" <<'PY'
import sys
import time

out = sys.argv[1]
ts = []
for line in open(f"{out}/power.csv"):
    try:
        ts.append(float(line.split()[0]))
    except (ValueError, IndexError):
        pass
gaps = [(a, b - a) for a, b in zip(ts, ts[1:]) if b - a > 1.0]
span = ts[-1] - ts[0]
print(f"{len(ts)} samples over {span / 60:.1f} min, "
      f"coverage {1 - sum(g for _, g in gaps) / span:.2%}")
for a, g in gaps:
    print(f"  gap at {time.strftime('%H:%M:%S', time.localtime(a))}: {g:.1f} s")
print("logger events:")
for line in open(f"{out}/logger.log"):
    if any(k in line for k in ("silent", "lost", "USB error", "fresh restart", "not found")):
        print("  " + line.strip())
PY
echo "Saved in $OUT"
