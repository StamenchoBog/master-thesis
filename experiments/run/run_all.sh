#!/bin/bash
# One full run (one arm, one seed): ./run_all.sh naive|sisa SEED [--wan]
set -euo pipefail
cd "$(dirname "$0")"

for step in 01_prepare 02_instruments 03_train 04_recover 05_rejoin 06_collect; do
  "./$step.sh" "$@"
done
