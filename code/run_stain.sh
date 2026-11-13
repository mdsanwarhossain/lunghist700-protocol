#!/usr/bin/env bash
# The deferred Macenko pass and the stain control that depends on it.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/run_rest.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== Macenko encoding pass ==="
./code/features/extract_all.sh >>"$LOG" 2>&1 || say "macenko FAILED"
if [ -f data/features/resnet50_in1k__macenko/tiles.npy ]; then
  say "=== stage e3 ==="
  ./code/run_study.sh e3 >>"$LOG" 2>&1 && say "stage e3 OK" || say "stage e3 FAILED"
  say "=== report refresh ==="
  ./code/run_study.sh report >>"$LOG" 2>&1 && say "report OK" || say "report FAILED"
fi
say "STAIN DONE"
