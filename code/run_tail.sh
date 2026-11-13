#!/usr/bin/env bash
# Finish the stages that remain after e1 and e2, one at a time.
# e1 and e2 already have their tables, so they are not repeated.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/run_rest.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for stage in e4 e5 transfer curve attribution report; do
  say "=== stage $stage ==="
  ./code/run_study.sh "$stage" >>"$LOG" 2>&1 \
    && say "stage $stage OK" || say "stage $stage FAILED (continuing)"
done
say "MAIN RESULTS COMPLETE"

if [ ! -f data/features/resnet50_in1k__macenko/tiles.npy ]; then
  say "=== deferred: Macenko encoding pass ==="
  ./code/features/extract_all.sh >>"$LOG" 2>&1 || say "macenko pass FAILED"
fi
if [ -f data/features/resnet50_in1k__macenko/tiles.npy ]; then
  for stage in e3 report; do
    say "=== stage $stage ==="
    ./code/run_study.sh "$stage" >>"$LOG" 2>&1 && say "stage $stage OK" || say "stage $stage FAILED"
  done
fi
say "ALL REMAINING STAGES DONE"
