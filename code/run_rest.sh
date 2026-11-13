#!/usr/bin/env bash
# Run everything that is left, strictly one job at a time.
#
# Two governed jobs running at once defeat the governor: neither knows about
# the other, their combined load reaches the panic threshold within seconds,
# and both then sit in a long cooldown. Measured over an hour of overlap the
# analysis job spent 46 of 53 minutes cooling. Serialising costs nothing in
# total compute and avoids that entirely.
#
# Ordering matters as much as serialising. Only stage e3 needs the Macenko
# pass, so the analyses run as soon as the last *raw* encoder (dinov2) is
# cached, and the Macenko pass is deferred to the end. That puts the headline
# results in hand hours earlier without changing the total work.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/run_rest.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

need=data/features/dinov2_vitb14__raw/tiles.npy
say "waiting for the last raw encoder (dinov2)"
while [ ! -f "$need" ]; do
  pgrep -f 'features/extract_embeddings' >/dev/null || \
    pgrep -f 'features/extract_all' >/dev/null || {
      say "extraction is no longer running and dinov2 is absent; continuing without it"
      break
    }
  sleep 60
done
[ -f "$need" ] && say "dinov2 cached"

# Stop the extraction driver so the deferred Macenko pass does not start and
# contend with the analyses. Nothing is lost: every finished pass is on disk
# and every unfinished one is block-checkpointed.
if pgrep -f 'features/extract_all' >/dev/null; then
  say "pausing the extraction driver so the analyses get the machine"
  pkill -f 'features/extract_all' 2>/dev/null
  pkill -f 'thermal_run.py --threads .* extract_embeddings' 2>/dev/null
  sleep 5
  pkill -f 'features/extract_embeddings' 2>/dev/null
  sleep 3
fi

say "encoded passes available:"
for d in data/features/*__*/; do
  [ -f "$d/tiles.npy" ] && say "  $(basename "$d")"
done

for stage in e1 e2 e4 e5 transfer curve attribution report; do
  say "=== stage $stage ==="
  ./code/run_study.sh "$stage" >>"$LOG" 2>&1 \
    && say "stage $stage OK" || say "stage $stage FAILED (continuing)"
done
say "MAIN RESULTS COMPLETE"

# Now the deferred Macenko pass and the stain control that depends on it.
if [ ! -f data/features/resnet50_in1k__macenko/tiles.npy ]; then
  say "=== deferred: Macenko encoding pass ==="
  THREADS=12 ./code/features/extract_all.sh >>"$LOG" 2>&1 || say "macenko pass FAILED"
fi
if [ -f data/features/resnet50_in1k__macenko/tiles.npy ]; then
  say "=== stage e3 ==="
  ./code/run_study.sh e3 >>"$LOG" 2>&1 && say "stage e3 OK" || say "stage e3 FAILED"
  say "=== report (refresh with e3) ==="
  ./code/run_study.sh report >>"$LOG" 2>&1 && say "report OK" || say "report FAILED"
fi
say "ALL REMAINING STAGES DONE"
