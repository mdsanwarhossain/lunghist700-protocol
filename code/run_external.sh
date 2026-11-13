#!/usr/bin/env bash
# The external LC25000 validation, which the earlier runners omitted.
# Waits for the curve stage to finish so the two never contend.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs/run_rest.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for the curve stage before starting external validation"
while pgrep -f 'transfer_and_curves' >/dev/null 2>&1; do sleep 60; done
say "=== stage external ==="
./code/run_study.sh external >>"$LOG" 2>&1 && say "stage external OK" || say "stage external FAILED"
say "=== report refresh ==="
./code/run_study.sh report >>"$LOG" 2>&1 && say "report OK" || say "report FAILED"
say "EXTERNAL DONE"
