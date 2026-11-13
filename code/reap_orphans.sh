#!/usr/bin/env bash
# Kill pipeline workers whose thermal supervisor has died.
#
# Stopping a backgrounded task kills the shell wrapper and thermal_run.py, but
# a worker that was SIGSTOPped at that moment survives, reparented to init and
# stopped forever: nothing is left to send it SIGCONT. It then holds memory,
# keeps its output directory open, and -- worst -- a later relaunch writes the
# same shard files alongside it. One such orphan cost this project five hours
# of apparent progress.
#
# Matching is deliberately strict. A naive `pgrep -f` also matches the
# supervisor (the worker's path appears in its own argv), any shell whose
# command line mentions it, and this script -- killing the supervisor would be
# far worse than the orphan it is meant to remove. So a candidate must be a
# Python process whose argv[1] is one of the worker scripts, and it is spared
# whenever its parent is a live thermal_run.
#
#   code/reap_orphans.sh          report only
#   code/reap_orphans.sh --kill   kill what it finds
set -uo pipefail
kill_them=${1:-}

WORKERS='run_benchmark.py|run_fusion.py|extract_embeddings.py|transfer_and_curves.py|attribution.py|external_lc25000.py|validate_external.py'

found=0
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  [ -r "$d/cmdline" ] || continue
  argv=$(tr '\0' '\n' < "$d/cmdline" 2>/dev/null)
  [ -n "$argv" ] || continue

  # argv[0] must be a python interpreter, argv[1] one of the worker scripts.
  a0=$(printf '%s\n' "$argv" | sed -n 1p)
  a1=$(printf '%s\n' "$argv" | sed -n 2p)
  case "$a0" in *python*) ;; *) continue ;; esac
  printf '%s\n' "$a1" | grep -qE "($WORKERS)\$" || continue
  # the supervisor runs python too, but its argv[1] is thermal_run.py, so it
  # is already excluded by the test above; this is belt and braces.
  printf '%s\n' "$argv" | grep -q thermal_run && continue

  ppid=$(awk '{print $4}' "$d/stat" 2>/dev/null)
  state=$(awk '{print $3}' "$d/stat" 2>/dev/null)
  if [ -r "/proc/$ppid/cmdline" ] &&
     tr '\0' ' ' < "/proc/$ppid/cmdline" 2>/dev/null | grep -q thermal_run; then
    continue   # a live supervisor owns it
  fi

  found=$((found + 1))
  echo "orphan pid=$pid state=$state ppid=$ppid :: $(basename "$a1")"
  if [ "$kill_them" = "--kill" ]; then
    kill -9 "$pid" 2>/dev/null && echo "  killed"
  fi
done

if [ "$found" -eq 0 ]; then
  echo "no orphaned workers"
elif [ "$kill_them" != "--kill" ]; then
  echo "re-run with --kill to remove them"
fi
