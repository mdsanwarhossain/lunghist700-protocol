#!/usr/bin/env bash
# Reproduce the whole study from the extracted features.
# Usage: code/run_study.sh [stage ...]   (default: all stages)
#
# Every heavy stage runs through code/thermal_run.py, which bounds the thread
# pool and suspends the job if the CPU package gets too hot. Stages write their
# own tables, so a stage that has already produced its output can be skipped by
# naming only the stages still wanted.
set -uo pipefail
set -o pipefail   # a stage's own exit code must survive the pipe to tee
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
IDX=results/tables/image_index.csv
T=results/tables
L=results/logs
THREADS="${THREADS:-6}"
# k is bounded by the scarcest class: 8 distinct patients for full7.
KFOLD="${KFOLD:-8}"
mkdir -p "$T" "$L" results/figures

# One job at a time, bounded threads, thermally governed. The governor waits
# for a cool idle machine before each stage, duty-cycles while running and
# suspends on a shut lid, so stages never stack heat on one another.
run() { $PY code/thermal_run.py --threads "$THREADS" -- $PY "$@"; }

stages=("${@:-audit e1 e2 e3 e4 e5 transfer curve attribution external report}")
has() { [[ " ${stages[*]} " == *" $1 "* ]]; }

if has audit; then
  echo "### dataset audit"
  $PY code/preprocess/build_index.py --data-root data/data --out "$IDX"
  $PY code/analysis/audit_dataset.py --index "$IDX" --outdir "$T"
fi

# E1: the protocol comparison that is the study's headline claim.
if has e1; then
  echo "### E1 leakage / protocol study"
  run code/experiments/run_benchmark.py --grid code/experiments/grids/e1_leakage.json \
     --out-prefix "$T/e1_leakage" --n-splits "$KFOLD" --seeds 0 1 2 3 4 2>&1 | tee "$L/e1.log"
fi

# E2: which frozen representation, under the correct protocol only.
if has e2; then
  echo "### E2 representation benchmark"
  run code/experiments/run_benchmark.py --grid code/experiments/grids/e2_backbones.json \
     --out-prefix "$T/e2_backbones" --n-splits "$KFOLD" --seeds 0 1 2 3 4 2>&1 | tee "$L/e2.log"
fi

# E3: stain normalisation as a domain-shift control.
if has e3; then
  echo "### E3 stain normalisation"
  run code/experiments/run_benchmark.py --grid code/experiments/grids/e3_stain.json \
     --out-prefix "$T/e3_stain" --n-splits "$KFOLD" --seeds 0 1 2 3 4 2>&1 | tee "$L/e3.log"
fi

# E4: aggregation heads, including attention-MIL over the full tile grid.
if has e4; then
  echo "### E4 heads / MIL"
  run code/experiments/run_benchmark.py --grid code/experiments/grids/e4_heads.json \
     --out-prefix "$T/e4_heads" --n-splits "$KFOLD" --seeds 0 1 2 2>&1 | tee "$L/e4.log"
fi

# E5: everything reachable from the cached embeddings, as a cumulative
# ablation. Cheap, because it adds no encoding pass.
if has e5; then
  echo "### E5 fusion / pooling / tuned head"
  run code/experiments/run_fusion.py --out-prefix "$T/e5_fusion" \
     --n-splits "$KFOLD" --seeds 0 1 2 3 4 2>&1 | tee "$L/e5.log"
fi

if has transfer; then
  echo "### cross-magnification transfer"
  for b in resnet50_in1k phikon_vitb16; do
    run code/experiments/transfer_and_curves.py --backbone $b --which transfer --n-splits "$KFOLD" \
       --tasks super3 full7 2>&1 | tee "$L/transfer_$b.log"
  done
fi

if has curve; then
  echo "### patient learning curve"
  run code/experiments/transfer_and_curves.py --backbone phikon_vitb16 --which curve --n-splits "$KFOLD" \
     --tasks super3 full7 2>&1 | tee "$L/curve.log"
fi

if has attribution; then
  echo "### attention faithfulness (deletion / insertion vs random order)"
  run code/experiments/attribution.py --backbone phikon_vitb16 \
     --tasks full7 super3 --n-splits "$KFOLD" --seeds 0 1 2>&1 | tee "$L/attribution.log"
fi

if has external; then
  echo "### LC25000 group recovery + external validation"
  ./code/experiments/fetch_lc25000.sh 2>&1 | tee "$L/lc25000_dl.log"
  run code/experiments/external_lc25000.py --backbone resnet50_in1k 2>&1 | tee "$L/external_groups.log"
  run code/experiments/validate_external.py 2>&1 | tee "$L/external_eval.log"
fi

if has report; then
  echo "### aggregate, figures, report"
  $PY code/analysis/make_report.py 2>&1 | tee "$L/report.log"
  $PY code/analysis/make_related_table.py 2>&1 | tee -a "$L/report.log"
  $PY code/analysis/make_numbers.py 2>&1 | tee -a "$L/report.log"
fi
echo "STUDY COMPLETE"
