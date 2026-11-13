#!/usr/bin/env bash
# Encode the dataset once per (backbone, stain-normalisation) pair, one pass at
# a time under the thermal governor.
#
# Passes run strictly sequentially. Running two encoders side by side with a
# thread pool each oversubscribes the CPU, which both slows the job down and
# heats the package until the machine throttles; a single pass at a bounded
# thread count finishes sooner. Every pass is block-checkpointed, so an
# interrupted run resumes where it stopped.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
IDX=results/tables/image_index.csv
OUT=data/features
THREADS="${THREADS:-6}"
# Rest between passes: the package sensor falls in seconds but the heatsink
# does not, and stacking passes back to back is what soaked the chassis on the
# run that cost us the machine. Each pass therefore waits for a genuinely cool
# idle laptop before it starts.
COOL_BETWEEN="${COOL_BETWEEN:-65}"
mkdir -p "$OUT" results/logs

# Cheapest first, so the pipeline has usable features early; the stain arm is
# run only for the two backbones carried into the ablation.
# The stain arm was cut from four passes to one. Each pass costs most of a
# working day on this CPU, and stain normalisation is a control here, not a
# result: within a single-centre cohort it is expected to change little, and
# its purpose is to show that the protocol effect is not an artefact of stain
# variation. One pass answers that. Macenko on ResNet-50 is the pass kept,
# because Macenko is the method used by the only prior study on this dataset
# that normalised at all (Wang et al.), and ResNet-50 is the descriptor's
# backbone, so the control sits on the most comparable configuration.
PASSES=(
  efficientnet_b0:raw
  resnet50_in1k:raw
  phikon_vitb16:raw
  dinov2_vitb14:raw
  resnet50_in1k:macenko
)

for spec in "${PASSES[@]}"; do
  b="${spec%%:*}"; n="${spec##*:}"
  if [ -f "$OUT/${b}__${n}/tiles.npy" ]; then echo "skip ${b}__${n} (done)"; continue; fi
  echo "=== $b / $n ==="
  $PY code/thermal_run.py --gate-only --cool "$COOL_BETWEEN" --wait-timeout 2400
  $PY code/thermal_run.py --threads "$THREADS" -- \
      $PY code/features/extract_embeddings.py --index "$IDX" --backbone "$b" \
      --norm "$n" --outroot "$OUT" --threads "$THREADS" --keep-shards \
    2>&1 | grep -v "unauthenticated requests"
  if [ ! -f "$OUT/${b}__${n}/tiles.npy" ]; then
    echo "FAILED: ${b}__${n}" >&2; exit 1
  fi
done
echo "ALL EXTRACTION DONE"
