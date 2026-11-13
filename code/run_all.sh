#!/usr/bin/env bash
# End-to-end: fetch the data, encode it once, run every experiment, build the
# report. Safe to re-run; each stage skips work already on disk.
set -euo pipefail
cd "$(dirname "$0")/.."
./code/preprocess/fetch_lunghist700.sh
./.venv/bin/python code/preprocess/build_index.py --data-root data/data \
    --out results/tables/image_index.csv
./code/features/extract_all.sh
./code/run_study.sh
