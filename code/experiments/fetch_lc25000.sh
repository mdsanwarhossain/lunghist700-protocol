#!/usr/bin/env bash
# Fetch the LC25000 parquet shards used for external validation.
# A shard counts as present only if its parquet footer reads back, so an
# interrupted download is retried instead of being silently accepted.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=data/lc25000
PY=./.venv/bin/python
mkdir -p "$OUT"
$PY - <<'PYEOF' > "$OUT/.filelist"
import json, urllib.request
d = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/1aurent/LC25000"))
for s in d["siblings"]:
    if s["rfilename"].endswith(".parquet"):
        print(s["rfilename"])
PYEOF

valid() { $PY -c "import sys,pyarrow.parquet as pq; pq.ParquetFile(sys.argv[1]).metadata" "$1" 2>/dev/null; }

while read -r rf; do
  dest="$OUT/$(basename "$rf")"
  if valid "$dest"; then echo "have $(basename "$rf")"; continue; fi
  for attempt in 1 2 3; do
    echo "fetching $rf (attempt $attempt)"
    curl -sL --retry 3 --retry-delay 5 -o "$dest" \
      "https://huggingface.co/datasets/1aurent/LC25000/resolve/main/$rf"
    valid "$dest" && break
    echo "  incomplete, retrying"
  done
  valid "$dest" || { echo "FAILED: $rf"; exit 1; }
done < "$OUT/.filelist"
echo LC25000_DOWNLOAD_DONE
$PY -c "
import glob, pyarrow.parquet as pq
print(sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob('data/lc25000/*.parquet')), 'rows total')"
