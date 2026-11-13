#!/usr/bin/env bash
# Fetch and unpack LungHist700 (Diosdado et al., Sci Data 2024).
#
# The archive is checksum-verified against the figshare record before it is
# unpacked, so a truncated download cannot quietly become a smaller dataset.
set -euo pipefail
cd "$(dirname "$0")/../.."

DOI=10.6084/m9.figshare.25459174
URL=https://ndownloader.figshare.com/files/45206104
MD5=82cb8790152631eba41da228522ad0ef
RAR=data/raw/LungHist700.rar
mkdir -p data/raw data

if [ -f "$RAR" ] && [ "$(md5sum "$RAR" | cut -d' ' -f1)" = "$MD5" ]; then
  echo "have verified $RAR"
else
  echo "fetching LungHist700 (640 MB) from figshare $DOI"
  curl -L --retry 3 --retry-delay 5 -o "$RAR" "$URL"
  got=$(md5sum "$RAR" | cut -d' ' -f1)
  [ "$got" = "$MD5" ] || { echo "md5 mismatch: $got != $MD5" >&2; exit 1; }
fi

if [ -f data/data/data.csv ]; then
  echo "already unpacked -> data/data"
else
  command -v unar >/dev/null || command -v unrar >/dev/null || {
    echo "need 'unar' or 'unrar' to unpack a .rar (apt install unar)" >&2; exit 1; }
  if command -v unar >/dev/null; then unar -q -o data "$RAR"; else unrar x -idq "$RAR" data/; fi
fi

n=$(find data/data/images -name '*.jpg' | wc -l)
echo "unpacked $n images (expect 691) + data/data/data.csv"
[ "$n" -eq 691 ] || { echo "unexpected image count" >&2; exit 1; }
