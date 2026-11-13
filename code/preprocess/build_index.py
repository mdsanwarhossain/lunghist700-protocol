"""Build a canonical, verified image index for LungHist700.

The shipped ``data.csv`` keys images by (superclass, subclass, resolution,
image_id); ``image_id`` alone is NOT unique (134 distinct values for 691 rows),
so every downstream join must use the full 4-tuple. This module materialises
that join once, verifies it against the files on disk, and writes
``results/tables/image_index.csv``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

SUBCLASS_ORDER = {"bd": 0, "md": 1, "pd": 2}  # well -> moderately -> poorly (ordinal)
SUPERCLASSES = ["aca", "nor", "scc"]
SEVEN_CLASSES = ["aca_bd", "aca_md", "aca_pd", "nor", "scc_bd", "scc_md", "scc_pd"]


def _filename(row: pd.Series) -> str:
    """Reconstruct the on-disk name: nor rows carry an empty subclass."""
    sub = "" if pd.isna(row.subclass) else str(row.subclass)
    return f"{row.superclass}_{sub}_{row.resolution}_{row.image_id}.jpg".replace("__", "_")


def _class_dir(row: pd.Series) -> str:
    sub = "" if pd.isna(row.subclass) else str(row.subclass)
    return f"{row.superclass}_{sub}" if sub else str(row.superclass)


def build(data_root: Path, out_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(data_root / "data.csv", keep_default_na=True)
    meta.columns = [c.strip() for c in meta.columns]

    recs = []
    for row in meta.itertuples(index=False):
        row = pd.Series(row._asdict())
        cdir = _class_dir(row)
        path = data_root / "images" / cdir / _filename(row)
        sub = "" if pd.isna(row.subclass) else str(row.subclass)
        recs.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "class_dir": cdir,
                "superclass": row.superclass,
                "subclass": sub,
                "label7": cdir,
                "magnification": row.resolution,
                "image_id": row.image_id,
                "patient_id": int(row.patient_id),
                # grade is ordinal within a carcinoma type; normal tissue has none
                "grade": SUBCLASS_ORDER.get(sub, -1),
            }
        )
    idx = pd.DataFrame(recs)

    missing = idx.loc[~idx.exists, "path"].tolist()
    if missing:
        raise SystemExit(f"{len(missing)} indexed files absent from disk, e.g. {missing[:3]}")

    on_disk = {str(p) for p in (data_root / "images").rglob("*.jpg")}
    unindexed = on_disk - set(idx.path)
    if unindexed:
        raise SystemExit(f"{len(unindexed)} files on disk are not in data.csv, e.g. {list(unindexed)[:3]}")

    # Verify pixel geometry and fingerprint every file so duplicate detection and
    # the "all images are 1200x1600" claim are checked rather than assumed.
    sizes, hashes = [], []
    for p in idx.path:
        with Image.open(p) as im:
            sizes.append(f"{im.size[0]}x{im.size[1]}")
        hashes.append(hashlib.md5(Path(p).read_bytes()).hexdigest())
    idx["pixel_size"] = sizes
    idx["md5"] = hashes

    idx = idx.drop(columns=["exists"]).sort_values(["label7", "magnification", "image_id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(out_csv, index=False)
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = build(a.data_root, a.out)
    print(f"indexed {len(df)} images, {df.patient_id.nunique()} patients -> {a.out}")
    print(df.pixel_size.value_counts().to_string())
