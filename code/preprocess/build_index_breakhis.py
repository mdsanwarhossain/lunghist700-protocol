"""Build a canonical image index for BreakHis, the external replication cohort.

BreakHis ships 7,909 breast histopathology images from 82 patients at four
magnifications. The patient is encoded in the filename, for example
``SOB_B_TA-14-4659-40-001.png``:

    SOB   biopsy procedure
    B     benign (M for malignant)          -> the binary task
    TA    tumour type, eight in total       -> the eight-class task
    14-4659  slide/patient identifier       -> the grouping variable
    40    magnification
    001   sequence number within the slide

Only one magnification is indexed by default. Pooling all four would put the
same tissue region, imaged at 40X and at 400X, on both sides of an image-level
split, which is a *second* leakage channel on top of the patient one and would
confound the contrast this cohort exists to replicate.

The output columns deliberately match ``build_index.py`` so that the feature
extractor, the splitters and the benchmark driver run unchanged.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# tumour-type abbreviation -> readable name; the eight-class task's label set
TUMOUR_TYPES = {
    "A": "adenosis", "F": "fibroadenoma", "PT": "phyllodes", "TA": "tubular-adenoma",
    "DC": "ductal", "LC": "lobular", "MC": "mucinous", "PC": "papillary",
}
SUPERCLASSES = {"B": "benign", "M": "malignant"}

# SOB_B_TA-14-4659-40-001.png
NAME = re.compile(r"^SOB_(?P<cls>[BM])_(?P<type>[A-Z]{1,2})[-_]"
                  r"(?P<slide>[0-9]+-[0-9A-Za-z]+)-(?P<mag>\d+)-(?P<seq>\d+)\.png$")


def build(root: Path, magnification: str, out_csv: Path) -> pd.DataFrame:
    recs = []
    for p in sorted(root.rglob("*.png")):
        m = NAME.match(p.name)
        if not m or m.group("mag") != magnification:
            continue
        cls, ttype, slide = m.group("cls"), m.group("type"), m.group("slide")
        if cls not in SUPERCLASSES or ttype not in TUMOUR_TYPES:
            continue
        recs.append({
            "path": str(p),
            "exists": True,
            "class_dir": f"{SUPERCLASSES[cls]}_{TUMOUR_TYPES[ttype]}",
            "superclass": SUPERCLASSES[cls],
            "subclass": TUMOUR_TYPES[ttype],
            # named label7 only so the shared Task machinery can address it;
            # on this cohort it carries eight classes, not seven.
            "label7": TUMOUR_TYPES[ttype],
            "magnification": "both",   # single-magnification index, see docstring
            "image_id": f"{slide}-{m.group('seq')}",
            "patient_slide": slide,
            "grade": -1,
        })
    idx = pd.DataFrame(recs)
    if not len(idx):
        raise SystemExit(f"no images matched under {root} at {magnification}X")

    # The splitters want an integer group; keep the original string beside it.
    slides = sorted(idx.patient_slide.unique())
    idx["patient_id"] = idx.patient_slide.map({s: i for i, s in enumerate(slides)})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(idx)} images, {len(slides)} patients, "
          f"{idx.superclass.nunique()} superclasses, {idx.label7.nunique()} types, "
          f"magnification {magnification}X")
    print(idx.groupby(['superclass', 'label7']).size().to_string())
    print("images per patient: min %d median %.0f max %d"
          % (idx.groupby('patient_id').size().min(),
             idx.groupby('patient_id').size().median(),
             idx.groupby('patient_id').size().max()))
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="directory holding the extracted BreaKHis_v1 tree")
    ap.add_argument("--magnification", default="100", choices=["40", "100", "200", "400"])
    ap.add_argument("--out", type=Path,
                    default=Path("results/tables/breakhis_index.csv"))
    a = ap.parse_args()
    build(a.root, a.magnification, a.out)
