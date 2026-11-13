"""Audit LungHist700 for properties that constrain any honest evaluation.

Reports (a) the patient->class incidence structure, which bounds how many
patient-disjoint folds the 7-class task can support, (b) exact-duplicate files,
(c) per-patient image counts, and (d) the imbalance profile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEVEN = ["aca_bd", "aca_md", "aca_pd", "nor", "scc_bd", "scc_md", "scc_pd"]


def main(index_csv: Path, outdir: Path) -> None:
    df = pd.read_csv(index_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    rep: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        rep.append(s)

    say("=" * 78)
    say("LungHist700 dataset audit")
    say("=" * 78)
    say(f"images={len(df)}  patients={df.patient_id.nunique()}  "
        f"magnifications={sorted(df.magnification.unique())}")
    say(f"pixel sizes: {df.pixel_size.unique().tolist()}")
    say()

    # ---- exact duplicates -------------------------------------------------
    dup = df[df.duplicated("md5", keep=False)].sort_values("md5")
    say(f"[1] exact byte-duplicate images: {len(dup)}")
    if len(dup):
        for h, g in dup.groupby("md5"):
            say("    " + " == ".join(Path(p).name for p in g.path))
    say()

    # ---- class / magnification counts ------------------------------------
    say("[2] class x magnification counts")
    ct = pd.crosstab(df.label7, df.magnification, margins=True)
    say(ct.to_string())
    say()
    say("    superclass counts: " + df.superclass.value_counts().to_dict().__str__())
    imb = df.label7.value_counts()
    say(f"    7-class imbalance ratio (max/min) = {imb.max()/imb.min():.2f}")
    say()

    # ---- patient -> class incidence --------------------------------------
    say("[3] patient x class incidence (number of images)")
    pc = pd.crosstab(df.patient_id, df.label7)
    pc = pc.reindex(columns=[c for c in SEVEN if c in pc.columns], fill_value=0)
    say(pc.to_string())
    pc.to_csv(outdir / "patient_class_incidence.csv")
    say()

    n_classes_per_patient = (pc > 0).sum(axis=1)
    say("    classes per patient: " + n_classes_per_patient.value_counts().sort_index().to_dict().__str__())
    say(f"    patients contributing >1 of the 7 classes: {(n_classes_per_patient > 1).sum()}")

    pats_per_class = (pc > 0).sum(axis=0)
    say()
    say("[4] DISTINCT PATIENTS PER CLASS  <-- bounds patient-disjoint CV")
    for c, n in pats_per_class.items():
        flag = "  <-- limits k" if n < 10 else ""
        say(f"    {c:8s}: {n:2d} patients, {int(pc[c].sum()):3d} images{flag}")
    say(f"    minimum patients in any of the 7 classes = {pats_per_class.min()}")
    say(f"    => a patient-disjoint, class-stratified design cannot exceed "
        f"k = {pats_per_class.min()} folds for the 7-class task")

    sup = pd.crosstab(df.patient_id, df.superclass)
    sup_pats = (sup > 0).sum(axis=0)
    say()
    say("    distinct patients per superclass: " + sup_pats.to_dict().__str__())
    say(f"    => 3-class task bound: k <= {sup_pats.min()} folds")
    say()

    # ---- images per patient ----------------------------------------------
    ipp = df.groupby("patient_id").size()
    say("[5] images per patient")
    say(f"    min={ipp.min()} median={ipp.median():.0f} max={ipp.max()} mean={ipp.mean():.1f}")
    say(f"    top-5 contributors hold {ipp.nlargest(5).sum()}/{len(df)} images "
        f"({100*ipp.nlargest(5).sum()/len(df):.1f}%)")
    say()

    # ---- paired magnification --------------------------------------------
    say("[6] magnification coverage per (patient, class)")
    pm = df.groupby(["patient_id", "label7"]).magnification.nunique()
    say(f"    (patient,class) groups with both 20x and 40x: {(pm == 2).sum()}/{len(pm)}")
    say()

    # ---- what the original paper's single 10% test split implies ---------
    say("[7] size of the original single 80/10/10 patient-wise split")
    say(f"    10% of {len(df)} images ~= {round(0.10*len(df))} images, drawn from "
        f"~{max(1, round(0.10*df.patient_id.nunique()))} of {df.patient_id.nunique()} patients")
    say("    => one test fold rests on a handful of patients; a single split cannot")
    say("       separate model quality from patient sampling noise.")

    (outdir / "dataset_audit.txt").write_text("\n".join(rep) + "\n")
    print(f"\nwritten -> {outdir/'dataset_audit.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    main(a.index, a.outdir)
