"""Clinical metrics table: sensitivity, specificity, precision, AUC, and the
patient-level interval beside the fold-level one.

Reads results/tables/clinical_metrics.csv (written by clinical_metrics.py) and
results/tables/summary_e2_backbones.csv, and selects the same best
patient-disjoint configuration per task that the headline figures use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

T = Path("results/tables")

TASKS = [("super3", "3-class diagnosis"),
         ("full7", "7-class subtype + grade"),
         ("grade3", "3-class differentiation grade")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("paper/tables/clinical_metrics.tex"))
    a = ap.parse_args()

    cm = pd.read_csv(T / "clinical_metrics.csv")
    summ = pd.read_csv(T / "summary_e2_backbones.csv")
    d = summ[(summ.protocol == "patient") & (summ["head"] == "linear")
             & (summ.magnification == "both")]

    rows = []
    for task, label in TASKS:
        g = d[d.task == task]
        if not len(g):
            continue
        best = g.loc[g.balanced_accuracy_mean.idxmax()]
        sel = cm[(cm.task == task) & (cm.backbone == best.backbone)
                 & (cm.representation == best.representation)
                 & (cm.magnification == best.magnification)]
        v = {r.metric: r for r in sel.itertuples()}
        if "balanced_accuracy" not in v:
            continue
        ba = v["balanced_accuracy"]
        rows.append(
            f"{label} & {v['sensitivity'].value:.3f} & "
            f"{v['specificity'].value:.3f} & {v['precision'].value:.3f} & "
            f"{v['auc_ovr'].value:.3f} & "
            f"[{best.balanced_accuracy_lo:.3f}, {best.balanced_accuracy_hi:.3f}] & "
            f"[{ba.lo:.3f}, {ba.hi:.3f}] & "
            f"{(ba.hi - ba.lo) / (best.balanced_accuracy_hi - best.balanced_accuracy_lo):.1f}$\\times$ \\\\")

    tex = "\n".join([
        r"\footnotesize",
        r"\begin{tabular}{@{}l c c c c c c c@{}}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Pooled out-of-fold predictions}"
        r" & \multicolumn{3}{c}{95\% interval on balanced accuracy} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
        r"Task & Sens. & Spec. & Prec. & AUC & over folds & over patients"
        r" & width \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tex + "\n")
    print(f"wrote {a.out} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
