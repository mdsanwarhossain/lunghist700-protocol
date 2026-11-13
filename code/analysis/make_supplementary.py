"""Render every Supplementary table from the result CSVs.

The Supplementary exists to show the full grids behind the eight display items
the main text is allowed. Generating it from the same tables the main text
reads means the two cannot disagree, and a stage that has not run leaves a
visible note rather than a stale number.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

T = Path("results/tables")
OUT = Path("paper/tables")
DISPLAY = {"resnet50_in1k": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
           "phikon_vitb16": "Phikon ViT-B/16", "dinov2_vitb14": "DINOv2 ViT-B/14"}
TASK = {"super3": "3-class", "full7": "7-class", "grade3": "grading"}


def esc(v) -> str:
    s = DISPLAY.get(str(v), str(v))
    for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("+", "$+$")):
        s = s.replace(a, b)
    return s


def table(rows: list[str], header: str, spec: str, note: str = "") -> str:
    if not rows:
        return (r"\textit{" + (note or "This stage did not run; no table.") + "}")
    return "\n".join([r"\footnotesize", r"\begin{tabular}{@{}" + spec + r"@{}}",
                      r"\toprule", header + r" \\", r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"])


def read(name: str):
    p = T / name
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d if len(d) else None


def ci(r, m="balanced_accuracy"):
    lo, hi = r.get(f"{m}_lo"), r.get(f"{m}_hi")
    if lo is None or not np.isfinite(lo):
        return f"{r[f'{m}_mean']:.3f}"
    return f"{r[f'{m}_mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUT)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # S2: per-fold composition, one seed, all three protocols
    d = read("e1_leakage_folds.csv")
    if d is not None:
        g = d[(d.seed == 0) & (d["head"] == "linear") & (d.task == "full7")]
        rows = [" & ".join([r.protocol, str(int(r.fold)), str(int(r.n_train)),
                            str(int(r.n_test)), str(int(r.n_train_patients)),
                            str(int(r.n_test_patients)), str(int(r.patient_overlap)),
                            str(int(r.image_overlap))]) + r" \\"
                for _, r in g.sort_values(["protocol", "fold"]).iterrows()]
        (a.outdir / "s2_folds.tex").write_text(table(
            rows, "Protocol & Fold & $n$ train & $n$ test & train pat. & test pat. "
                  "& patient overlap & image overlap", "l c c c c c c c") + "\n")
        written.append("s2_folds")

    # S3 / S5 / S8: summary grids
    for src, name, keys in (
        ("summary_e1_leakage.csv", "s3_protocol",
         ["task", "protocol", "head", "representation"]),
        ("summary_e2_backbones.csv", "s5_backbones",
         ["backbone", "task", "representation", "magnification"]),
        ("summary_e4_heads.csv", "s8_heads", ["backbone", "task", "head"]),
    ):
        d = read(src)
        if d is None:
            continue
        keys = [k for k in keys if k in d.columns]
        rows = []
        for _, r in d.sort_values(keys).iterrows():
            cells = [esc(TASK.get(r[k], r[k])) for k in keys]
            cells += [str(int(r.n_folds)), ci(r),
                      f"{r.get('macro_f1_mean', float('nan')):.3f}",
                      f"{r.get('mcc_mean', float('nan')):.3f}"]
            rows.append(" & ".join(cells) + r" \\")
        hdr = " & ".join([k.replace("_", " ") for k in keys] +
                         ["folds", "balanced accuracy [95\\% CI]", "macro-$F_1$", "MCC"])
        (a.outdir / f"{name}.tex").write_text(
            table(rows, hdr, "l" * len(keys) + " c c c c") + "\n")
        written.append(name)

    # S4: paired inflation with statistics
    d = read("leakage_inflation.csv")
    if d is not None:
        rows = [" & ".join([esc(TASK.get(r.task, r.task)), r["head"], r.protocol,
                            f"{r.base_mean:.3f}", f"{r.cur_mean:.3f}",
                            f"$+${100*r.delta:.1f}" if r.delta >= 0 else f"{100*r.delta:.1f}",
                            f"{r.cohens_d:.2f}", f"{r.p_holm:.2g}"]) + r" \\"
                for _, r in d.sort_values(["task", "head", "protocol"]).iterrows()]
        (a.outdir / "s4_inflation.tex").write_text(table(
            rows, "Task & Head & Protocol & Patient & Comparator & $\\Delta$ (pts) "
                  "& Cohen's $d$ & $p_{\\text{Holm}}$", "l l l c c c c c") + "\n")
        written.append("s4_inflation")

    # S9: fusion ablation
    d = read("e5_fusion_folds.csv")
    if d is not None:
        g = d.groupby(["task", "arm"]).balanced_accuracy.agg(["mean", "std", "count"])
        rows = [" & ".join([esc(TASK.get(t, t)), esc(arm), f"{r['mean']:.3f}",
                            f"{r['std']:.3f}", str(int(r['count']))]) + r" \\"
                for (t, arm), r in g.iterrows()]
        (a.outdir / "s9_fusion.tex").write_text(table(
            rows, "Task & Arm & Balanced accuracy & SD & folds", "l l c c c") + "\n")
        written.append("s9_fusion")

    # S10: calibration and risk-coverage
    cal, rc = read("calibration_full7.csv"), read("risk_coverage_full7.csv")
    parts = []
    if cal is not None:
        r = cal.iloc[0]
        parts.append(table(
            [" & ".join([f"{r.accuracy:.3f}", f"{r.mean_confidence:.3f}",
                         f"{r.ece:.3f}", f"{r.mce:.3f}",
                         f"{r.ece_temperature_scaled:.3f}",
                         f"{r.fitted_temperature:.1f}"]) + r" \\"],
            "Accuracy & Mean confidence & ECE & MCE & ECE (scaled) & Temperature",
            "c c c c c c"))
    if rc is not None:
        sel = rc[rc.coverage.isin([0.2, 0.4, 0.5, 0.6, 0.8, 1.0])]
        parts.append(table(
            [" & ".join([f"{r.coverage:.2f}", f"{r.selective_risk:.3f}",
                         f"{r.threshold:.3f}", str(int(r.n_retained))]) + r" \\"
             for _, r in sel.iterrows()],
            "Coverage & Selective risk & Threshold & $n$ retained", "c c c c"))
    if parts:
        (a.outdir / "s10_calibration.tex").write_text(
            "\n\n\\vspace{1em}\n\n".join(parts) + "\n")
        written.append("s10_calibration")

    # S11: LC25000 group recovery
    d = read("lc25000_group_recovery.csv")
    if d is not None:
        rows = [" & ".join([esc(r.lc_label), str(int(r.n_images)),
                            str(int(r.n_groups_recovered)),
                            str(int(r.n_originals_documented)),
                            f"{r.median_group_size:.0f}", str(int(r.max_group_size)),
                            str(int(r.groups_of_size_20))]) + r" \\"
                for _, r in d.iterrows()]
        (a.outdir / "s11_lc25000.tex").write_text(table(
            rows, "Class & Images & Groups recovered & Originals documented "
                  "& Median size & Max size & Groups of size 20", "l c c c c c c") + "\n")
        written.append("s11_lc25000")

    # S7: stain control, if it ever ran
    d = read("summary_e3_stain.csv")
    if d is not None:
        rows = [" & ".join([esc(r.backbone), r.norm, esc(TASK.get(r.task, r.task)),
                            str(int(r.n_folds)), ci(r)]) + r" \\"
                for _, r in d.sort_values(["task", "norm"]).iterrows()]
        (a.outdir / "s7_stain.tex").write_text(table(
            rows, "Encoder & Normalisation & Task & folds & balanced accuracy [95\\% CI]",
            "l l l c c") + "\n")
        written.append("s7_stain")

    print(f"wrote {len(written)} supplementary tables: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
