"""Assemble every result table and figure the manuscript cites."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import aggregate as A  # noqa: E402
from analysis import calibration as C  # noqa: E402
from analysis import figures as F  # noqa: E402
from analysis import interpretability as I  # noqa: E402
from experiments import protocols as P  # noqa: E402

T = Path("results/tables")
FIG = Path("results/figures")
TASK_CLASSES = {k: list(v.classes) for k, v in P.TASKS.items()}


def _read(name: str) -> pd.DataFrame | None:
    p = T / name
    return pd.read_csv(p) if p.exists() else None


def _fmt(m: float, lo: float, hi: float) -> str:
    if not np.isfinite(m):
        return "-"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def report_leakage(lines: list[str]) -> pd.DataFrame | None:
    folds = _read("e1_leakage_folds.csv")
    if folds is None:
        return None
    s = A.summarize(folds)
    s.to_csv(T / "summary_e1_leakage.csv", index=False)

    lines.append("\n## Protocol comparison (identical features and classifier)\n")
    hdr = ("| task | representation | head | protocol | unit | n folds | "
           "balanced accuracy [95% CI] | macro-F1 | MCC | patient overlap |")
    lines += [hdr, "|" + "---|" * 10]
    for _, r in s.sort_values(["task", "representation", "head", "protocol"]).iterrows():
        lines.append(
            f"| {r.task} | {r.representation} | {r['head']} | {r.protocol} | {r.unit} | "
            f"{int(r.n_folds)} | "
            f"{_fmt(r.balanced_accuracy_mean, r.balanced_accuracy_lo, r.balanced_accuracy_hi)} | "
            f"{r.macro_f1_mean:.3f} | {r.mcc_mean:.3f} | {int(r.patient_overlap_max)} |")

    # inflation attributable purely to relaxing the split
    rows = []
    # Grouping on representation and magnification as well as task and head
    # keeps each side of the contrast one score per fold; see the pairing guard
    # in aggregate.compare.
    gkeys = ["task", "head", "representation", "magnification"]
    for vals, g in folds.groupby(gkeys):
        sel = dict(zip(gkeys, vals))
        if g[g.protocol == "patient"].empty:
            continue
        cmp = A.compare(g, "balanced_accuracy", ["protocol"],
                        {**sel, "protocol": "patient"},
                        [{"protocol": "image"}, {"protocol": "tile"}])
        if len(cmp):
            cmp.insert(0, "task", sel["task"])
            cmp.insert(1, "head", sel["head"])
            cmp.insert(2, "representation", sel["representation"])
            rows.append(cmp)
    if rows:
        inf = pd.concat(rows, ignore_index=True)
        inf.to_csv(T / "leakage_inflation.csv", index=False)
        lines.append("\n### Inflation from relaxing the split\n")
        lines += ["| task | representation | head | relaxed protocol | "
                  "patient-disjoint | relaxed | delta [95% CI] | Cohen d | "
                  "p (corrected t) | p (Holm) |",
                  "|" + "---|" * 10]
        for _, r in inf.iterrows():
            lines.append(
                f"| {r.task} | {r.representation} | {r['head']} | {r.protocol} | {r.base_mean:.3f} | "
                f"{r.cur_mean:.3f} | {r.delta:+.3f} [{r.delta_lo:.3f}, {r.delta_hi:.3f}] | "
                f"{r.cohens_d:.2f} | {r.p_corrected_t:.2g} | {r.p_holm:.2g} |")
    F.fig_leakage(folds, FIG)
    return folds


def report_backbones(lines: list[str]):
    folds = _read("e2_backbones_folds.csv")
    if folds is None:
        return None, None
    s = A.summarize(folds)
    s.to_csv(T / "summary_e2_backbones.csv", index=False)

    lines.append("\n## Frozen representations, patient-disjoint 5x5-fold CV\n")
    for task in ("super3", "full7", "grade3"):
        d = s[(s.task == task) & (s["head"] == "linear") & (s.magnification == "both")]
        if d.empty:
            continue
        lines.append(f"\n### {task}\n")
        extra = " QWK |" if task == "grade3" else ""
        lines += [f"| backbone | representation | balanced accuracy [95% CI] | "
                  f"macro-F1 | MCC | AUC (OvR) |{extra}",
                  "|" + "---|" * (6 + (1 if extra else 0))]
        for _, r in d.sort_values("balanced_accuracy_mean", ascending=False).iterrows():
            q = f" {r.get('qwk_mean', float('nan')):.3f} |" if extra else ""
            auc = r.get("auc_ovr_mean", float("nan"))
            lines.append(
                f"| {r.backbone} | {r.representation} | "
                f"{_fmt(r.balanced_accuracy_mean, r.balanced_accuracy_lo, r.balanced_accuracy_hi)} | "
                f"{r.macro_f1_mean:.3f} | {r.mcc_mean:.3f} | "
                f"{auc:.3f} |{q}")

    # the trivial control: tissue coverage and stain colour only
    hc = s[(s.representation == "handcrafted") & (s.magnification == "both")]
    if len(hc):
        lines.append("\n### Shortcut control: tissue coverage and stain colour only\n")
        lines += ["| task | balanced accuracy [95% CI] | macro-F1 |", "|---|---|---|"]
        for _, r in hc.iterrows():
            lines.append(f"| {r.task} | {_fmt(r.balanced_accuracy_mean, r.balanced_accuracy_lo, r.balanced_accuracy_hi)} | "
                         f"{r.macro_f1_mean:.3f} |")

    # magnification, within the correct protocol
    lines.append("\n### Magnification\n")
    d = s[(s["head"] == "linear") & (s.representation == "tilemean")]
    lines += ["| backbone | task | 20x | 40x | pooled |", "|---|---|---|---|---|"]
    for (b, task), g in d.groupby(["backbone", "task"]):
        gv = {r.magnification: r.balanced_accuracy_mean for _, r in g.iterrows()}
        lines.append(f"| {b} | {task} | {gv.get('20x', float('nan')):.3f} | "
                     f"{gv.get('40x', float('nan')):.3f} | {gv.get('both', float('nan')):.3f} |")

    # statistical contrast of each backbone against the descriptor's ResNet-50
    cons = []
    for task in folds.task.unique():
        g = folds[(folds.task == task) & (folds["head"] == "linear") &
                  (folds.representation == "tilemean") & (folds.magnification == "both")]
        others = [{"backbone": b} for b in g.backbone.unique() if b != "resnet50_in1k"]
        c = A.compare(g, "balanced_accuracy", ["backbone"],
                      {"task": task, "head": "linear", "representation": "tilemean",
                       "magnification": "both", "backbone": "resnet50_in1k"}, others)
        if len(c):
            c.insert(0, "task", task)
            cons.append(c)
    if cons:
        cc = pd.concat(cons, ignore_index=True)
        cc.to_csv(T / "backbone_contrasts.csv", index=False)
        lines.append("\n### Each representation against ResNet-50 (ImageNet)\n")
        lines += ["| task | backbone | ResNet-50 | this | delta [95% CI] | Cohen d | p (Holm) |",
                  "|" + "---|" * 7]
        for _, r in cc.iterrows():
            lines.append(f"| {r.task} | {r.backbone} | {r.base_mean:.3f} | {r.cur_mean:.3f} | "
                         f"{r.delta:+.3f} [{r.delta_lo:.3f}, {r.delta_hi:.3f}] | "
                         f"{r.cohens_d:.2f} | {r.p_holm:.2g} |")
    F.fig_backbones(s, FIG)
    return folds, s


def report_errors_and_calibration(lines: list[str]) -> None:
    preds = _read("e2_backbones_preds.csv")
    if preds is None:
        return
    best = None
    s = _read("summary_e2_backbones.csv")
    if s is not None:
        d = s[(s.task == "full7") & (s["head"] == "linear") & (s.magnification == "both") &
              (s.representation == "tilemean")]
        if len(d):
            best = d.sort_values("balanced_accuracy_mean", ascending=False).iloc[0].backbone
    best = best or "resnet50_in1k"
    lines.append(f"\n## Error structure and confidence (7-class, {best}, patient-disjoint)\n")

    sel = preds[(preds.task == "full7") & (preds.backbone == best) &
                (preds["head"] == "linear") & (preds.representation == "tilemean") &
                (preds.magnification == "both") & (preds.protocol == "patient")]
    if sel.empty:
        return
    classes = TASK_CLASSES["full7"]
    F.fig_confusion(sel, classes, FIG, "fig_confusion_full7",
                    f"7-class, {best}, patient-disjoint")
    s3 = preds[(preds.task == "super3") & (preds.backbone == best) &
               (preds["head"] == "linear") & (preds.representation == "tilemean") &
               (preds.magnification == "both") & (preds.protocol == "patient")]
    if len(s3):
        F.fig_confusion(s3, TASK_CLASSES["super3"], FIG, "fig_confusion_super3",
                        f"3-class, {best}, patient-disjoint")

    err = P.graded_confusion_summary(sel.y_true.values, sel.y_pred.values)
    lines.append("\n### Where the 7-class errors fall\n")
    lines += ["| error type | share of all predictions |", "|---|---|"]
    for k, v in err.items():
        lines.append(f"| {k.replace('err_', '').replace('_', ' ')} | {v:.3f} |")

    cal = C.analyse(sel, classes)
    rel, rc = cal.pop("_reliability"), cal.pop("_risk_coverage")
    rel.to_csv(T / "reliability_full7.csv", index=False)
    rc.to_csv(T / "risk_coverage_full7.csv", index=False)
    pd.DataFrame([cal]).to_csv(T / "calibration_full7.csv", index=False)
    lines.append("\n### Calibration and selective prediction\n")
    lines += ["| quantity | value |", "|---|---|"]
    for k, v in cal.items():
        lines.append(f"| {k.replace('_', ' ')} | {v:.4f} |")
    F.fig_calibration(rel, rc, FIG, f"7-class, {best}")

    # case-level versus field-level
    pv = I.patient_vs_field(sel, classes)
    pv.to_csv(T / "patient_vs_field_full7.csv", index=False)
    lines.append("\n### Field-level against case-level reporting\n")
    lines += ["| aggregation | case balanced accuracy | field balanced accuracy | delta |",
              "|---|---|---|---|"]
    for rule, g in pv.groupby("rule"):
        lines.append(f"| {rule} | {g.case_balanced_accuracy.mean():.3f} | "
                     f"{g.field_balanced_accuracy.mean():.3f} | {g.delta.mean():+.3f} |")


def report_stain(lines: list[str]) -> None:
    e2, e3 = _read("e2_backbones_folds.csv"), _read("e3_stain_folds.csv")
    if e3 is None or e2 is None:
        return
    base = e2[(e2["head"] == "linear") & (e2.representation == "tilemean") &
              (e2.magnification == "both") & (e2.norm == "raw")]
    folds = pd.concat([base, e3], ignore_index=True)
    s = A.summarize(folds)
    s.to_csv(T / "summary_e3_stain.csv", index=False)
    lines.append("\n## Stain normalisation\n")
    lines += ["| backbone | task | normalisation | balanced accuracy [95% CI] | delta vs raw | p (Holm) |",
              "|" + "---|" * 6]
    for (b, task), g in folds.groupby(["backbone", "task"]):
        if "raw" not in set(g.norm):
            continue
        cmp = A.compare(g, "balanced_accuracy", ["norm"],
                        {"backbone": b, "task": task, "norm": "raw",
                         "head": "linear", "representation": "tilemean",
                         "magnification": "both"},
                        [{"norm": n} for n in g.norm.unique() if n != "raw"])
        raw = s[(s.backbone == b) & (s.task == task) & (s.norm == "raw")]
        if len(raw):
            r = raw.iloc[0]
            lines.append(f"| {b} | {task} | raw | "
                         f"{_fmt(r.balanced_accuracy_mean, r.balanced_accuracy_lo, r.balanced_accuracy_hi)} | - | - |")
        for _, r in cmp.iterrows():
            rr = s[(s.backbone == b) & (s.task == task) & (s.norm == r.norm)]
            ci = (_fmt(rr.iloc[0].balanced_accuracy_mean, rr.iloc[0].balanced_accuracy_lo,
                       rr.iloc[0].balanced_accuracy_hi) if len(rr) else "-")
            lines.append(f"| {b} | {task} | {r.norm} | {ci} | {r.delta:+.3f} | {r.p_holm:.2g} |")


def report_heads(lines: list[str]) -> None:
    e4 = _read("e4_heads_folds.csv")
    e2 = _read("e2_backbones_folds.csv")
    if e4 is None:
        return
    parts = [e4]
    if e2 is not None:
        parts.append(e2[(e2["head"] == "linear") & (e2.representation == "tilemean") &
                        (e2.magnification == "both")])
    folds = pd.concat(parts, ignore_index=True)
    s = A.summarize(folds)
    s.to_csv(T / "summary_e4_heads.csv", index=False)
    lines.append("\n## Aggregation head (patient-disjoint)\n")
    lines += ["| backbone | task | head | balanced accuracy [95% CI] | macro-F1 |",
              "|" + "---|" * 5]
    for _, r in s.sort_values(["task", "backbone", "head"]).iterrows():
        lines.append(f"| {r.backbone} | {r.task} | {r['head']} | "
                     f"{_fmt(r.balanced_accuracy_mean, r.balanced_accuracy_lo, r.balanced_accuracy_hi)} | "
                     f"{r.macro_f1_mean:.3f} |")


def report_transfer_curve(lines: list[str]) -> None:
    tr = [p for p in T.glob("magnification_transfer_*.csv")]
    if tr:
        d = pd.concat([pd.read_csv(p) for p in tr], ignore_index=True)
        lines.append("\n## Cross-magnification transfer (patient-disjoint)\n")
        lines += ["| backbone | task | train | test | matched | balanced accuracy | macro-F1 |",
                  "|" + "---|" * 7]
        g = d.groupby(["backbone", "task", "train_mag", "test_mag", "matched"]).agg(
            ba=("balanced_accuracy", "mean"), f1=("macro_f1", "mean")).reset_index()
        for _, r in g.iterrows():
            lines.append(f"| {r.backbone} | {r.task} | {r.train_mag} | {r.test_mag} | "
                         f"{'yes' if r.matched else 'no'} | {r.ba:.3f} | {r.f1:.3f} |")
        drop = (g[g.matched].ba.mean() - g[~g.matched].ba.mean())
        lines.append(f"\nMean balanced-accuracy cost of a magnification mismatch: "
                     f"{drop:.3f}.\n")
        F.fig_magnification(d, FIG)
    cu = [p for p in T.glob("learning_curve_*.csv")]
    if cu:
        d = pd.concat([pd.read_csv(p) for p in cu], ignore_index=True)
        lines.append("\n## Patient-count learning curve\n")
        lines += ["| task | training patients | balanced accuracy |", "|---|---|---|"]
        for (task, frac), g in d.groupby(["task", "fraction"]):
            lines.append(f"| {task} | {g.n_train_patients.mean():.0f} | "
                         f"{g.balanced_accuracy.mean():.3f} |")
        F.fig_learning_curve(d, FIG)


def report_attribution(lines: list[str]) -> None:
    """Attention ranking against a random-order control on the same bags."""
    from scipy import stats

    fa = sorted(T.glob("attribution_faithfulness_*.csv"))
    if fa:
        d = pd.concat([pd.read_csv(p) for p in fa], ignore_index=True)
        lines.append("\n## Is the attention faithful? Deletion and insertion\n")
        lines += ["| backbone | task | tile order | AUC deletion (lower better) | "
                  "AUC insertion (higher better) | gap |", "|" + "---|" * 6]
        for (b, task), g in d.groupby(["backbone", "task"]):
            for mode, gg in g.groupby("mode"):
                lines.append(f"| {b} | {task} | {mode} | {gg.auc_deletion.mean():.3f} | "
                             f"{gg.auc_insertion.mean():.3f} | "
                             f"{gg.faithfulness_gap.mean():+.3f} |")
            # paired over images: the same bag ranked by attention and at random
            w = g.pivot_table(index=["seed", "fold", "image"], columns="mode",
                              values="faithfulness_gap")
            if {"attention", "random"}.issubset(w.columns):
                w = w.dropna()
                if len(w) > 2:
                    t, pv = stats.wilcoxon(w.attention, w.random)
                    lines.append(f"\nPaired over {len(w)} held-out fields "
                                 f"({b}, {task}): attention-ordered gap exceeds the "
                                 f"random-order gap by "
                                 f"{(w.attention - w.random).mean():+.3f} "
                                 f"(Wilcoxon p={pv:.2g}).\n")
    co = sorted(T.glob("attention_concentration_*.csv"))
    if co:
        d = pd.concat([pd.read_csv(p) for p in co], ignore_index=True)
        lines.append("\n### How peaked the attention is\n")
        lines += ["| backbone | task | class | tiles | normalised entropy | "
                  "top-1 share | top-5 share |", "|" + "---|" * 7]
        for (b, task, lab), g in d.groupby(["backbone", "task", "label"]):
            lines.append(f"| {b} | {task} | {lab} | {g.n_tiles.mean():.0f} | "
                         f"{g.normalised_entropy.mean():.3f} | "
                         f"{g.top1_share.mean():.3f} | {g.top5_share.mean():.3f} |")


def report_external(lines: list[str]) -> None:
    rec = _read("lc25000_group_recovery.csv")
    if rec is not None:
        lines.append("\n## LC25000: recovering the discarded augmentation groups\n")
        lines += ["| LC25000 class | images | groups recovered | documented originals | "
                  "median group size |", "|" + "---|" * 5]
        for _, r in rec.iterrows():
            lines.append(f"| {r.lc_label} | {int(r.n_images)} | {int(r.n_groups_recovered)} | "
                         f"{int(r.n_originals_documented)} | {r.median_group_size:.0f} |")
    for p in T.glob("external_lc25000_*.csv"):
        d = pd.read_csv(p)
        lines.append(f"\n### Zero-shot transfer to LC25000 lung ({p.stem.split('_')[-1]})\n")
        lines += ["| test set | images | independent groups | balanced accuracy | "
                  "naive CI width | group-bootstrap CI width |", "|" + "---|" * 6]
        for ts, g in d.groupby("test_set"):
            lines.append(f"| {ts} | {g.n_images.mean():.0f} | "
                         f"{g.n_independent_groups.mean():.0f} | "
                         f"{g.balanced_accuracy.mean():.3f} | "
                         f"{g.naive_ci_width.mean():.4f} | {g.cluster_ci_width.mean():.4f} |")


def report_design_and_literature() -> None:
    """Figures that depend only on the dataset audit and the literature table.

    These are produced on every report run, including before any experiment
    stage has finished, so the manuscript's framing figures are never stale.
    """
    inc = _read("patient_class_incidence.csv")
    if inc is not None:
        F.fig_design(inc, FIG)

    rel = _read("related_work.csv")
    if rel is None or "headline_value" not in rel.columns:
        return
    rel["headline_value"] = pd.to_numeric(rel.headline_value, errors="coerce")
    # Our own marker is added only once the patient-disjoint summary exists.
    ours = None
    s2 = _read("summary_e2_backbones.csv")
    if s2 is not None and len(s2):
        best = (s2[(s2.task == "super3") & (s2.protocol == "patient")]
                .sort_values("balanced_accuracy_mean").tail(1))
        if len(best):
            r = best.iloc[0]
            ours = (float(r.balanced_accuracy_mean), float(r.balanced_accuracy_lo),
                    float(r.balanced_accuracy_hi))
    F.fig_literature(rel, FIG, ours=ours)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/RESULTS.md"))
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    report_design_and_literature()

    lines = ["# Results", "",
             "Generated by `code/analysis/make_report.py`. Every interval is a "
             "percentile bootstrap over cross-validation folds; every p-value is "
             "a Nadeau-Bengio corrected resampled t-test with Holm adjustment "
             "across the contrasts of its own comparison.", ""]
    audit = Path(T / "dataset_audit.txt")
    if audit.exists():
        lines += ["## Dataset audit", "", "```", audit.read_text().strip(), "```", ""]
    report_leakage(lines)
    report_backbones(lines)
    report_errors_and_calibration(lines)
    report_stain(lines)
    report_heads(lines)
    report_transfer_curve(lines)
    report_attribution(lines)
    report_external(lines)

    a.out.write_text("\n".join(lines) + "\n")
    print(f"report -> {a.out}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
