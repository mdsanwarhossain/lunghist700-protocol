"""Emit every number the manuscript quotes as a LaTeX macro.

A paper that hand-types its results drifts from them the moment a stage is
re-run. Instead the prose cites macros, and this script regenerates them from
the tables on every build. A macro whose stage has not run yet renders as a
visible PENDING marker, so an unfilled number is impossible to miss and
impossible to mistake for a measurement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

T = Path("results/tables")

# macro name -> short description shown if the number is not available yet
WANTED = {
    "eOneSuperThreePatient":  "E1 super3 patient-disjoint balanced accuracy",
    "eOneSuperThreeImage":    "E1 super3 image-level balanced accuracy",
    "eOneSuperThreeTile":     "E1 super3 tile-level balanced accuracy",
    "eOneSuperThreeInflation": "E1 super3 image-minus-patient inflation",
    "eOneSuperThreeInflationVal": "E1 super3 inflation, bare value",
    "eOneFullSevenPatient":   "E1 full7 patient-disjoint balanced accuracy",
    "eOneFullSevenImage":     "E1 full7 image-level balanced accuracy",
    "eOneFullSevenInflation": "E1 full7 image-minus-patient inflation",
    "eOneFullSevenInflationVal": "E1 full7 inflation, bare value",
    "mechInflationAtOne":     "inflation at one image per group",
    "mechInflationAtTwo":     "inflation at two images per group",
    "mechInflationAtFour":    "inflation at four images per group",
    "mechInflationFull":      "inflation at the full cohort",
    "mechShareByFour":        "share of the full inflation reached by four",
    "mechNNFull":             "same-group nearest neighbours, ungrouped, full",
    "mechNNAtTwo":            "same-group nearest neighbours, ungrouped, cap 2",
    "mechImagesPerGroup":     "mean images per group, full cohort",
    "mechBhAtOne":            "BreakHis inflation at one image per group",
    "mechBhFull":             "BreakHis inflation at the full cohort",
    "mechCtrlLow":            "control: inflation at the fewest images per group",
    "mechCtrlHigh":           "control: inflation at the most images per group",
    "mechCtrlLowCell":        "control: low cell, images x groups",
    "mechCtrlHighCell":       "control: high cell, images x groups",
    "mechCtrlN":              "control: images per cell",
    "bhPatientBinary":        "BreakHis binary, patient-disjoint",
    "bhImageBinary":          "BreakHis binary, image-level",
    "bhInflationBinary":      "BreakHis binary inflation",
    "bhPatientTypeEight":     "BreakHis 8-type, patient-disjoint",
    "bhImageTypeEight":       "BreakHis 8-type, image-level",
    "bhInflationTypeEight":   "BreakHis 8-type inflation",
    "bhPatients":             "BreakHis patient count",
    "bhImages":               "BreakHis image count at the audited magnification",
    "cmSensSuperThree":       "clinical macro sensitivity, best super3",
    "cmSpecSuperThree":       "clinical macro specificity, best super3",
    "cmPrecSuperThree":       "clinical macro precision, best super3",
    "cmAucSuperThree":        "one-vs-rest AUC, best super3",
    "cmSensFullSeven":        "clinical macro sensitivity, best full7",
    "cmSpecFullSeven":        "clinical macro specificity, best full7",
    "cmPrecFullSeven":        "clinical macro precision, best full7",
    "cmAucFullSeven":         "one-vs-rest AUC, best full7",
    "cmSensGradeThree":       "clinical macro sensitivity, best grade3",
    "cmSpecGradeThree":       "clinical macro specificity, best grade3",
    "cmPrecGradeThree":       "clinical macro precision, best grade3",
    "cmAucGradeThree":        "one-vs-rest AUC, best grade3",
    "cmPatientSuperThree":    "best super3 with a patient-level interval",
    "cmPatientFullSeven":     "best full7 with a patient-level interval",
    "cmPatientGradeThree":    "best grade3 with a patient-level interval",
    "cmWidthRatioSuperThree": "patient/fold interval width ratio, super3",
    "cmWidthRatioFullSeven":  "patient/fold interval width ratio, full7",
    "cmWidthRatioGradeThree": "patient/fold interval width ratio, grade3",
    "eTwoBestSuperThree":     "E2 best patient-disjoint super3",
    "eTwoBestSuperThreeBare": "E2 best super3, no encoder name",
    "eTwoBestFullSeven":      "E2 best patient-disjoint full7",
    "eTwoBestFullSevenBare": "E2 best full7, no encoder name",
    "eTwoBestGradeThree":     "E2 best patient-disjoint grade3",
    "eTwoBestGradeThreeBare": "E2 best grade3, no encoder name",
    "eTwoSpread":             "E2 spread across frozen encoders",
    "eTwoSpreadSuperThree":   "E2 encoder spread, super3",
    "eTwoSpreadFullSeven":    "E2 encoder spread, full7",
    "eTwoSpreadGradeThree":   "E2 encoder spread, grade3",
    "eTwoDinoSuperThree":     "E2 DINOv2 super3 patient-disjoint",
    "eTwoDinoFullSeven":      "E2 DINOv2 full7 patient-disjoint",
    "eTwoDinoGradeThree":     "E2 DINOv2 grade3 patient-disjoint",
    "eTwoPhikonFullSeven":    "E2 Phikon full7 patient-disjoint",
    "eTwoPhikonGradeThree":   "E2 Phikon grade3 patient-disjoint",
    "eTwoRatioSuperThree":    "protocol effect / encoder spread, super3",
    "eTwoRatioFullSeven":     "protocol effect / encoder spread, full7",
    "eTwoRatioGradeThree":    "protocol effect / encoder spread, grade3",
    "eTwoResnetSuperThree":   "E2 ResNet-50 super3 patient-disjoint",
    "eOneGradeThreePatient":  "E1 grade3 patient-disjoint balanced accuracy",
    "eOneGradeThreeImage":    "E1 grade3 image-level balanced accuracy",
    "eOneGradeThreeInflation": "E1 grade3 image-minus-patient inflation",
    "eOneGradeThreeInflationVal": "E1 grade3 inflation, bare value",
    "eFiveBestSuperThree":    "E5 best fused super3",
    "eFiveBestFullSeven":     "E5 best fused full7",
    "eFiveGainSuperThree":    "E5 gain over the single-encoder baseline, super3",
    "calAccuracy":            "full7 accuracy at the calibration stage",
    "calConfidence":          "full7 mean predicted confidence",
    "calGap":                 "full7 overconfidence gap",
    "calEce":                 "full7 expected calibration error",
    "calEceScaled":           "full7 ECE after temperature scaling",
    "calTemperature":         "fitted temperature",
    "calRiskAtEighty":            "selective risk at 80% coverage",
    "calRiskAtFifty":            "selective risk at 50% coverage",
    "stainRawSuperThree":     "ResNet-50 super3, unnormalised",
    "stainMacSuperThree":     "ResNet-50 super3, Macenko",
    "stainDeltaSuperThree":   "Macenko minus raw, super3",
    "stainDeltaFullSeven":    "Macenko minus raw, full7",
    "lcGroupsRecovered":      "LC25000 groups recovered per class",
    "lcGroupsExpected":       "LC25000 originals documented per class",
    "lcMedianGroup":          "median recovered group size",
    "curveSuperThreeLow":     "super3 with the fewest training patients",
    "curveSuperThreeHigh":    "super3 with the full cohort",
    "curveFullSevenLow":      "full7 with the fewest training patients",
    "curveFullSevenHigh":     "full7 with the full cohort",
    "curveLowN":              "patients in the smallest subsample",
    "curveHighN":             "patients in the full cohort",
    "magTwenty":              "E2 super3 patient-disjoint at 20x",
    "magForty":               "E2 super3 patient-disjoint at 40x",
    "magBoth":                "E2 super3 patient-disjoint, both magnifications",
    "trMatchedTwenty":            "super3 trained and tested at 20x",
    "trMatchedForty":            "super3 trained and tested at 40x",
    "trDown":                 "super3 trained at 40x, tested at 20x",
    "trUp":                   "super3 trained at 20x, tested at 40x",
    "eOneSuperThreeD":        "Cohen's d for the super3 image-vs-patient contrast",
    "eOneFullSevenD":         "Cohen's d for the full7 image-vs-patient contrast",
    "eFourAttnSuperThree":    "E4 attention-MIL super3",
    "eFourMlpSuperThree":     "E4 MLP super3",
    "eFourAttnFullSeven":     "E4 attention-MIL full7",
    "eFourMlpFullSeven":      "E4 MLP full7",
}


# Internal identifiers must never reach the page: the underscore in e.g.
# "phikon_vitb16" is a LaTeX special character and would break the build, and
# a reader should see the model's published name in any case.
DISPLAY = {
    "resnet50_in1k": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "phikon_vitb16": "Phikon ViT-B/16",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
}


def _tex(v: object) -> str:
    """Make an arbitrary label safe to typeset."""
    s = DISPLAY.get(str(v), str(v))
    for a, b in (("\\", "/"), ("_", "\\_"), ("&", "\\&"), ("%", "\\%"),
                 ("#", "\\#"), ("$", "\\$")):
        s = s.replace(a, b)
    return s


def _read(name: str) -> pd.DataFrame | None:
    p = T / name
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d if len(d) else None


def _fmt(m: float, lo: float | None = None, hi: float | None = None) -> str:
    if lo is None or not np.isfinite(lo):
        return f"{m:.3f}"
    return f"{m:.3f} (95\\% CI {lo:.3f}--{hi:.3f})"


def _mean_ci(g: pd.DataFrame, metric: str = "balanced_accuracy") -> tuple:
    v = g[metric].dropna().values
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan, None, None)
    rng = np.random.default_rng(0)
    b = rng.choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def collect() -> dict[str, str]:
    out: dict[str, str] = {}

    folds = _read("e1_leakage_folds.csv")
    if folds is not None:
        d = folds[folds["head"] == "linear"]
        for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven"),
                          ("grade3", "GradeThree")):
            per = {}
            for proto in ("patient", "image", "tile"):
                g = d[(d.task == task) & (d.protocol == proto)]
                if len(g):
                    m, lo, hi = _mean_ci(g)
                    per[proto] = m
                    key = f"eOne{tag}{proto.capitalize()}"
                    if key in WANTED:
                        out[key] = _fmt(m, lo, hi)
            if "patient" in per and "image" in per:
                delta = 100 * (per["image"] - per["patient"])
                out[f"eOne{tag}Inflation"] = f"{delta:+.1f} percentage points"
                # Bare variant: the abstract lists all three inflations in one
                # series, where repeating the unit three times wastes words
                # against the 250-word limit.
                out[f"eOne{tag}InflationVal"] = f"{delta:+.1f}"

    s2 = _read("summary_e2_backbones.csv")
    if s2 is not None:
        # Restrict to the pooled-magnification arm. Only ResNet-50 was run at
        # single magnifications, so selecting the maximum over all of them
        # would reward that encoder for having been given more arms rather
        # than for being better; "both" is the setting every encoder shares.
        d = s2[(s2.protocol == "patient") & (s2["head"] == "linear") &
               (s2.magnification == "both")]
        for task, key in (("super3", "eTwoBestSuperThree"),
                          ("full7", "eTwoBestFullSeven"),
                          ("grade3", "eTwoBestGradeThree")):
            g = d[d.task == task]
            if len(g):
                r = g.loc[g.balanced_accuracy_mean.idxmax()]
                bare = _fmt(r.balanced_accuracy_mean, r.balanced_accuracy_lo,
                            r.balanced_accuracy_hi)
                out[key] = bare + f" ({_tex(r.backbone)})"
                # Variant without the encoder name, for the abstract, where the
                # encoder is already named and nested parentheses read badly.
                out[key + "Bare"] = bare
        sp = {}
        for task in ("super3", "full7", "grade3"):
            g = d[(d.task == task) & (d.representation == "tilemean") &
                  (d.magnification == "both")]
            if len(g) > 1:
                sp[task] = 100 * (g.balanced_accuracy_mean.max()
                                  - g.balanced_accuracy_mean.min())
        for task, tag in (("super3", "SuperThreeh"), ("full7", "FullSevenh"),
                          ("grade3", "GradeThreeh")):
            if task in sp:
                out[f"eTwoSpread{tag[:-1]}"] = f"{sp[task]:.1f} percentage points"
        g = d[(d.task == "super3") & (d.representation == "tilemean") &
              (d.magnification == "both") & (d.backbone == "resnet50_in1k")]
        if len(g):
            out["eTwoResnetSuperThree"] = f"{g.balanced_accuracy_mean.iloc[0]:.3f}"
        # Per-encoder figures on the shared tile-mean arm, so the three
        # pretraining regimes can be named in the text without typing a number.
        for bb, nick in (("dinov2_vitb14", "Dino"), ("phikon_vitb16", "Phikon")):
            for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven"),
                              ("grade3", "GradeThree")):
                key = f"eTwo{nick}{tag}"
                if key not in WANTED:
                    continue
                g = d[(d.task == task) & (d.representation == "tilemean") &
                      (d.magnification == "both") & (d.backbone == bb)]
                if len(g):
                    out[key] = f"{g.balanced_accuracy_mean.iloc[0]:.3f}"
        # How many times larger the protocol effect is than the encoder spread.
        for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven"),
                          ("grade3", "GradeThree")):
            infl = out.get(f"eOne{tag}InflationVal")
            if task in sp and infl and sp[task] > 0:
                out[f"eTwoRatio{tag}"] = f"{abs(float(infl)) / sp[task]:.1f}"
        if sp:
            vals = [f"{v:.1f}" for v in (sp.get("super3"), sp.get("full7"),
                                         sp.get("grade3")) if v is not None]
            # Serial list, so the sentence reads as English: "14.8, 3.5 and 1.6"
            # rather than "14.8, 3.5, 1.6".
            joined = (" and ".join([", ".join(vals[:-1]), vals[-1]])
                      if len(vals) > 1 else vals[0])
            out["eTwoSpread"] = (
                joined + " percentage points for the three-, seven- and "
                "grading tasks")

    if s2 is not None:
        g = s2[(s2.protocol == "patient") & (s2["head"] == "linear") &
               (s2.backbone == "resnet50_in1k") & (s2.task == "super3") &
               (s2.representation == "tilemean")]
        for key, mag in (("magTwenty", "20x"), ("magForty", "40x"),
                         ("magBoth", "both")):
            r = g[g.magnification == mag]
            if len(r):
                out[key] = f"{r.balanced_accuracy_mean.iloc[0]:.3f}"

    st = _read("e3_stain_folds.csv")
    if st is not None:
        st = st[st.backbone == "resnet50_in1k"]
        for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven")):
            g = st[st.task == task].groupby("norm").balanced_accuracy.mean()
            if {"raw", "macenko"} <= set(g.index):
                if tag == "SuperThree":
                    out["stainRawSuperThree"] = f"{g['raw']:.3f}"
                    out["stainMacSuperThree"] = f"{g['macenko']:.3f}"
                out[f"stainDelta{tag}"] = f"{100*(g['macenko']-g['raw']):+.1f} points"

    lc = _read("lc25000_group_recovery.csv")
    if lc is not None:
        out["lcGroupsRecovered"] = f"{lc.n_groups_recovered.mean():.0f}"
        out["lcGroupsExpected"] = f"{lc.n_originals_documented.iloc[0]:.0f}"
        out["lcMedianGroup"] = f"{lc.median_group_size.median():.0f}"

    cu = _read("learning_curve_phikon_vitb16__raw.csv")
    if cu is not None:
        g = cu.groupby(["task", "fraction"]).agg(
            n=("n_train_patients", "mean"), b=("balanced_accuracy", "mean"))
        for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven")):
            if task in g.index.get_level_values(0):
                t = g.loc[task]
                out[f"curve{tag}Low"] = f"{t.b.iloc[0]:.3f}"
                out[f"curve{tag}High"] = f"{t.b.iloc[-1]:.3f}"
                out["curveLowN"] = f"{t.n.iloc[0]:.0f}"
                out["curveHighN"] = f"{t.n.iloc[-1]:.0f}"

    tr = _read("magnification_transfer_phikon_vitb16__raw.csv")
    if tr is not None:
        g = tr[tr.task == "super3"].groupby(
            ["train_mag", "test_mag"]).balanced_accuracy.mean()
        for key, pair in (("trMatchedTwenty", ("20x", "20x")),
                          ("trMatchedForty", ("40x", "40x")),
                          ("trDown", ("40x", "20x")),
                          ("trUp", ("20x", "40x"))):
            if pair in g.index:
                out[key] = f"{g.loc[pair]:.3f}"

    cal = _read("calibration_full7.csv")
    if cal is not None:
        r = cal.iloc[0]
        out["calAccuracy"] = f"{r.accuracy:.3f}"
        out["calConfidence"] = f"{r.mean_confidence:.3f}"
        out["calGap"] = f"{r.overconfidence_gap:.3f}"
        out["calEce"] = f"{r.ece:.3f}"
        out["calEceScaled"] = f"{r.ece_temperature_scaled:.3f}"
        out["calTemperature"] = f"{r.fitted_temperature:.1f}"
        out["calRiskAtEighty"] = f"{r.risk_at_80_coverage:.3f}"
        out["calRiskAtFifty"] = f"{r.risk_at_50_coverage:.3f}"

    inf = _read("leakage_inflation.csv")
    if inf is not None:
        for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven")):
            g = inf[(inf.task == task) & (inf["head"] == "linear") &
                    (inf.protocol == "image")]
            if len(g):
                out[f"eOne{tag}D"] = f"{g.cohens_d.iloc[0]:.2f}"

    e4 = _read("summary_e4_heads.csv")
    if e4 is not None:
        for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven")):
            for head, hs in (("attnmil", "Attn"), ("mlp", "Mlp")):
                g = e4[(e4.task == task) & (e4["head"] == head)]
                if len(g):
                    r = g.iloc[0]
                    out[f"eFour{hs}{tag}"] = _fmt(r.balanced_accuracy_mean,
                                                  r.balanced_accuracy_lo,
                                                  r.balanced_accuracy_hi)

    e5 = _read("e5_fusion_folds.csv")
    if e5 is not None:
        for task, key in (("super3", "eFiveBestSuperThree"), ("full7", "eFiveBestFullSeven")):
            d = e5[e5.task == task]
            if not len(d):
                continue
            by = d.groupby("arm").apply(lambda g: _mean_ci(g)[0], include_groups=False)
            best = by.idxmax()
            g = d[d.arm == best]
            m, lo, hi = _mean_ci(g)
            out[key] = _fmt(m, lo, hi) + f" ({_tex(best)})"
            if task == "super3":
                base = by[by.index.str.startswith("baseline")]
                if len(base):
                    out["eFiveGainSuperThree"] = \
                        f"{100 * (by.max() - base.iloc[0]):+.1f} percentage points"
    # ---- clinical metrics and the patient-level interval --------------------
    # The fold bootstrap resamples folds; this one resamples patients, which is
    # the quantity a reader assumes an interval bounds on a 45-patient cohort.
    cm = _read("clinical_metrics.csv")
    summ = _read("summary_e2_backbones.csv")
    if cm is not None and summ is not None:
        # Same restriction as the eTwoBest block above: the pooled-magnification
        # arm is the one every encoder shares, so the clinical metrics describe
        # the same configuration the headline figures do.
        d = summ[(summ.protocol == "patient") & (summ["head"] == "linear") &
                 (summ.magnification == "both")]
        for task, tag in (("super3", "SuperThree"), ("full7", "FullSeven"),
                          ("grade3", "GradeThree")):
            g = d[d.task == task]
            if not len(g):
                continue
            best = g.loc[g.balanced_accuracy_mean.idxmax()]
            sel = cm[(cm.task == task) & (cm.backbone == best.backbone)
                     & (cm.representation == best.representation)
                     & (cm.magnification == best.magnification)]
            if not len(sel):
                continue
            val = {r.metric: r for r in sel.itertuples()}
            for metric, key in (("sensitivity", "cmSens"), ("specificity", "cmSpec"),
                                ("precision", "cmPrec"), ("auc_ovr", "cmAuc")):
                if metric in val:
                    out[key + tag] = f"{val[metric].value:.3f}"
            if "balanced_accuracy" in val:
                r = val["balanced_accuracy"]
                out["cmPatient" + tag] = _fmt(r.value, r.lo, r.hi)
                fold_w = best.balanced_accuracy_hi - best.balanced_accuracy_lo
                pat_w = r.hi - r.lo
                if fold_w > 0:
                    out["cmWidthRatio" + tag] = f"{pat_w / fold_w:.1f}"

    # ---- BreakHis, the external replication cohort --------------------------
    bh = _read("breakhis_folds.csv")
    if bh is not None:
        d = bh[(bh["head"] == "linear") & (bh.representation == "tilemean")]
        for task, tag in (("bh_binary", "Binary"), ("bh_type8", "TypeEight")):
            g = d[d.task == task]
            if not len(g):
                continue
            vals = {}
            for proto, key in (("patient", "bhPatient"), ("image", "bhImage")):
                gg = g[g.protocol == proto]
                if not len(gg):
                    continue
                m, lo, hi = _mean_ci(gg)
                vals[proto] = m
                out[key + tag] = _fmt(m, lo, hi)
            if "patient" in vals and "image" in vals:
                out["bhInflation" + tag] = \
                    f"{100 * (vals['image'] - vals['patient']):+.1f} percentage points"
        if "n_test_patients" in bh.columns:
            tot = bh[(bh.protocol == "patient") & (bh.seed == bh.seed.min())
                     & (bh.task == bh.task.iloc[0])]
            if len(tot):
                out["bhPatients"] = str(int(tot.n_test_patients.sum()))
        idx = T / "breakhis_index.csv"
        if idx.exists():
            out["bhImages"] = f"{len(pd.read_csv(idx)):,}".replace(",", "{,}")

    # ---- the mechanism curve, replicated on the second cohort ---------------
    bh_mech = _read("mechanism_breakhis.csv")
    if bh_mech is not None:
        ps = (bh_mech.groupby(["cap", "seed", "protocol"]).balanced_accuracy
              .mean().reset_index())
        w2 = ps.pivot_table(index=["cap", "seed"], columns="protocol",
                            values="balanced_accuracy").reset_index()
        w2["infl"] = 100 * (w2["image"] - w2["patient"])
        cv = w2.groupby("cap").infl.mean()
        if 1 in cv.index:
            out["mechBhAtOne"] = f"{cv.loc[1]:+.1f} percentage points"
        out["mechBhFull"] = f"{cv.iloc[-1]:+.1f} percentage points"

    ctrl = _read("mechanism_control_lunghist700.csv")
    if ctrl is not None:
        ps = (ctrl.groupby(["per_group", "n_groups", "seed", "protocol"])
              .agg(ba=("balanced_accuracy", "mean"),
                   n=("n_images", "first")).reset_index())
        w3 = ps.pivot_table(index=["per_group", "n_groups", "seed", "n"],
                            columns="protocol", values="ba").reset_index()
        w3["infl"] = 100 * (w3["image"] - w3["patient"])
        cc = (w3.groupby(["per_group", "n_groups", "n"]).infl.mean()
              .reset_index().sort_values("per_group"))
        lo, hi = cc.iloc[0], cc.iloc[-1]
        out["mechCtrlLow"] = f"{lo.infl:+.1f} percentage points"
        out["mechCtrlHigh"] = f"{hi.infl:+.1f} percentage points"
        out["mechCtrlLowCell"] = f"{int(lo.per_group)} images from {int(lo.n_groups)} patients"
        out["mechCtrlHighCell"] = f"{int(hi.per_group)} images from {int(hi.n_groups)} patients"
        out["mechCtrlN"] = f"{int(cc.n.min())}--{int(cc.n.max())}"

    mech = _read("mechanism_lunghist700.csv")
    if mech is not None:
        per_seed = (mech.groupby(["cap", "seed", "protocol"])
                    .agg(ba=("balanced_accuracy", "mean"),
                         ipg=("images_per_group", "first")).reset_index())
        w = per_seed.pivot_table(index=["cap", "seed", "ipg"],
                                 columns="protocol", values="ba").reset_index()
        w["infl"] = 100 * (w["image"] - w["patient"])
        curve = (w.groupby(["cap", "ipg"]).infl.mean().reset_index()
                 .sort_values("ipg"))
        by_cap = dict(zip(curve.cap, curve.infl))
        full = curve.iloc[-1]
        for cap, key in ((1, "mechInflationAtOne"), (2, "mechInflationAtTwo"),
                         (4, "mechInflationAtFour")):
            if cap in by_cap:
                out[key] = f"{by_cap[cap]:+.1f} percentage points"
        out["mechInflationFull"] = f"{full.infl:+.1f} percentage points"
        out["mechImagesPerGroup"] = f"{full.ipg:.1f}"
        if 4 in by_cap and full.infl > 0:
            out["mechShareByFour"] = f"{by_cap[4] / full.infl:.0%}".replace("%", "\\%")
        nn = (mech[mech.protocol == "image"]
              .groupby("cap").nn_same_group.mean())
        if len(nn):
            out["mechNNFull"] = f"{nn.iloc[-1]:.2f}"
        if 2 in nn.index:
            out["mechNNAtTwo"] = f"{nn.loc[2]:.2f}"

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/numbers.tex"))
    a = ap.parse_args()

    have = collect()
    lines = ["% Generated by code/analysis/make_numbers.py -- do not edit.",
             "% Every figure quoted in the manuscript resolves through these macros,",
             "% so the prose cannot drift from results/tables.", ""]
    for name, desc in WANTED.items():
        val = have.get(name)
        body = val if val else f"\\PENDING{{{desc}}}"
        lines.append(f"\\newcommand{{\\{name}}}{{{body}}}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {len(have)}/{len(WANTED)} numbers available")


if __name__ == "__main__":
    main()
