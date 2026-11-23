"""Macro sensitivity, specificity, precision and F1, with patient-level intervals.

The fold tables already carry balanced accuracy, MCC, macro F1 and one-versus-rest
AUC. Clinical readers expect sensitivity and specificity as well, and they expect
an interval that bounds *patient* sampling rather than fold sampling.

Both are computed here from the stored out-of-fold predictions, so nothing is
re-fitted. Every per-class count decomposes over patients, which makes the
patient bootstrap a weighted sum of per-patient count vectors rather than a
re-evaluation: resample patients with replacement, carry all of that patient's
images and all seeds with them, and recompute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

B_RESAMPLES = 10000
SEED = 0


def _counts(df: pd.DataFrame, classes: list[str]) -> dict[str, np.ndarray]:
    """Per-patient, per-class count matrices; every metric below is a ratio of
    sums of these, so a patient bootstrap only reweights the rows."""
    pats = np.sort(df.patient_id.unique())
    pix = {p: i for i, p in enumerate(pats)}
    cix = {c: i for i, c in enumerate(classes)}
    P, C = len(pats), len(classes)
    tp = np.zeros((P, C))      # true class c and predicted c
    sup = np.zeros((P, C))     # true class c                  (TP + FN)
    pred = np.zeros((P, C))    # predicted c                   (TP + FP)
    neg = np.zeros((P, C))     # true class != c               (TN + FP)
    tn = np.zeros((P, C))      # true != c and predicted != c  (TN)
    for pid, yt, yp in zip(df.patient_id.values, df.y_true.values, df.y_pred.values):
        i = pix[pid]
        ti, pi = cix.get(yt), cix.get(yp)
        if ti is None:
            continue
        sup[i, ti] += 1
        if pi is not None:
            pred[i, pi] += 1
            if ti == pi:
                tp[i, ti] += 1
        for c in range(C):
            if c != ti:
                neg[i, c] += 1
                if pi != c:
                    tn[i, c] += 1
    return {"tp": tp, "sup": sup, "pred": pred, "neg": neg, "tn": tn, "n_patients": P}


def _metrics(w: np.ndarray, m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Macro metrics for one or many patient-weight vectors. ``w`` is (B, P)."""
    tp, sup, pred = w @ m["tp"], w @ m["sup"], w @ m["pred"]
    neg, tn = w @ m["neg"], w @ m["tn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)          # sensitivity / recall
        spec = np.where(neg > 0, tn / neg, np.nan)
        prec = np.where(pred > 0, tp / pred, np.nan)
        f1 = np.where((rec + prec) > 0, 2 * rec * prec / (rec + prec), 0.0)
    return {
        "sensitivity": np.nanmean(rec, axis=1),
        "specificity": np.nanmean(spec, axis=1),
        "precision": np.nanmean(prec, axis=1),
        "macro_f1": np.nanmean(f1, axis=1),
        "balanced_accuracy": np.nanmean(rec, axis=1),  # identical by definition
        "accuracy": tp.sum(axis=1) / np.maximum(sup.sum(axis=1), 1),
    }


def evaluate(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    m = _counts(df, classes)
    P = m["n_patients"]
    point = _metrics(np.ones((1, P)), m)
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, P, size=(B_RESAMPLES, P))
    w = np.zeros((B_RESAMPLES, P))
    np.add.at(w, (np.repeat(np.arange(B_RESAMPLES), P), draw.ravel()), 1.0)
    boot = _metrics(w, m)
    rows = []
    for k in point:
        b = boot[k][np.isfinite(boot[k])]
        rows.append({"metric": k, "value": float(point[k][0]),
                     "lo": float(np.percentile(b, 2.5)),
                     "hi": float(np.percentile(b, 97.5)),
                     "n_patients": P, "n_rows": len(df)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, nargs="+",
                    default=[Path("results/tables/e2_backbones_preds.csv")])
    ap.add_argument("--out", type=Path,
                    default=Path("results/tables/clinical_metrics.csv"))
    ap.add_argument("--folds", type=Path,
                    default=Path("results/tables/e2_backbones_folds.csv"),
                    help="source of the fold-level AUC, which needs probabilities")
    a = ap.parse_args()

    pred = pd.concat([pd.read_csv(p) for p in a.preds], ignore_index=True)
    # NB: ``.head`` is a DataFrame method, so the column must be indexed by name.
    pred = pred[(pred.protocol == "patient") & (pred["head"] == "linear")
                & (pred.unit == "image")]

    out = []
    for task, g_task in pred.groupby("task"):
        classes = sorted(set(g_task.y_true.dropna().unique()))
        for (bb, rep, mag), g in g_task.groupby(["backbone", "representation",
                                                 "magnification"]):
            r = evaluate(g, classes)
            r.insert(0, "magnification", mag)
            r.insert(0, "representation", rep)
            r.insert(0, "backbone", bb)
            r.insert(0, "task", task)
            out.append(r)
    res = pd.concat(out, ignore_index=True)

    # Fold-level AUC, carried through so the clinical table can quote it.
    if a.folds.exists():
        f = pd.read_csv(a.folds)
        f = f[(f.protocol == "patient") & (f["head"] == "linear") & (f.unit == "image")]
        auc = (f.groupby(["task", "backbone", "representation", "magnification"])
                 .auc_ovr.agg(["mean", "count"]).reset_index())
        auc = auc[auc["count"] > 0]
        auc = auc.rename(columns={"mean": "value"})
        auc["metric"] = "auc_ovr"
        auc["lo"] = np.nan
        auc["hi"] = np.nan
        res = pd.concat([res, auc.drop(columns="count")], ignore_index=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(res)} rows, {res.task.nunique()} tasks, "
          f"{res.backbone.nunique()} backbones)")


if __name__ == "__main__":
    main()
