"""Turn per-fold records into reportable estimates with uncertainty.

Fold scores from repeated cross-validation are not independent, so a naive
paired t-test over folds is anti-conservative. Comparisons therefore use the
Nadeau-Bengio corrected resampled t-test alongside a distribution-free
Wilcoxon test, and intervals come from a percentile bootstrap over folds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONFIG_KEYS = ["backbone", "norm", "task", "protocol", "representation",
               "head", "magnification", "unit"]
METRICS = ["balanced_accuracy", "macro_f1", "mcc", "accuracy", "auc_ovr",
           "kappa", "qwk", "weighted_f1"]


def bootstrap_ci(v: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 - 100 * alpha / 2))


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in CONFIG_KEYS if k in folds.columns]
    out = []
    for vals, g in folds.groupby(keys, dropna=False):
        rec = dict(zip(keys, vals if isinstance(vals, tuple) else (vals,)))
        rec["n_folds"] = len(g)
        rec["n_test_median"] = g.n_test.median()
        rec["patient_overlap_max"] = g.patient_overlap.max()
        rec["image_overlap_max"] = g.image_overlap.max()
        for m in METRICS:
            if m not in g.columns or g[m].isna().all():
                continue
            v = g[m].dropna().values
            lo, hi = bootstrap_ci(v)
            rec[f"{m}_mean"] = v.mean()
            rec[f"{m}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            rec[f"{m}_lo"] = lo
            rec[f"{m}_hi"] = hi
        for m in ["err_normal_vs_malignant", "err_subtype",
                  "err_grade_adjacent", "err_grade_distant"]:
            if m in g.columns and not g[m].isna().all():
                rec[f"{m}_mean"] = g[m].dropna().mean()
        out.append(rec)
    return pd.DataFrame(out)


def corrected_t(a: np.ndarray, b: np.ndarray, n_train: float, n_test: float):
    """Nadeau-Bengio corrected resampled t-test for repeated k-fold scores."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    se = d.std(ddof=1) * np.sqrt(1.0 / n + (n_test / max(n_train, 1)))
    t = d.mean() / max(se, 1e-12)
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def compare(folds: pd.DataFrame, metric: str, keys: list[str],
            baseline: dict, contrasts: list[dict]) -> pd.DataFrame:
    """Paired comparison of each contrast against a baseline configuration."""
    def pick(sel: dict) -> pd.DataFrame:
        m = np.ones(len(folds), bool)
        for k, v in sel.items():
            m &= (folds[k] == v).values
        return folds[m].sort_values(["seed", "fold"])

    base = pick(baseline)
    rows = []
    for c in contrasts:
        cur = pick({**baseline, **c})
        if base.empty or cur.empty:
            continue
        j = ["seed", "fold"]
        # Folds are paired by (seed, fold), so each side must hold exactly one
        # score per pair. An under-specified selection - a baseline that still
        # spans several representations, say - would otherwise cross-join here
        # and silently average unrelated configurations into the contrast.
        for side, df in (("baseline", base), ("contrast", cur)):
            dup = df.duplicated(subset=j).sum()
            if dup:
                raise ValueError(
                    f"{side} selection {baseline if side == 'baseline' else {**baseline, **c}} "
                    f"is not unique per (seed, fold): {dup} duplicate pairs. "
                    "Add the missing configuration keys to the selection.")
        mg = base[j + [metric, "n_train", "n_test"]].merge(
            cur[j + [metric]], on=j, suffixes=("_base", "_cur"),
            validate="one_to_one")
        if len(mg) < 2:
            continue
        a, b = mg[f"{metric}_cur"].values, mg[f"{metric}_base"].values
        t, p_t = corrected_t(a, b, mg.n_train.mean(), mg.n_test.mean())
        try:
            _, p_w = stats.wilcoxon(a, b)
        except ValueError:
            p_w = np.nan
        diff = a - b
        sd = diff.std(ddof=1)
        rows.append({
            **c, "metric": metric, "n_pairs": len(mg),
            "base_mean": b.mean(), "cur_mean": a.mean(), "delta": diff.mean(),
            "delta_lo": bootstrap_ci(diff)[0], "delta_hi": bootstrap_ci(diff)[1],
            "cohens_d": diff.mean() / sd if sd > 0 else np.nan,
            "t_corrected": t, "p_corrected_t": p_t, "p_wilcoxon": p_w,
        })
    df = pd.DataFrame(rows)
    if len(df):  # Holm-Bonferroni across the contrasts of one comparison
        o = np.argsort(df.p_corrected_t.values)
        adj = np.empty(len(df))
        run = 0.0
        for r, i in enumerate(o):
            run = max(run, (len(df) - r) * df.p_corrected_t.values[i])
            adj[i] = min(1.0, run)
        df["p_holm"] = adj
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in a.folds], ignore_index=True)
    s = summarize(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out, index=False)
    print(f"{len(s)} configurations summarised -> {a.out}")
