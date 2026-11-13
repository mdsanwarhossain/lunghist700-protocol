"""Calibration and selective prediction.

A classifier that is to triage slides must say when it does not know. This
module reports expected calibration error, a reliability curve, and the
risk-coverage trade-off obtained by abstaining on low-confidence fields -
none of which the existing LungHist700 literature reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Maximum calibration error over non-empty bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            gaps.append(abs(correct[m].mean() - conf[m].mean()))
    return float(max(gaps)) if gaps else float("nan")


def reliability_table(conf: np.ndarray, correct: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "mean_confidence": conf[m].mean() if m.sum() else np.nan,
                     "empirical_accuracy": correct[m].mean() if m.sum() else np.nan})
    return pd.DataFrame(rows)


def risk_coverage(conf: np.ndarray, correct: np.ndarray,
                  grid: np.ndarray | None = None) -> pd.DataFrame:
    """Error rate among retained samples as the abstention threshold moves."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    order = np.argsort(-conf)
    c_sorted = correct[order]
    rows = []
    for cov in grid:
        k = max(1, int(round(cov * len(conf))))
        rows.append({"coverage": k / len(conf),
                     "selective_risk": 1.0 - c_sorted[:k].mean(),
                     "threshold": float(conf[order][k - 1]), "n_retained": k})
    df = pd.DataFrame(rows)
    # area under the risk-coverage curve: lower is a better uncertainty ranking
    df.attrs["aurc"] = float(np.trapezoid(df.selective_risk, df.coverage) /
                             (df.coverage.max() - df.coverage.min()))
    return df


def temperature_scale(logp: np.ndarray, y_idx: np.ndarray,
                      grid: np.ndarray | None = None) -> float:
    """Single-parameter temperature fitted by grid search on held-out log-probs."""
    grid = np.linspace(0.5, 5.0, 91) if grid is None else grid
    best, bestT = np.inf, 1.0
    for T in grid:
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        nll = -(z[np.arange(len(y_idx)), y_idx] -
                np.log(np.exp(z).sum(axis=1))).mean()
        if nll < best:
            best, bestT = nll, float(T)
    return bestT


def _apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-12, 1)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def held_out_temperature(proba: np.ndarray, y_idx: np.ndarray,
                         patient_id: np.ndarray, seed: int = 0
                         ) -> tuple[float, np.ndarray]:
    """Temperature fitted on one half of the patients and applied to the other.

    Fitting the temperature on the same predictions it is then scored on
    reports the best case a single parameter could ever reach, not the gain a
    deployed system would see. The patients are therefore split in two, each
    half is scaled by the temperature fitted on the other, and the returned
    temperature is the mean of the two fits.
    """
    rng = np.random.default_rng(seed)
    pats = np.unique(patient_id)
    rng.shuffle(pats)
    half = set(pats[: len(pats) // 2].tolist())
    a = np.array([p in half for p in patient_id])
    p_cal = np.empty_like(proba)
    temps = []
    for fit_mask, apply_mask in ((a, ~a), (~a, a)):
        if fit_mask.sum() < 10 or apply_mask.sum() == 0:
            p_cal[apply_mask] = proba[apply_mask]
            continue
        T = temperature_scale(np.log(np.clip(proba[fit_mask], 1e-12, 1)),
                              y_idx[fit_mask])
        temps.append(T)
        p_cal[apply_mask] = _apply_temperature(proba[apply_mask], T)
    return (float(np.mean(temps)) if temps else 1.0), p_cal


def analyse(preds: pd.DataFrame, classes: list[str]) -> dict:
    """Calibration and selective-prediction summary for one configuration's
    pooled out-of-fold predictions."""
    conf = preds.p_max.values.astype(float)
    correct = (preds.y_true.values == preds.y_pred.values).astype(float)
    pcols = [f"p_{c}" for c in classes if f"p_{c}" in preds.columns]
    proba = preds[pcols].values.astype(float)
    y_idx = np.array([pcols.index(f"p_{t}") if f"p_{t}" in pcols else 0
                      for t in preds.y_true])
    T_oracle = temperature_scale(np.log(np.clip(proba, 1e-12, 1)), y_idx)
    p_oracle = _apply_temperature(proba, T_oracle)
    pid = (preds.patient_id.values if "patient_id" in preds.columns
           else np.arange(len(preds)))
    T, p_cal = held_out_temperature(proba, y_idx, pid)
    rc = risk_coverage(conf, correct)
    return {
        "n": len(preds),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "overconfidence_gap": float(conf.mean() - correct.mean()),
        "ece": ece(conf, correct),
        "mce": mce(conf, correct),
        # scaled by a temperature fitted on held-out patients, which is what a
        # deployed calibrator could achieve, with the in-sample fit alongside
        # it as the unattainable upper bound
        "ece_temperature_scaled": ece(p_cal.max(axis=1), correct),
        "ece_temperature_scaled_insample": ece(p_oracle.max(axis=1), correct),
        "fitted_temperature": T,
        "fitted_temperature_insample": T_oracle,
        "aurc": rc.attrs["aurc"],
        "risk_at_80_coverage": float(rc.loc[(rc.coverage - 0.8).abs().idxmin(), "selective_risk"]),
        "risk_at_50_coverage": float(rc.loc[(rc.coverage - 0.5).abs().idxmin(), "selective_risk"]),
        "_reliability": reliability_table(conf, correct),
        "_risk_coverage": rc,
    }
