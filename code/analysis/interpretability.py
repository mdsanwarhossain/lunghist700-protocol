"""Quantitative attribution evaluation and patient-level reporting.

The dataset descriptor validates its explanations qualitatively: Grad-CAM masks
are shown and "cross-checked with the medical team". That cannot distinguish an
attribution that drives the prediction from one that merely looks plausible.
Here attention over tissue tiles is scored by deletion and insertion curves
against a random-order control, which is a falsifiable test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def deletion_insertion(model, bag: np.ndarray, class_idx: int,
                       order: np.ndarray, n_steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Probability of ``class_idx`` as tiles are removed in ``order`` (deletion)
    and as they are added in ``order`` (insertion).

    A faithful ranking makes deletion fall fast and insertion rise fast.
    """
    n = len(bag)
    ks = np.unique(np.linspace(0, n, n_steps + 1).astype(int))
    dele, ins = [], []
    for k in ks:
        keep = order[k:]                      # drop the k highest-ranked tiles
        sub = bag[keep] if len(keep) else bag[order[-1:]]
        dele.append(model.predict_proba([sub])[0, class_idx])
        add = order[:k] if k > 0 else order[:1]
        ins.append(model.predict_proba([bag[add]])[0, class_idx])
    return np.asarray(dele), np.asarray(ins)


def attribution_faithfulness(model, bags: list[np.ndarray], attentions: list[np.ndarray],
                             y_true: np.ndarray, classes: list[str],
                             n_images: int = 60, seed: int = 0) -> pd.DataFrame:
    """Compare attention-ordered against randomly-ordered tile removal.

    ``auc_deletion`` lower is better and ``auc_insertion`` higher is better; the
    random control sets the level that a non-informative ranking would reach.
    """
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(bags), size=min(n_images, len(bags)), replace=False)
    rows = []
    for i in sel:
        bag, att = bags[i], attentions[i]
        if len(bag) < 6:
            continue
        ci = classes.index(y_true[i]) if y_true[i] in classes else 0
        for mode in ("attention", "random"):
            order = (np.argsort(-att) if mode == "attention"
                     else rng.permutation(len(bag)))
            d, s = deletion_insertion(model, bag, ci, order)
            rows.append({
                "image": int(i), "mode": mode, "label": y_true[i],
                "n_tiles": len(bag),
                "auc_deletion": float(np.trapezoid(d, dx=1 / (len(d) - 1))),
                "auc_insertion": float(np.trapezoid(s, dx=1 / (len(s) - 1))),
                "p_full": float(d[0]),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["faithfulness_gap"] = df.auc_insertion - df.auc_deletion
    return df


def attention_concentration(attentions: list[np.ndarray],
                            labels: np.ndarray) -> pd.DataFrame:
    """How peaked the attention is per class.

    Normalised entropy near 1 means the head spreads credit over all tissue,
    near 0 means a handful of tiles decide the field - which matters when the
    claim is that the model localises diagnostic morphology.
    """
    rows = []
    for a, lab in zip(attentions, labels):
        a = np.asarray(a, float)
        a = a / max(a.sum(), 1e-12)
        ent = -(a * np.log(np.maximum(a, 1e-12))).sum()
        rows.append({
            "label": lab, "n_tiles": len(a),
            "normalised_entropy": float(ent / np.log(len(a))) if len(a) > 1 else np.nan,
            "top1_share": float(a.max()),
            "top5_share": float(np.sort(a)[-5:].sum()) if len(a) >= 5 else float(a.sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Patient-level reporting
# --------------------------------------------------------------------------
def aggregate_to_patient(preds: pd.DataFrame, classes: list[str],
                         rule: str = "mean_proba") -> pd.DataFrame:
    """Collapse per-field predictions to one call per (patient, fold, seed).

    A pathologist signs out a case, not a field of view, so the clinically
    meaningful error rate is the case-level one. Fields of a patient that carry
    different reference labels are kept as separate cases.
    """
    pcols = [f"p_{c}" for c in classes if f"p_{c}" in preds.columns]
    out = []
    for (seed, fold, pid, yt), g in preds.groupby(["seed", "fold", "patient_id", "y_true"]):
        if rule == "mean_proba":
            score = g[pcols].mean().values
            call = pcols[int(np.argmax(score))][2:]
            conf = float(score.max())
        elif rule == "majority":
            vc = g.y_pred.value_counts()
            call = vc.index[0]
            conf = float(vc.iloc[0] / len(g))
        else:
            raise KeyError(rule)
        out.append({"seed": seed, "fold": fold, "patient_id": pid, "y_true": yt,
                    "y_pred": call, "n_fields": len(g), "confidence": conf,
                    "correct": int(call == yt), "rule": rule})
    return pd.DataFrame(out)


def patient_vs_field(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Field-level against case-level accuracy, fold by fold."""
    rows = []
    for rule in ("mean_proba", "majority"):
        pat = aggregate_to_patient(preds, classes, rule)
        for (seed, fold), g in pat.groupby(["seed", "fold"]):
            f = preds[(preds.seed == seed) & (preds.fold == fold)]
            # macro over reference classes, so a dominant class cannot carry it
            bal_case = np.nanmean([gg.correct.mean() for _, gg in g.groupby("y_true")])
            bal_field = np.nanmean([(gg.y_true == gg.y_pred).mean()
                                    for _, gg in f.groupby("y_true")])
            rows.append({"rule": rule, "seed": seed, "fold": fold,
                         "n_cases": len(g),
                         "case_balanced_accuracy": bal_case,
                         "field_balanced_accuracy": bal_field,
                         "delta": bal_case - bal_field})
    return pd.DataFrame(rows)
