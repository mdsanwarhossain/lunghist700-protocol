"""Evaluation protocols, metrics and splitters.

The central methodological claim of this work is that the *protocol*, not the
architecture, dominates reported accuracy on LungHist700. The three splitters
below are therefore first-class objects, applied to identical features and
identical classifiers so that the only thing that varies is what is allowed to
be shared between train and test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
SUPERCLASSES = ["aca", "nor", "scc"]
SEVEN = ["aca_bd", "aca_md", "aca_pd", "nor", "scc_bd", "scc_md", "scc_pd"]
GRADES = ["bd", "md", "pd"]

# Ordinal position of each 7-class label along the differentiation axis, used
# for the grade-aware (quadratic-weighted) agreement statistic. Normal tissue
# is ungraded and is excluded from the ordinal view rather than given a rank.
SEVEN_ORDINAL = {"aca_bd": 0, "aca_md": 1, "aca_pd": 2,
                 "scc_bd": 0, "scc_md": 1, "scc_pd": 2}

# BreakHis, the external replication cohort: two benign/malignant superclasses
# and eight tumour types. Kept beside the LungHist700 label sets rather than in
# a separate module so that one TASKS registry serves both cohorts.
BH_SUPERCLASSES = ["benign", "malignant"]
BH_TYPES = ["adenosis", "ductal", "fibroadenoma", "lobular",
            "mucinous", "papillary", "phyllodes", "tubular-adenoma"]


@dataclass(frozen=True)
class Task:
    name: str
    label_col: str
    classes: tuple[str, ...]
    ordinal: bool = False
    subset: str | None = None  # pandas query restricting the rows used


TASKS = {
    # 3-superclass diagnosis: the only task the dataset descriptor reported
    "super3": Task("super3", "superclass", tuple(SUPERCLASSES)),
    # full 7-class subtype+grade: published for the first time here
    "full7": Task("full7", "label7", tuple(SEVEN)),
    # differentiation grade within carcinoma: intrinsically ordinal
    "grade3": Task("grade3", "subclass", tuple(GRADES), ordinal=True,
                   subset="superclass != 'nor'"),

    # --- external replication cohort (BreakHis) ---------------------------
    # The same protocol contrast, run on a second dataset that also ships a
    # patient identifier. Coarse and fine tasks mirror super3 and full7.
    "bh_binary": Task("bh_binary", "superclass", tuple(BH_SUPERCLASSES)),
    "bh_type8": Task("bh_type8", "label7", tuple(BH_TYPES)),
}


# --------------------------------------------------------------------------
# Splitters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    name: str
    group_col: str | None
    description: str


PROTOCOLS = {
    # Correct: no patient may appear on both sides of the split.
    "patient": Protocol("patient", "patient_id",
                        "patient-disjoint, class-stratified folds"),
    # Leaky-1: images are split independently, so several fields from one
    # patient (and one slide) straddle the split. Widely used in this literature.
    "image": Protocol("image", None,
                      "image-level stratified folds, patients may straddle"),
    # Leaky-2: tiles from one image straddle the split, the LC25000-style
    # augmentation-traceability failure reproduced on real data.
    "tile": Protocol("tile", None,
                     "tile-level stratified folds, tiles of one image straddle"),
}


def make_folds(y: np.ndarray, groups: np.ndarray | None, n_splits: int,
               seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Class-stratified folds, patient-disjoint when ``groups`` is given."""
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups=groups))


def check_disjoint(train_g: np.ndarray, test_g: np.ndarray) -> int:
    """Number of groups shared across a split; 0 for a sound patient-wise fold."""
    return len(set(train_g.tolist()) & set(test_g.tolist()))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None,
             classes: tuple[str, ...], ordinal: bool = False) -> dict[str, float]:
    """Metric panel. Balanced accuracy and MCC lead because plain accuracy is
    inflated by the dominant classes on this imbalanced dataset."""
    m: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if ordinal:
        # Confusing well- with moderately-differentiated is a smaller clinical
        # error than confusing well- with poorly-differentiated; QWK is the
        # standard statistic that encodes that ordering.
        m["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                     labels=list(classes))
    if proba is not None and len(classes) > 1:
        try:
            m["auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                         average="macro", labels=list(classes))
        except ValueError:
            m["auc_ovr"] = float("nan")
    return m


def graded_confusion_summary(y_true, y_pred) -> dict[str, float]:
    """For the 7-class task: how errors distribute over clinically distinct
    failure types. Subtype errors change treatment; adjacent-grade errors are
    the documented zone of inter-observer disagreement."""
    def parts(lbl: str) -> tuple[str, str]:
        return (lbl, "") if lbl == "nor" else tuple(lbl.split("_"))

    n = len(y_true)
    wrong_subtype = wrong_grade_adj = wrong_grade_far = malig_vs_normal = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        ts, tg = parts(t)
        ps, pg = parts(p)
        if (ts == "nor") != (ps == "nor"):
            malig_vs_normal += 1
        elif ts != ps:
            wrong_subtype += 1
        else:
            d = abs(SEVEN_ORDINAL[t] - SEVEN_ORDINAL[p])
            if d == 1:
                wrong_grade_adj += 1
            else:
                wrong_grade_far += 1
    return {
        "err_normal_vs_malignant": malig_vs_normal / n,
        "err_subtype": wrong_subtype / n,
        "err_grade_adjacent": wrong_grade_adj / n,
        "err_grade_distant": wrong_grade_far / n,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval over fold scores."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))
