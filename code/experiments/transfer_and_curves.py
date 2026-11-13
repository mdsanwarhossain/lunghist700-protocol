"""Two experiments the dataset descriptor leaves open.

(1) Cross-magnification transfer. The descriptor reports 90% at 20x and 82% at
    40x without explaining the gap, and never asks whether a model trained at
    one magnification survives at the other - the situation a deployed tool
    meets when a laboratory changes objective.
(2) Patient-count learning curve. The descriptor's 45 patients are the binding
    constraint; quantifying the slope tells future collectors whether more
    patients or more fields per patient buys more accuracy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import protocols as P  # noqa: E402
from experiments.run_benchmark import FeatureStore  # noqa: E402
from models import heads as H  # noqa: E402


def _fit_eval(X, y, tr, te, head: str, seed: int, task: P.Task):
    model = H.HEADS[head](seed=seed).fit(X[tr], y[tr])
    proba = model.predict_proba(X[te])
    classes = list(model.classes_)
    pred = np.array(classes)[proba.argmax(1)]
    pm = np.zeros((len(pred), len(task.classes)))
    for j, c in enumerate(task.classes):
        if c in classes:
            pm[:, j] = proba[:, classes.index(c)]
    return P.evaluate(y[te], pred, pm, task.classes, ordinal=task.ordinal)


def magnification_transfer(store: FeatureStore, tasks: list[str], rep: str,
                           head: str, seeds: list[int], n_splits: int = 8):
    """Patient-disjoint folds are drawn once; the train side is then restricted
    to one magnification and the test side to one magnification, so matched and
    mismatched conditions share identical patient partitions."""
    index = store.index
    X_all = store.representation(rep)
    rows = []
    for tname in tasks:
        task = P.TASKS[tname]
        d = index.query(task.subset) if task.subset else index
        ridx = d.index.values
        y = d[task.label_col].fillna("").values.astype(str)
        keep = np.isin(y, task.classes)
        ridx, y = ridx[keep], y[keep]
        mag = index.loc[ridx, "magnification"].values
        pat = index.loc[ridx, "patient_id"].values
        X = X_all[ridx]

        for seed in seeds:
            for k, (tr, te) in enumerate(P.make_folds(y, pat, n_splits, seed)):
                for mtr in ("20x", "40x"):
                    for mte in ("20x", "40x"):
                        tr_m = tr[mag[tr] == mtr]
                        te_m = te[mag[te] == mte]
                        if len(tr_m) < 30 or len(te_m) < 10:
                            continue
                        if len(np.unique(y[tr_m])) < len(task.classes):
                            continue
                        m = _fit_eval(X, y, tr_m, te_m, head, seed, task)
                        rows.append(dict(
                            backbone=store.backbone, norm=store.norm, task=tname,
                            representation=rep, head=head, train_mag=mtr,
                            test_mag=mte, matched=(mtr == mte), seed=seed, fold=k,
                            n_train=len(tr_m), n_test=len(te_m),
                            patient_overlap=P.check_disjoint(pat[tr_m], pat[te_m]),
                            **m))
    return pd.DataFrame(rows)


def learning_curve(store: FeatureStore, tasks: list[str], rep: str, head: str,
                   seeds: list[int], fractions=(0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
                   n_splits: int = 8):
    """Subsample *patients* (not images) from the training side of each
    patient-disjoint fold, keeping the test side fixed."""
    index = store.index
    X_all = store.representation(rep)
    rows = []
    for tname in tasks:
        task = P.TASKS[tname]
        d = index.query(task.subset) if task.subset else index
        ridx = d.index.values
        y = d[task.label_col].fillna("").values.astype(str)
        keep = np.isin(y, task.classes)
        ridx, y = ridx[keep], y[keep]
        pat = index.loc[ridx, "patient_id"].values
        X = X_all[ridx]

        for seed in seeds:
            rng = np.random.default_rng(seed)
            for k, (tr, te) in enumerate(P.make_folds(y, pat, n_splits, seed)):
                tr_pat = np.unique(pat[tr])
                for frac in fractions:
                    n_keep = max(len(task.classes), int(round(frac * len(tr_pat))))
                    sel_pat = rng.choice(tr_pat, size=min(n_keep, len(tr_pat)),
                                         replace=False)
                    tr_s = tr[np.isin(pat[tr], sel_pat)]
                    if len(np.unique(y[tr_s])) < len(task.classes) or len(tr_s) < 30:
                        continue
                    m = _fit_eval(X, y, tr_s, te, head, seed, task)
                    rows.append(dict(
                        backbone=store.backbone, norm=store.norm, task=tname,
                        representation=rep, head=head, fraction=frac,
                        n_train_patients=len(sel_pat), n_train=len(tr_s),
                        n_test=len(te), seed=seed, fold=k, **m))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("results/tables/image_index.csv"))
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--norm", default="raw")
    ap.add_argument("--representation", default="tilemean")
    ap.add_argument("--head", default="linear")
    ap.add_argument("--tasks", nargs="+", default=["super3", "full7"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir", type=Path, default=Path("results/tables"))
    ap.add_argument("--which", nargs="+", default=["transfer", "curve"])
    # k is bounded by the scarcest class: 8 distinct patients for full7.
    ap.add_argument("--n-splits", type=int, default=8)
    a = ap.parse_args()

    index = pd.read_csv(a.index)
    store = FeatureStore(a.backbone, a.norm, index)
    a.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.backbone}__{a.norm}"
    if "transfer" in a.which:
        t = magnification_transfer(store, a.tasks, a.representation, a.head,
                                   a.seeds, n_splits=a.n_splits)
        t.to_csv(a.outdir / f"magnification_transfer_{tag}.csv", index=False)
        print("\n-- cross-magnification balanced accuracy --")
        print(t.groupby(["task", "train_mag", "test_mag"]).balanced_accuracy
              .agg(["mean", "std", "count"]).round(3).to_string())
    if "curve" in a.which:
        c = learning_curve(store, a.tasks, a.representation, a.head,
                           a.seeds, n_splits=a.n_splits)
        c.to_csv(a.outdir / f"learning_curve_{tag}.csv", index=False)
        print("\n-- learning curve (balanced accuracy vs train patients) --")
        print(c.groupby(["task", "fraction"])[["n_train_patients", "balanced_accuracy"]]
              .mean().round(3).to_string())
