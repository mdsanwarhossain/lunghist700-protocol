"""Does the attention actually carry the prediction?

The dataset descriptor validates its explanations by showing Grad-CAM maps and
having them "cross-checked with the medical team". A map that looks plausible
is not evidence that it is what the model used, so here the attention-MIL
head's own tile ranking is tested by deletion and insertion against a
random-order control on the same bags and the same fitted model. A ranking
that carries the decision must beat the control; one that does not is
decoration.

Only held-out bags are used, so the ranking is never tested on a field the head
was fitted on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import interpretability as I  # noqa: E402
from experiments import protocols as P  # noqa: E402
from experiments.run_benchmark import FeatureStore  # noqa: E402
from models import heads as H  # noqa: E402


def run(store: FeatureStore, task_name: str, seeds: list[int], n_splits: int,
        n_images: int, epochs: int):
    task = P.TASKS[task_name]
    index = store.index
    d = index.query(task.subset) if task.subset else index
    rows_all = d.index.values
    y_all = d[task.label_col].fillna("").values.astype(str)
    keep = np.isin(y_all, task.classes)
    rows_all, y_all = rows_all[keep], y_all[keep]
    pat = index.loc[rows_all, "patient_id"].values

    faith, conc = [], []
    for seed in seeds:
        for k, (tr, te) in enumerate(P.make_folds(y_all, pat, n_splits, seed)):
            model = H.AttentionMILHead(seed=seed, epochs=epochs)
            model.fit(store.bags(rows_all[tr]), y_all[tr])
            bags_te = store.bags(rows_all[te])
            proba, atts = model.predict_proba(bags_te, return_attention=True)
            classes = list(model.classes_)

            f = I.attribution_faithfulness(model, bags_te, atts, y_all[te],
                                           classes, n_images=n_images, seed=seed)
            c = I.attention_concentration(atts, y_all[te])
            for df, sink in ((f, faith), (c, conc)):
                if len(df):
                    df.insert(0, "task", task_name)
                    df.insert(1, "backbone", store.backbone)
                    df.insert(2, "seed", seed)
                    df.insert(3, "fold", k)
                    sink.append(df)
            acc = (np.array(classes)[proba.argmax(1)] == y_all[te]).mean()
            print(f"  [{store.backbone}/{task_name}] seed {seed} fold {k}: "
                  f"acc={acc:.3f}  {len(bags_te)} held-out bags", flush=True)
    return (pd.concat(faith, ignore_index=True) if faith else pd.DataFrame(),
            pd.concat(conc, ignore_index=True) if conc else pd.DataFrame())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("results/tables/image_index.csv"))
    ap.add_argument("--backbone", default="phikon_vitb16")
    ap.add_argument("--norm", default="raw")
    ap.add_argument("--tasks", nargs="+", default=["full7"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    # matches the k used by every other stage; see the audit bound
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--n-images", type=int, default=25,
                    help="held-out bags probed per fold")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--outdir", type=Path, default=Path("results/tables"))
    a = ap.parse_args()

    index = pd.read_csv(a.index)
    store = FeatureStore(a.backbone, a.norm, index)
    a.outdir.mkdir(parents=True, exist_ok=True)
    for t in a.tasks:
        f, c = run(store, t, a.seeds, a.n_splits, a.n_images, a.epochs)
        tag = f"{a.backbone}__{a.norm}_{t}"
        if len(f):
            f.to_csv(a.outdir / f"attribution_faithfulness_{tag}.csv", index=False)
            print("\n-- deletion / insertion AUC, attention vs random order --")
            print(f.groupby("mode")[["auc_deletion", "auc_insertion",
                                     "faithfulness_gap"]].mean().round(4).to_string())
        if len(c):
            c.to_csv(a.outdir / f"attention_concentration_{tag}.csv", index=False)
            print("\n-- attention concentration per class --")
            print(c.groupby("label")[["normalised_entropy", "top1_share",
                                      "top5_share"]].mean().round(3).to_string())
