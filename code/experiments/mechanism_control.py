"""Control: is the inflation driven by images per group, or just by sample size?

Capping images per group also shrinks the dataset, so the sweep in
``mechanism.py`` varies two things at once. This control holds the total number
of images roughly fixed and trades groups against images per group: many groups
with few images each, against few groups with many. If the inflation still
rises with images per group at constant n, sample size is not the explanation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import protocols as P  # noqa: E402
from experiments.run_benchmark import FeatureStore, set_featroot  # noqa: E402
from models.heads import LinearHead  # noqa: E402


def _sample(idx: pd.DataFrame, per_group: int, n_groups: int,
            seed: int) -> np.ndarray | None:
    """``n_groups`` groups contributing ``per_group`` images each."""
    rng = np.random.default_rng(10_000 * seed + 100 * per_group + n_groups)
    big = [g for g, r in idx.groupby("patient_id").groups.items()
           if len(r) >= per_group]
    if len(big) < n_groups:
        return None
    chosen = rng.choice(big, n_groups, replace=False)
    out = []
    for g in chosen:
        r = np.asarray(idx.groupby("patient_id").groups[g])
        out.append(rng.choice(r, per_group, replace=False))
    return np.sort(np.concatenate(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--featroot", type=Path, default=Path("data/features"))
    ap.add_argument("--backbone", default="resnet50_in1k")
    ap.add_argument("--label-col", default="superclass")
    ap.add_argument("--cells", type=str, nargs="+", required=True,
                    help="per_group:n_groups pairs, chosen to match total n")
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cohort", default="lunghist700")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    idx = pd.read_csv(a.index).reset_index(drop=True)
    set_featroot(a.featroot)
    store = FeatureStore(a.backbone, "raw", idx)
    X_all = store.representation("tilemean")
    y_all = idx[a.label_col].astype(str).values
    g_all = idx.patient_id.values

    rows = []
    for cell in a.cells:
        per_group, n_groups = (int(v) for v in cell.split(":"))
        for seed in a.seeds:
            sub = _sample(idx, per_group, n_groups, seed)
            if sub is None:
                print(f"  {cell}: not enough groups with {per_group} images")
                break
            X, y, g = X_all[sub], y_all[sub], g_all[sub]
            if len(np.unique(y)) < 2:
                continue
            for protocol, groups in (("patient", g), ("image", None)):
                try:
                    folds = P.make_folds(y, groups, a.n_splits, seed)
                except ValueError:
                    continue
                for k, (tr, te) in enumerate(folds):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    m = LinearHead(seed=seed).fit(X[tr], y[tr])
                    pred = m.classes_[m.predict_proba(X[te]).argmax(1)]
                    rows.append(dict(
                        cohort=a.cohort, per_group=per_group,
                        n_groups=n_groups, n_images=len(sub), seed=seed,
                        fold=k, protocol=protocol,
                        balanced_accuracy=balanced_accuracy_score(y[te], pred)))
        d = [r for r in rows if r["per_group"] == per_group
             and r["n_groups"] == n_groups]
        if d:
            pat = np.mean([r["balanced_accuracy"] for r in d
                           if r["protocol"] == "patient"])
            img = np.mean([r["balanced_accuracy"] for r in d
                           if r["protocol"] == "image"])
            print(f"{per_group:>3} images x {n_groups:>3} groups "
                  f"= {d[0]['n_images']:>4} images   grouped {pat:.3f}"
                  f"  ungrouped {img:.3f}   inflation {100*(img-pat):+5.1f} pp",
                  flush=True)

    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
