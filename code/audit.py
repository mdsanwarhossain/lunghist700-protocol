"""Split-sensitivity audit: how much of a benchmark's accuracy is the partition?

Run it on any image dataset that ships a grouping variable (patient, slide,
subject, augmentation group). It encodes the images once with a frozen encoder,
fits an identical linear head under a grouped and an ungrouped partition, and
reports the difference together with an interval bootstrapped over groups.

    python code/audit.py --index my_index.csv --out results/my_audit

``--index`` is a CSV with at least:

    path        path to the image file
    group       the grouping variable (patient id, slide id, ...)
    label       the class label

Column names are configurable with --group-col / --label-col / --path-col.
Nothing is trained: the encoder stays frozen, so this runs on a CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import protocols as P  # noqa: E402
from models.heads import LinearHead  # noqa: E402


def _normalise(idx: pd.DataFrame, path_col: str, group_col: str,
               label_col: str) -> pd.DataFrame:
    """Rename a user index into the schema the cached feature store expects."""
    missing = [c for c in (path_col, group_col, label_col) if c not in idx.columns]
    if missing:
        raise SystemExit(f"index is missing column(s): {', '.join(missing)}")
    out = pd.DataFrame({
        "path": idx[path_col].astype(str),
        "exists": [Path(p).is_file() for p in idx[path_col]],
        "label7": idx[label_col].astype(str),
        "superclass": idx[label_col].astype(str),
        "subclass": idx[label_col].astype(str),
        "magnification": "both",
        "grade": -1,
    })
    groups = sorted(idx[group_col].astype(str).unique())
    out["patient_id"] = idx[group_col].astype(str).map(
        {g: i for i, g in enumerate(groups)}).values
    out["group_label"] = idx[group_col].astype(str).values
    if not out.exists.all():
        raise SystemExit(f"{(~out.exists).sum()} of {len(out)} image paths do not exist")
    return out


def _group_bootstrap(per_image: pd.DataFrame, classes: list[str],
                     n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Balanced accuracy with an interval over groups, not folds.

    Every per-class count decomposes over groups, so resampling groups is a
    reweighting of per-group count vectors rather than a re-evaluation.
    """
    pats = np.sort(per_image.patient_id.unique())
    pix = {p: i for i, p in enumerate(pats)}
    cix = {c: i for i, c in enumerate(classes)}
    hit = np.zeros((len(pats), len(classes)))
    sup = np.zeros((len(pats), len(classes)))
    for g, t, pr in zip(per_image.patient_id, per_image.y_true, per_image.y_pred):
        if t not in cix:
            continue
        i, j = pix[g], cix[t]
        sup[i, j] += 1
        if t == pr:
            hit[i, j] += 1

    def bacc(w: np.ndarray) -> np.ndarray:
        h, s = w @ hit, w @ sup
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = np.where(s > 0, h / s, np.nan)
        return np.nanmean(rec, axis=1)

    point = float(bacc(np.ones((1, len(pats))))[0])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(pats), size=(n, len(pats)))
    w = np.zeros((n, len(pats)))
    np.add.at(w, (np.repeat(np.arange(n), len(pats)), draw.ravel()), 1.0)
    b = bacc(w)
    b = b[np.isfinite(b)]
    return point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output prefix; writes <out>_folds.csv and <out>_summary.csv")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--group-col", default="group")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--backbone", default="resnet50_in1k")
    ap.add_argument("--featroot", type=Path, default=Path("data/features_audit"))
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--skip-encode", action="store_true",
                    help="reuse an existing cache under --featroot")
    a = ap.parse_args()

    idx = _normalise(pd.read_csv(a.index), a.path_col, a.group_col, a.label_col)
    n_groups = idx.patient_id.nunique()
    classes = sorted(idx.label7.unique())
    print(f"=== split-sensitivity audit ===")
    print(f"{len(idx)} images, {n_groups} groups, {len(classes)} classes")

    # Step 1: the grouping variable must actually separate the data.
    if n_groups < a.n_splits:
        raise SystemExit(f"only {n_groups} groups for {a.n_splits} folds; "
                         f"lower --n-splits or supply a finer grouping")
    per_group = idx.groupby("patient_id").label7.nunique()
    print(f"groups contributing a single class: {(per_group == 1).sum()}/{n_groups}"
          "  (the higher this is, the more the group identifies the label)")

    # Step 2: encode once.
    normalised = a.out.parent / (a.out.name + "_index.csv")
    normalised.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(normalised, index=False)
    store_dir = a.featroot / f"{a.backbone}__raw"
    if not a.skip_encode or not (store_dir / "tiles.npy").exists():
        from features import extract_embeddings as E
        E.run(normalised, a.backbone, "raw", store_dir, keep_shards=True)

    from experiments.run_benchmark import FeatureStore, set_featroot
    set_featroot(a.featroot)
    store = FeatureStore(a.backbone, "raw", idx)

    # Steps 3 and 4: matched arms, and disjointness verified rather than assumed.
    task = P.Task("audit", "label7", tuple(classes))
    rows, preds = [], []
    X = store.representation("tilemean")
    y = idx.label7.values
    g = idx.patient_id.values
    for protocol, groups in (("grouped", g), ("ungrouped", None)):
        for seed in a.seeds:
            for k, (tr, te) in enumerate(P.make_folds(y, groups, a.n_splits, seed)):
                model = LinearHead(seed=seed).fit(X[tr], y[tr])
                proba = model.predict_proba(X[te])
                cls = list(model.classes_)
                pred = np.array(cls)[proba.argmax(1)]
                pm = np.zeros((len(pred), len(classes)))
                for j, c in enumerate(classes):
                    if c in cls:
                        pm[:, j] = proba[:, cls.index(c)]
                m = P.evaluate(y[te], pred, pm, tuple(classes))
                rows.append(dict(protocol=protocol, seed=seed, fold=k,
                                 n_train=len(tr), n_test=len(te),
                                 group_overlap=P.check_disjoint(g[tr], g[te]), **m))
                for n, i in enumerate(te):
                    preds.append(dict(protocol=protocol, seed=seed, fold=k,
                                      patient_id=int(g[i]), y_true=y[i],
                                      y_pred=pred[n]))
    folds = pd.DataFrame(rows)
    pred_df = pd.DataFrame(preds)
    folds.to_csv(f"{a.out}_folds.csv", index=False)

    # Step 5: the contrast, and an interval over groups.
    piv = folds.pivot_table(index=["seed", "fold"], columns="protocol",
                            values="balanced_accuracy")
    delta = (piv["ungrouped"] - piv["grouped"]).dropna()
    from scipy.stats import wilcoxon
    stat = wilcoxon(delta) if len(delta) > 5 else None
    point, lo, hi = _group_bootstrap(
        pred_df[pred_df.protocol == "grouped"], classes)

    overlap = int(folds[folds.protocol == "grouped"].group_overlap.max())
    summary = pd.DataFrame([{
        "grouped_mean": piv["grouped"].mean(),
        "grouped_group_ci_lo": lo, "grouped_group_ci_hi": hi,
        "ungrouped_mean": piv["ungrouped"].mean(),
        "inflation_pp": 100 * delta.mean(),
        "wilcoxon_p": getattr(stat, "pvalue", float("nan")),
        "max_group_overlap_in_grouped_folds": overlap,
        "n_groups": n_groups, "n_images": len(idx),
    }])
    summary.to_csv(f"{a.out}_summary.csv", index=False)

    print(f"\ngrouped   balanced accuracy {piv['grouped'].mean():.3f}"
          f"  [{lo:.3f}, {hi:.3f}]  (interval over groups)")
    print(f"ungrouped balanced accuracy {piv['ungrouped'].mean():.3f}")
    print(f"INFLATION {100 * delta.mean():+.1f} percentage points"
          + (f"  (Wilcoxon p = {stat.pvalue:.2e})" if stat is not None else ""))
    print(f"groups shared across a grouped fold: {overlap}  (must be 0)")
    if overlap:
        print("WARNING: the grouped protocol did not separate the groups.")
    print(f"\nwrote {a.out}_folds.csv and {a.out}_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
