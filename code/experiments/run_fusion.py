"""E5: how far the patient-disjoint result can be pushed from cached features.

Everything here reuses the embedding store, so the stage costs minutes rather
than the hours an encoding pass costs. Four levers are available once the
embeddings exist, and they are applied cumulatively so that the gain can be
attributed to each rather than only reported in total:

  1. regularisation tuned by nested cross-validation inside each training fold;
  2. pooling over an image's tiles beyond the plain mean;
  3. fusion of encoders by concatenating their per-image representations;
  4. ensembling encoders by averaging their predicted probabilities.

Fusion is not novel on this dataset -- Vailankanni et al. fuse self-supervised
objectives and Sarita et al. fuse supervised backbones -- so the contribution
is not the mechanism but the fact that it is measured under a patient-disjoint
protocol and extended to the seven-class task. Every choice is made inside the
training folds; nothing here sees a test patient.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import protocols as P  # noqa: E402
from experiments.run_benchmark import FeatureStore, _subset_rows  # noqa: E402
from models import heads as H  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# There are 691 images and up to 8192 concatenated dimensions, so the head is
# fitted in a PCA basis estimated inside the training fold. This is both better
# conditioned than the raw space at n << d and the only way the concatenated
# arms finish in reasonable time; fitting the basis on training rows only keeps
# the test patients unseen.
PCA_DIM = 256   # overridden by --pca-dim; 0 disables the projection


def _project(Xtr: np.ndarray, Xte: np.ndarray, seed: int):
    if not PCA_DIM:
        return Xtr, Xte
    k = min(PCA_DIM, Xtr.shape[1], max(2, Xtr.shape[0] - 1))
    if k >= Xtr.shape[1]:
        return Xtr, Xte
    pca = PCA(n_components=k, random_state=seed).fit(Xtr)
    return pca.transform(Xtr), pca.transform(Xte)


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------
def pool(tiles: np.ndarray, kind: str) -> np.ndarray:
    """Summarise one image's L2-normalised tile embeddings into one vector."""
    if kind == "mean":
        return tiles.mean(axis=0)
    if kind == "meanstd":
        # Cheap second-order summary: dispersion across tiles separates a
        # uniformly patterned field from a heterogeneous one, which is part of
        # what differentiation grade describes.
        return np.concatenate([tiles.mean(axis=0), tiles.std(axis=0)])
    if kind == "gem":
        # Generalised mean, p=3: weights the strongest responses without
        # discarding the rest, unlike a hard top-k.
        t = np.clip(tiles, 1e-6, None)
        return (t ** 3).mean(axis=0) ** (1.0 / 3.0)
    if kind == "topk":
        # Mean of the quarter of tiles with the largest norm, which is a crude
        # proxy for the most tissue-dense fields.
        k = max(1, int(round(0.25 * len(tiles))))
        idx = np.argsort(np.linalg.norm(tiles, axis=1))[-k:]
        return tiles[idx].mean(axis=0)
    raise KeyError(kind)


def pooled_matrix(store: FeatureStore, kind: str) -> np.ndarray:
    n = len(store.index)
    first = pool(H.l2(store.tiles[slice(*store._slices[0])]), kind)
    out = np.zeros((n, first.shape[0]), dtype=np.float32)
    out[0] = first
    for i in range(1, n):
        out[i] = pool(H.l2(store.tiles[slice(*store._slices[i])]), kind)
    return H.l2(out)


# --------------------------------------------------------------------------
# Head with regularisation chosen inside the training fold
# --------------------------------------------------------------------------
def fit_tuned(Xtr, ytr, gtr, seed: int, tune: bool):
    """Linear head; when ``tune`` is set, C is chosen by inner patient-grouped CV."""
    if not tune:
        return H.LinearHead(C=1.0, seed=seed).fit(Xtr, ytr), 1.0

    n_in = min(3, len(np.unique(gtr)), int(pd.Series(ytr).value_counts().min()))
    if n_in < 2:
        return H.LinearHead(C=1.0, seed=seed).fit(Xtr, ytr), 1.0

    inner = StratifiedGroupKFold(n_splits=n_in, shuffle=True, random_state=seed)
    best_c, best_s = 1.0, -np.inf
    for c in C_GRID:
        scores = []
        for a, b in inner.split(np.zeros(len(ytr)), ytr, groups=gtr):
            if len(np.unique(ytr[a])) < 2:
                continue
            m = H.LinearHead(C=c, seed=seed).fit(Xtr[a], ytr[a])
            pr = np.array(m.classes_)[m.predict_proba(Xtr[b]).argmax(1)]
            from sklearn.metrics import balanced_accuracy_score
            scores.append(balanced_accuracy_score(ytr[b], pr))
        if scores and np.mean(scores) > best_s:
            best_s, best_c = float(np.mean(scores)), c
    return H.LinearHead(C=best_c, seed=seed).fit(Xtr, ytr), best_c


def _aligned(proba, classes, task_classes):
    pm = np.zeros((proba.shape[0], len(task_classes)))
    for j, c in enumerate(task_classes):
        if c in classes:
            pm[:, j] = proba[:, classes.index(c)]
    return pm


# --------------------------------------------------------------------------
# One cumulative-ablation arm
# --------------------------------------------------------------------------
def run_arm(stores: dict[str, FeatureStore], feats: dict[str, np.ndarray],
            index: pd.DataFrame, task_name: str, arm: str, backbones: list[str],
            pooling: str, tune: bool, mode: str, n_splits: int,
            seeds: list[int]) -> tuple[list[dict], list[dict]]:
    task = P.TASKS[task_name]
    rows = _subset_rows(index, task, "both")
    sub = index.loc[rows]
    y = sub[task.label_col].fillna("").values.astype(str)
    groups = sub.patient_id.values
    keep = np.isin(y, task.classes)
    rows, y, groups = rows[keep], y[keep], groups[keep]

    mats = [feats[f"{b}|{pooling}"][rows] for b in backbones]
    Xcat = np.hstack(mats) if len(mats) > 1 else mats[0]

    fold_rows, pred_rows = [], []
    for seed in seeds:
        for k, (tr, te) in enumerate(P.make_folds(y, groups, n_splits, seed)):
            if len(np.unique(y[tr])) < 2:
                continue
            if mode == "concat":
                Xa, Xb = _project(Xcat[tr], Xcat[te], seed)
                model, c = fit_tuned(Xa, y[tr], groups[tr], seed, tune)
                pm = _aligned(model.predict_proba(Xb), list(model.classes_),
                              task.classes)
            else:  # average each encoder's probabilities
                acc, c = [], 1.0
                for M in mats:
                    Xa, Xb = _project(M[tr], M[te], seed)
                    model, c = fit_tuned(Xa, y[tr], groups[tr], seed, tune)
                    acc.append(_aligned(model.predict_proba(Xb),
                                        list(model.classes_), task.classes))
                pm = np.mean(acc, axis=0)
            pred = np.array(task.classes)[pm.argmax(1)]

            m = P.evaluate(y[te], pred, pm, task.classes, ordinal=task.ordinal)
            if task_name == "full7":
                m.update(P.graded_confusion_summary(y[te], pred))
            fold_rows.append(dict(
                arm=arm, backbone="+".join(backbones), norm="raw", task=task_name,
                protocol="patient", representation=f"{pooling}:{mode}",
                head="linear-tuned" if tune else "linear", magnification="both",
                unit="image", seed=seed, fold=k, C=c,
                n_train=len(tr), n_test=len(te),
                n_train_patients=len(set(groups[tr].tolist())),
                n_test_patients=len(set(groups[te].tolist())),
                patient_overlap=P.check_disjoint(groups[tr], groups[te]),
                image_overlap=0, **m))
            for n, i in enumerate(te):
                pred_rows.append(dict(
                    arm=arm, task=task_name, protocol="patient", seed=seed, fold=k,
                    image_idx=int(rows[i]), patient_id=int(groups[i]),
                    y_true=y[i], y_pred=pred[n], p_max=float(pm[n].max()),
                    **{f"p_{c2}": float(pm[n, j]) for j, c2 in enumerate(task.classes)}))
    return fold_rows, pred_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("results/tables/image_index.csv"))
    ap.add_argument("--out-prefix", type=Path, default=Path("results/tables/e5_fusion"))
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50_in1k", "phikon_vitb16",
                             "efficientnet_b0"])
    ap.add_argument("--tasks", nargs="+", default=["super3", "full7", "grade3"])
    ap.add_argument("--n-splits", type=int, default=8)
    # The projection was added purely to make the concatenated arms affordable.
    # It costs accuracy (ResNet-50 tile-mean drops from 0.722 to 0.698), so it
    # must be possible to switch off when the comparison is what matters.
    ap.add_argument("--pca-dim", type=int, default=256,
                    help="PCA components fitted inside each training fold; 0 = none")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    global PCA_DIM
    PCA_DIM = a.pca_dim
    index = pd.read_csv(a.index)
    have = [b for b in a.backbones if (Path("data/features") / f"{b}__raw" / "tiles.npy").exists()]
    missing = [b for b in a.backbones if b not in have]
    if missing:
        print(f"[e5] not yet encoded, skipping: {', '.join(missing)}", flush=True)
    if not have:
        print("[e5] no encoded backbones available; nothing to do", flush=True)
        return
    print(f"[e5] using {', '.join(have)}", flush=True)

    stores = {b: FeatureStore(b, "raw", index) for b in have}
    poolings = ["mean", "meanstd", "gem", "topk"]
    feats: dict[str, np.ndarray] = {}
    for b, st in stores.items():
        for pk in poolings:
            t0 = time.time()
            feats[f"{b}|{pk}"] = pooled_matrix(st, pk)
            print(f"  pooled {b}/{pk} in {time.time()-t0:.0f}s", flush=True)

    # Baseline is the single best-known encoder with an untuned linear head;
    # each later arm adds exactly one lever so the ablation is readable.
    base = "resnet50_in1k" if "resnet50_in1k" in have else have[0]
    arms = [("baseline: single encoder, mean pooling, C=1", [base], "mean", False, "concat"),
            ("+ tuned regularisation", [base], "mean", True, "concat")]
    for pk in poolings:
        if pk != "mean":
            arms.append((f"+ {pk} pooling", [base], pk, True, "concat"))
    if len(have) > 1:
        arms.append(("+ encoder fusion (concatenate)", have, "mean", True, "concat"))
        arms.append(("+ encoder ensemble (average)", have, "mean", True, "average"))

    folds, preds = [], []
    for task in a.tasks:
        for arm, bks, pk, tune, mode in arms:
            t0 = time.time()
            f, p = run_arm(stores, feats, index, task, arm, bks, pk, tune, mode,
                           a.n_splits, a.seeds)
            folds += f
            preds += p
            if f:
                bacc = np.mean([r["balanced_accuracy"] for r in f])
                print(f"  [{task}] {arm:46s} bacc={bacc:.3f}  ({time.time()-t0:.0f}s)",
                      flush=True)

    a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(f"{a.out_prefix}_folds.csv", index=False)
    pd.DataFrame(preds).to_csv(f"{a.out_prefix}_preds.csv", index=False)
    print(f"[e5] wrote {a.out_prefix}_folds.csv ({len(folds)} fold rows)")


if __name__ == "__main__":
    main()
