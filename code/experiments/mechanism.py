"""When does the partition matter? The mechanism behind the inflation.

Two measurements tell a reader how large the effect is on two cohorts. They do
not tell the reader what to expect on a third. This experiment isolates the
mechanism so that the size of the effect becomes predictable from two
properties a practitioner can compute before running anything:

  * how many images each group contributes. With one image per group, a grouped
    and an ungrouped split are the same partition, so the inflation must vanish;
    it should grow as groups get larger.
  * how similar the test images are to the training images in feature space.
    This is measurable *without* the grouping variable, which matters because
    the datasets where leakage does the most damage are exactly the ones that
    do not ship it.

Everything runs on cached embeddings: no encoder pass, no training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import protocols as P  # noqa: E402
from experiments.run_benchmark import FeatureStore, set_featroot  # noqa: E402
from models.heads import LinearHead, l2  # noqa: E402


def _subsample(idx: pd.DataFrame, cap: int, seed: int) -> np.ndarray:
    """At most ``cap`` images per group, drawn once per (cap, seed)."""
    rng = np.random.default_rng(1000 * seed + cap)
    keep = []
    for _, rows in idx.groupby("patient_id").groups.items():
        r = np.asarray(rows)
        keep.append(r if len(r) <= cap else rng.choice(r, cap, replace=False))
    return np.sort(np.concatenate(keep))


def _leak_score(Xtr: np.ndarray, Xte: np.ndarray,
                gtr: np.ndarray | None = None,
                gte: np.ndarray | None = None) -> tuple[float, float, float]:
    """How much closer is a test image's nearest training neighbour than chance?

    Raw cosine similarities between deep features are all high and therefore
    uninformative on their own, so the nearest-neighbour similarity is
    standardised against the distribution of all test-train similarities. The
    result answers a question a practitioner can ask *without* the grouping
    variable: does each test image have an unusually close partner on the
    training side? Returns (mean nearest similarity, standardised score,
    fraction of nearest neighbours drawn from the test image's own group).
    """
    A, B = l2(Xte), l2(Xtr)
    best = np.full(len(A), -1.0)
    arg = np.zeros(len(A), dtype=int)
    tot = cnt = 0.0
    sq = 0.0
    for i in range(0, len(A), 256):           # blocked, to bound memory
        sim = A[i:i + 256] @ B.T
        best[i:i + 256] = sim.max(axis=1)
        arg[i:i + 256] = sim.argmax(axis=1)
        tot += sim.sum(); sq += (sim ** 2).sum(); cnt += sim.size
    mu = tot / cnt
    sd = float(np.sqrt(max(sq / cnt - mu ** 2, 1e-12)))
    same = float("nan")
    if gtr is not None and gte is not None:
        same = float((gtr[arg] == gte).mean())
    return float(best.mean()), float(((best - mu) / sd).mean()), same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--featroot", type=Path, default=Path("data/features"))
    ap.add_argument("--backbone", default="resnet50_in1k")
    ap.add_argument("--label-col", default="superclass")
    ap.add_argument("--caps", type=int, nargs="+",
                    default=[1, 2, 3, 4, 6, 8, 12, 10**6])
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
    for cap in a.caps:
        for seed in a.seeds:
            sub = _subsample(idx, cap, seed)
            X, y, g = X_all[sub], y_all[sub], g_all[sub]
            if len(np.unique(y)) < 2:
                continue
            per_group = len(sub) / len(np.unique(g))
            for protocol, groups in (("patient", g), ("image", None)):
                try:
                    folds = P.make_folds(y, groups, a.n_splits, seed)
                except ValueError:
                    continue
                for k, (tr, te) in enumerate(folds):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    model = LinearHead(seed=seed).fit(X[tr], y[tr])
                    pred = model.classes_[model.predict_proba(X[te]).argmax(1)]
                    from sklearn.metrics import balanced_accuracy_score
                    lm, lz, same = _leak_score(X[tr], X[te], g[tr], g[te])
                    rows.append(dict(
                        cohort=a.cohort, cap=cap, seed=seed, fold=k,
                        protocol=protocol,
                        balanced_accuracy=balanced_accuracy_score(y[te], pred),
                        leak_sim=lm, leak_z=lz, nn_same_group=same,
                        n_images=len(sub), n_groups=int(len(np.unique(g))),
                        images_per_group=per_group,
                        group_overlap=P.check_disjoint(g[tr], g[te])))
        done = [r for r in rows if r["cap"] == cap]
        if done:
            pat = np.mean([r["balanced_accuracy"] for r in done
                           if r["protocol"] == "patient"])
            img = np.mean([r["balanced_accuracy"] for r in done
                           if r["protocol"] == "image"])
            print(f"cap {cap:>7}  images/group {done[0]['images_per_group']:5.1f}"
                  f"  grouped {pat:.3f}  ungrouped {img:.3f}"
                  f"  inflation {100 * (img - pat):+5.1f} pp", flush=True)

    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
