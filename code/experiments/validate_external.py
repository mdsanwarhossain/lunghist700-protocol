"""Train on LungHist700, test on LC25000 lung - twice.

The same fitted classifier is scored on (a) a sample of LC25000 lung images
drawn the way the literature draws it, augmented duplicates included, and (b)
one representative per recovered augmentation group. Comparing the two isolates
how much of a conventional LC25000 result is independent evidence and how much
is the same 750 fields counted twenty times over.
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


def cluster_bootstrap_ci(correct: np.ndarray, groups: np.ndarray,
                         n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap that resamples whole augmentation groups.

    Resampling individual images treats twenty rotations of one field as twenty
    observations and so understates the interval; resampling groups does not.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_group = {g: correct[groups == g] for g in uniq}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        draws[b] = np.concatenate([by_group[g] for g in pick]).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("results/tables/image_index.csv"))
    ap.add_argument("--backbone", default="resnet50_in1k")
    ap.add_argument("--cache", type=Path, default=Path("data/features/lc25000"))
    ap.add_argument("--outdir", type=Path, default=Path("results/tables"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    index = pd.read_csv(a.index)
    store = FeatureStore(a.backbone, "raw", index)
    task = P.TASKS["super3"]

    X_src = store.representation("tilemean")
    y_src = index.superclass.values.astype(str)

    meta = pd.read_csv(a.cache / "lc_meta.csv")
    rows = pd.read_csv(a.cache / f"lc_embedded_rows_{a.backbone}.csv").row.values
    X_ext = H.l2(np.load(a.cache / f"lc_tilemean_{a.backbone}.npy").astype(np.float32))
    ext = meta.set_index("row").loc[rows].reset_index()
    y_ext = ext.superclass.values.astype(str)
    gid = ext.group_id.values if "group_id" in ext.columns else np.arange(len(ext))

    # One representative per group: the deduplicated, independent test set.
    dedup = ext.groupby("group_id", sort=False).head(1).index.values

    recs = []
    for seed in a.seeds:
        model = H.LinearHead(seed=seed).fit(X_src, y_src)
        classes = list(model.classes_)
        proba = model.predict_proba(X_ext)
        pred = np.array(classes)[proba.argmax(1)]
        pm = np.zeros((len(pred), len(task.classes)))
        for j, c in enumerate(task.classes):
            if c in classes:
                pm[:, j] = proba[:, classes.index(c)]

        for name, sel in (("augmented_sample", np.arange(len(ext))),
                          ("deduplicated_groups", dedup)):
            m = P.evaluate(y_ext[sel], pred[sel], pm[sel], task.classes)
            correct = (y_ext[sel] == pred[sel]).astype(float)
            naive_lo = correct.mean() - 1.96 * correct.std(ddof=1) / np.sqrt(len(correct))
            naive_hi = correct.mean() + 1.96 * correct.std(ddof=1) / np.sqrt(len(correct))
            clo, chi = cluster_bootstrap_ci(correct, gid[sel], seed=seed)
            recs.append({
                "seed": seed, "test_set": name, "n_images": len(sel),
                "n_independent_groups": int(len(np.unique(gid[sel]))),
                **m,
                "naive_ci_lo": naive_lo, "naive_ci_hi": naive_hi,
                "naive_ci_width": naive_hi - naive_lo,
                "cluster_ci_lo": clo, "cluster_ci_hi": chi,
                "cluster_ci_width": chi - clo,
            })

    df = pd.DataFrame(recs)
    a.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.outdir / f"external_lc25000_{a.backbone}.csv", index=False)
    cols = ["n_images", "n_independent_groups", "balanced_accuracy", "macro_f1",
            "naive_ci_width", "cluster_ci_width"]
    print("\n-- LungHist700 -> LC25000 lung, zero-shot transfer --")
    print(df.groupby("test_set")[cols].mean().round(4).to_string())
    print(f"\nwritten -> {a.outdir}/external_lc25000_{a.backbone}.csv")


if __name__ == "__main__":
    main()
