"""Cross-validated benchmark driver.

One invocation sweeps a grid of (task x protocol x representation x head x
magnification) under repeated class-stratified cross-validation and writes two
long-format tables: per-fold scores, and per-sample out-of-fold predictions for
the calibration and error analyses.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import protocols as P  # noqa: E402
from models import heads as H  # noqa: E402

FEATROOT = Path("data/features")


def set_featroot(path: Path) -> None:
    """Point the feature store elsewhere, so a second cohort's embeddings do
    not collide with the first's under the same (backbone, norm) name."""
    global FEATROOT
    FEATROOT = Path(path)


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------
class FeatureStore:
    """Cached tile/global embeddings for one (backbone, normalisation) pair."""

    def __init__(self, backbone: str, norm: str, index: pd.DataFrame):
        d = FEATROOT / f"{backbone}__{norm}"
        self.tiles = np.load(d / "tiles.npy").astype(np.float32)
        self.tmeta = pd.read_csv(d / "tiles.csv")
        self.glob = np.load(d / "global.npy").astype(np.float32)
        self.hand = pd.read_csv(d / "handcrafted.csv")
        self.index = index.reset_index(drop=True)
        self.backbone, self.norm = backbone, norm
        # row ranges of each image's tiles; tiles.csv is written in image order
        self._slices: dict[int, tuple[int, int]] = {}
        ii = self.tmeta.image_idx.values
        bounds = np.flatnonzero(np.diff(ii, prepend=ii[0] - 1))
        for b, e in zip(bounds, list(bounds[1:]) + [len(ii)]):
            self._slices[int(ii[b])] = (int(b), int(e))
        self._mean: np.ndarray | None = None

    @property
    def tile_mean(self) -> np.ndarray:
        """L2-normalised mean of an image's tissue tiles."""
        if self._mean is None:
            out = np.zeros((len(self.index), self.tiles.shape[1]), dtype=np.float32)
            for i in range(len(self.index)):
                b, e = self._slices[i]
                out[i] = self.tiles[b:e].mean(axis=0)
            self._mean = H.l2(out)
        return self._mean

    def bags(self, rows: np.ndarray) -> list[np.ndarray]:
        return [H.l2(self.tiles[slice(*self._slices[int(i)])]) for i in rows]

    def representation(self, kind: str) -> np.ndarray:
        if kind == "tilemean":
            return self.tile_mean
        if kind == "global":
            return H.l2(self.glob)
        if kind == "tilemean+global":
            return np.hstack([self.tile_mean, H.l2(self.glob)])
        if kind == "handcrafted":
            # Deliberately trivial control: tissue coverage and stain colour
            # statistics only, no learned features whatsoever.
            cols = [c for c in self.hand.columns if c != "image_idx"]
            v = self.hand.sort_values("image_idx")[cols].values.astype(np.float32)
            return v
        raise KeyError(kind)


# --------------------------------------------------------------------------
# One cross-validated configuration
# --------------------------------------------------------------------------
def _subset_rows(index: pd.DataFrame, task: P.Task, mag: str) -> np.ndarray:
    d = index
    if task.subset:
        d = d.query(task.subset)
    if mag in ("20x", "40x"):
        d = d[d.magnification == mag]
    return d.index.values


def run_config(store: FeatureStore, task_name: str, protocol: str, rep: str,
               head_name: str, mag: str, n_splits: int, seeds: list[int],
               head_kw: dict | None = None, tile_max_per_image: int = 0):
    task = P.TASKS[task_name]
    proto = P.PROTOCOLS[protocol]
    index = store.index
    head_kw = dict(head_kw or {})

    rows = _subset_rows(index, task, mag)
    if protocol == "tile":
        # Evaluation unit is the tile: tiles of one image are allowed to
        # straddle the split, reproducing patch-level leakage.
        #
        # All 22,404 tiles make each fit roughly 32x larger than the
        # image-level arm and one configuration took over two hours; since the
        # arm exists to demonstrate a leakage mechanism rather than to reach a
        # ceiling, it is evaluated on a fixed per-image subsample. Keeping a
        # fixed number of tiles *per image* leaves every image represented and
        # leaves the leakage structure untouched -- tiles of one image still
        # straddle the split -- while making the comparison affordable.
        tm = store.tmeta
        sel = tm.image_idx.isin(rows).values
        if tile_max_per_image:
            keep = np.zeros(len(tm), bool)
            rs = np.random.default_rng(0)
            for img, idx in tm.index.to_series().groupby(tm.image_idx):
                idx = idx.values[sel[idx.values]]
                if len(idx) > tile_max_per_image:
                    idx = rs.choice(idx, tile_max_per_image, replace=False)
                keep[idx] = True
            sel = sel & keep
        X = H.l2(store.tiles[sel])
        meta = tm[sel].reset_index(drop=True)
        y = meta[task.label_col].fillna("").values.astype(str)
        groups = None
        unit = "tile"
        img_of_unit = meta.image_idx.values
        pat_of_unit = meta.patient_id.values
    else:
        X = store.representation(rep)[rows] if head_name != "attnmil" else None
        sub = index.loc[rows]
        y = sub[task.label_col].fillna("").values.astype(str)
        groups = sub.patient_id.values if proto.group_col else None
        unit = "image"
        img_of_unit = rows
        pat_of_unit = sub.patient_id.values

    keep = np.isin(y, task.classes)
    if not keep.all():
        y, img_of_unit, pat_of_unit = y[keep], img_of_unit[keep], pat_of_unit[keep]
        if X is not None:
            X = X[keep]
        if groups is not None:
            groups = groups[keep]
        rows = rows[keep] if unit == "image" else rows

    fold_rows, pred_rows = [], []
    for seed in seeds:
        folds = P.make_folds(y, groups, n_splits, seed)
        for k, (tr, te) in enumerate(folds):
            if len(np.unique(y[tr])) < 2:
                continue
            Head = H.HEADS[head_name]
            if head_name == "attnmil":
                model = Head(seed=seed, **head_kw)
                bag_rows = rows if unit == "image" else img_of_unit
                model.fit(store.bags(bag_rows[tr]), y[tr])
                proba = model.predict_proba(store.bags(bag_rows[te]))
            else:
                model = Head(seed=seed, **head_kw).fit(X[tr], y[tr])
                proba = model.predict_proba(X[te])
            classes = list(model.classes_)
            pred = np.array(classes)[proba.argmax(1)]

            # align probability columns to the task's canonical class order
            pm = np.zeros((len(pred), len(task.classes)))
            for j, c in enumerate(task.classes):
                if c in classes:
                    pm[:, j] = proba[:, classes.index(c)]

            m = P.evaluate(y[te], pred, pm, task.classes, ordinal=task.ordinal)
            if task_name == "full7":
                m.update(P.graded_confusion_summary(y[te], pred))
            fold_rows.append(dict(
                backbone=store.backbone, norm=store.norm, task=task_name,
                protocol=protocol, representation=rep if head_name != "attnmil" else "tiles",
                head=head_name, magnification=mag, unit=unit, seed=seed, fold=k,
                n_train=len(tr), n_test=len(te),
                n_train_patients=len(set(pat_of_unit[tr].tolist())),
                n_test_patients=len(set(pat_of_unit[te].tolist())),
                patient_overlap=P.check_disjoint(pat_of_unit[tr], pat_of_unit[te]),
                image_overlap=len(set(img_of_unit[tr].tolist()) & set(img_of_unit[te].tolist())),
                **m))
            for n, i in enumerate(te):
                pred_rows.append(dict(
                    task=task_name, protocol=protocol, backbone=store.backbone,
                    norm=store.norm, representation=rep, head=head_name,
                    magnification=mag, seed=seed, fold=k, unit=unit,
                    image_idx=int(img_of_unit[i]), patient_id=int(pat_of_unit[i]),
                    y_true=y[i], y_pred=pred[n],
                    p_max=float(pm[n].max()),
                    **{f"p_{c}": float(pm[n, j]) for j, c in enumerate(task.classes)}))
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("results/tables/image_index.csv"))
    ap.add_argument("--grid", type=Path, required=True, help="JSON list of config dicts")
    ap.add_argument("--out-prefix", type=Path, required=True)
    # k=8 is the bound the dataset audit derives: the scarcest of the seven
    # classes spans 8 distinct patients, so a patient-disjoint,
    # class-stratified design cannot exceed that. Defaulting to it keeps a
    # direct invocation consistent with the manuscript.
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--tile-max-per-image", type=int, default=8,
                    help="tiles kept per image under the tile protocol; 0 = all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--featroot", type=Path, default=FEATROOT,
                    help="root of the cached embedding store")
    a = ap.parse_args()
    set_featroot(a.featroot)

    index = pd.read_csv(a.index)
    grid = json.loads(a.grid.read_text())
    stores: dict[tuple[str, str], FeatureStore] = {}
    all_f, all_p = [], []
    t0 = time.time()

    for gi, cfg in enumerate(grid):
        key = (cfg["backbone"], cfg.get("norm", "raw"))
        if key not in stores:
            stores[key] = FeatureStore(*key, index)
        store = stores[key]
        seeds = cfg.get("seeds", a.seeds)
        combos = list(itertools.product(
            cfg["tasks"], cfg["protocols"], cfg["representations"],
            cfg["heads"], cfg["magnifications"]))
        for task, proto, rep, head, mag in combos:
            if head == "attnmil" and proto == "tile":
                continue  # a bag head is meaningless when the unit is a tile
            if proto == "tile" and rep != "tilemean":
                continue  # tile protocol ignores the image-level representation
            f, p = run_config(store, task, proto, rep, head, mag,
                              a.n_splits, seeds, cfg.get("head_kw"),
                              tile_max_per_image=a.tile_max_per_image)
            if len(f):
                all_f.append(f)
                all_p.append(p)
                # Write after every configuration, not only at the end. One
                # attention-MIL arm ran for eleven hours; losing a whole stage
                # because the last configuration was interrupted is not an
                # acceptable failure mode on hardware that needs babysitting.
                pd.concat(all_f, ignore_index=True).to_csv(
                    f"{a.out_prefix}_folds.csv", index=False)
                pd.concat(all_p, ignore_index=True).to_csv(
                    f"{a.out_prefix}_preds.csv", index=False)
                ba = f.balanced_accuracy
                print(f"[{gi+1}/{len(grid)}] {key[0]}/{key[1]} {task:7s} {proto:7s} "
                      f"{rep:16s} {head:8s} {mag:4s}  bal_acc={ba.mean():.3f}"
                      f"+-{ba.std():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    folds = pd.concat(all_f, ignore_index=True)
    folds.to_csv(f"{a.out_prefix}_folds.csv", index=False)
    pd.concat(all_p, ignore_index=True).to_csv(f"{a.out_prefix}_preds.csv", index=False)
    print(f"\n{len(folds)} fold-records in {time.time()-t0:.0f}s -> {a.out_prefix}_folds.csv")


if __name__ == "__main__":
    main()
