"""External validation on LC25000, with its lost augmentation traceability
reconstructed first.

LC25000 expands 250 original fields per class into 5,000 by rotation and
flipping and ships no mapping from an augmented image back to its source. The
LungHist700 descriptor names exactly this as the reason conclusions drawn on
LC25000 are unsafe, yet no published work repairs it.

Rotations and flips permute pixel positions and leave the pixel *multiset*
untouched, so a per-channel intensity histogram is an exact invariant of the
augmentation group. Grouping on that fingerprint recovers the original fields,
which both lets a model be scored on independent images and shows how much of a
reported LC25000 interval is replication rather than evidence.

Stage 1 (``fingerprint``) decodes every lung image but only histograms it, which
is cheap. Stage 2 (``embed``) encodes a chosen subset with the frozen backbone.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import backbones  # noqa: E402

# LC25000 ships as HuggingFace ClassLabel integers, with the string names
# carried in the parquet schema metadata rather than in the columns. The three
# lung diagnoses are mapped onto LungHist700's superclass vocabulary so a
# classifier trained there can be scored here without relabelling.
LC_SUPERCLASS = {"benign": "nor", "adenocarcinomas": "aca",
                 "squamous carcinomas": "scc"}
HIST_BINS = 48
TILE = 224


def class_names(parquet_dir: Path) -> dict[str, list[str]]:
    """Read the ClassLabel name lists for ``organ`` and ``label``."""
    import json
    f = sorted(parquet_dir.glob("*.parquet"))[0]
    md = pq.ParquetFile(f).schema_arrow.metadata[b"huggingface"]
    feats = json.loads(md.decode())["info"]["features"]
    return {k: feats[k]["names"] for k in ("organ", "label")}


def histogram_fingerprint(rgb: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Per-channel intensity histogram: invariant to rotation and flipping, so
    near-identical across one augmentation group."""
    return np.concatenate([
        np.histogram(rgb[..., c], bins=bins, range=(0, 256))[0] for c in range(3)
    ]).astype(np.float32) / rgb[..., 0].size


def _iter_lung(parquet_dir: Path):
    """Yield ``(row_index, label_name, jpeg_bytes)`` for every lung row, in file
    order. Colon rows are skipped; the index counts lung rows only, so it is
    stable across the fingerprint and embedding passes."""
    names = class_names(parquet_dir)
    organ_names, label_names = names["organ"], names["label"]
    n = 0
    for f in sorted(parquet_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            t = batch.to_pydict()
            for img, organ, label in zip(t["image"], t["organ"], t["label"]):
                org = organ if isinstance(organ, str) else organ_names[organ]
                if org != "lung":
                    continue
                lab = label if isinstance(label, str) else label_names[label]
                if lab not in LC_SUPERCLASS:
                    continue
                yield n, lab, (img["bytes"] if isinstance(img, dict) else img)
                n += 1


def stage_fingerprint(parquet_dir: Path, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    fps, recs = [], []
    for i, lab, raw in _iter_lung(parquet_dir):
        rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        fps.append(histogram_fingerprint(rgb))
        recs.append({"row": i, "lc_label": lab, "superclass": LC_SUPERCLASS[lab],
                     "height": rgb.shape[0], "width": rgb.shape[1]})
        if (i + 1) % 2500 == 0:
            print(f"  fingerprinted {i+1} lung images", flush=True)
    meta = pd.DataFrame(recs)
    np.save(cache / "lc_fingerprints.npy", np.stack(fps))
    meta.to_csv(cache / "lc_meta.csv", index=False)
    print(f"fingerprinted {len(meta)} lung images -> {cache}")
    return meta


def recover_groups(fingerprints: np.ndarray, labels: np.ndarray,
                   tol: float = 2e-3) -> np.ndarray:
    """Single-linkage grouping on L1 fingerprint distance, within each class."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    gid = np.full(len(fingerprints), -1, dtype=int)
    nxt = 0
    for lab in np.unique(labels):
        m = np.flatnonzero(labels == lab)
        if len(m) == 1:
            gid[m] = nxt
            nxt += 1
            continue
        cl = fcluster(linkage(pdist(fingerprints[m], metric="cityblock"),
                              method="single"), t=tol, criterion="distance")
        for c in np.unique(cl):
            gid[m[cl == c]] = nxt
            nxt += 1
    return gid


def group_diagnostics(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta.groupby(["lc_label", "group_id"]).size().rename("size").reset_index()
    rows = []
    for lab, gg in g.groupby("lc_label"):
        rows.append({"lc_label": lab, "n_images": int(gg["size"].sum()),
                     "n_groups_recovered": len(gg),
                     "n_originals_documented": 250,
                     "median_group_size": float(gg["size"].median()),
                     "min_group_size": int(gg["size"].min()),
                     "max_group_size": int(gg["size"].max()),
                     "groups_of_size_20": int((gg["size"] == 20).sum())})
    return pd.DataFrame(rows)


@torch.inference_mode()
def stage_embed(parquet_dir: Path, cache: Path, backbone: str,
                rows: set[int] | None, threads: int | None = None) -> None:
    """Encode selected rows with a 224-px tile grid, mean-pooled per image, so
    the representation matches the one used on LungHist700."""
    # The thermal supervisor communicates its thread budget through the
    # environment; a hardcoded value here would silently override it and run
    # the machine hotter than the governor intends.
    if threads is None:
        threads = int(os.environ.get("TORCH_NUM_THREADS",
                      os.environ.get("OMP_NUM_THREADS", 8)))
    torch.set_num_threads(threads)
    print(f"[embed] torch threads = {threads}", flush=True)
    model = backbones.load(backbone)
    mean = torch.tensor(model.mean).view(1, 3, 1, 1)
    std = torch.tensor(model.std).view(1, 3, 1, 1)

    def enc(tiles: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float().div_(255.0)
        return model((x - mean) / std).numpy()

    kept, vecs = [], []
    for i, lab, raw in _iter_lung(parquet_dir):
        if rows is not None and i not in rows:
            continue
        rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        H, W = rgb.shape[:2]
        tiles = [rgb[y:y + TILE, x:x + TILE]
                 for y in range(0, H - TILE + 1, TILE)
                 for x in range(0, W - TILE + 1, TILE)]
        if not tiles:
            tiles = [np.array(Image.fromarray(rgb).resize((TILE, TILE), Image.BILINEAR))]
        v = enc([np.ascontiguousarray(t) for t in tiles]).mean(axis=0)
        vecs.append(v.astype(np.float16))
        kept.append(i)
        if len(kept) % 250 == 0:
            print(f"  embedded {len(kept)} images", flush=True)

    np.save(cache / f"lc_tilemean_{backbone}.npy", np.stack(vecs))
    pd.DataFrame({"row": kept}).to_csv(cache / f"lc_embedded_rows_{backbone}.csv",
                                       index=False)
    print(f"embedded {len(kept)} LC25000 lung images with {backbone} -> {cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, default=Path("data/lc25000"))
    ap.add_argument("--cache", type=Path, default=Path("data/features/lc25000"))
    ap.add_argument("--backbone", default="resnet50_in1k")
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--outdir", type=Path, default=Path("results/tables"))
    ap.add_argument("--stages", nargs="+", default=["fingerprint", "group", "embed"])
    ap.add_argument("--embed-per-group", type=int, default=1,
                    help="representatives per recovered group to encode")
    ap.add_argument("--embed-extra-per-class", type=int, default=600,
                    help="additional random images per class, for the "
                         "replication-inflation comparison")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    a.outdir.mkdir(parents=True, exist_ok=True)

    if "fingerprint" in a.stages and not (a.cache / "lc_fingerprints.npy").exists():
        stage_fingerprint(a.parquet_dir, a.cache)

    meta = pd.read_csv(a.cache / "lc_meta.csv")
    if "group" in a.stages:
        fps = np.load(a.cache / "lc_fingerprints.npy")
        meta["group_id"] = recover_groups(fps, meta.lc_label.values, a.tol)
        meta.to_csv(a.cache / "lc_meta.csv", index=False)
        diag = group_diagnostics(meta)
        diag.to_csv(a.outdir / "lc25000_group_recovery.csv", index=False)
        print(diag.to_string(index=False))
        print(f"\nrecovered {meta.group_id.nunique()} groups from {len(meta)} "
              f"lung images; LC25000 documents 250 original fields per class")

    if "embed" in a.stages:
        rng = np.random.default_rng(0)
        sel: set[int] = set()
        if "group_id" in meta.columns:
            for _, g in meta.groupby("group_id"):
                sel.update(g.row.head(a.embed_per_group).tolist())
        for lab, g in meta.groupby("lc_label"):
            pool = g.row.values
            k = min(a.embed_extra_per_class, len(pool))
            sel.update(rng.choice(pool, size=k, replace=False).tolist())
        print(f"encoding {len(sel)} of {len(meta)} lung images")
        stage_embed(a.parquet_dir, a.cache, a.backbone, sel)
