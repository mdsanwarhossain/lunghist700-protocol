"""Encode every LungHist700 image once with a frozen backbone.

Writes, per (backbone, stain-normalisation) pair:
  tiles.npy     float16 [n_tiles, dim]   tile embeddings
  tiles.csv                              tile -> image/patient/label/position
  global.npy    float16 [n_images, dim]  whole field resized to 224 (the
                                         original paper's "resize" view)
  handcrafted.csv                        tissue-fraction and colour statistics,
                                         used for the shortcut-baseline control

Encoding is the only expensive step in the study and a full pass costs tens of
minutes per pair on CPU, so it is checkpointed: images are processed in blocks
and each finished block is written to ``_shards/`` immediately. A rerun skips
the blocks already on disk, which makes the pass survive an interrupted
machine instead of starting over.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import backbones  # noqa: E402
from preprocess import tiling  # noqa: E402

BLOCK = 25  # images per checkpoint shard


def _to_batch(arrs: list[np.ndarray], mean, std) -> torch.Tensor:
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


@torch.inference_mode()
def _encode_block(model, normalize, norm: str, idx: pd.DataFrame,
                  lo: int, hi: int, batch: int) -> dict:
    """Encode images [lo, hi) and return the arrays that make up one shard."""
    tile_vecs: list[np.ndarray] = []
    tile_meta: list[dict] = []
    global_vecs = np.zeros((hi - lo, model.dim), dtype=np.float16)
    hand: list[dict] = []
    buf: list[np.ndarray] = []
    buf_meta: list[dict] = []

    def flush() -> None:
        if not buf:
            return
        out = model(_to_batch(buf, model.mean, model.std)).cpu().numpy().astype(np.float16)
        tile_vecs.append(out)
        tile_meta.extend(buf_meta)
        buf.clear()
        buf_meta.clear()

    for i in range(lo, hi):
        row = idx.iloc[i]
        rgb = np.array(Image.open(row.path).convert("RGB"))
        mask = tiling.tissue_mask(rgb)
        hand.append({
            "image_idx": i,
            "tissue_fraction": float(mask.mean()),
            "mean_r": float(rgb[..., 0].mean()), "mean_g": float(rgb[..., 1].mean()),
            "mean_b": float(rgb[..., 2].mean()),
            "std_r": float(rgb[..., 0].std()), "std_g": float(rgb[..., 1].std()),
            "std_b": float(rgb[..., 2].std()),
        })

        rgb_n = normalize(rgb) if norm != "raw" else rgb

        # whole-field view, matching the original baseline's resize strategy
        gv = model(_to_batch([np.array(Image.fromarray(rgb_n).resize((224, 224),
                                                                     Image.BILINEAR))],
                             model.mean, model.std))
        global_vecs[i - lo] = gv[0].cpu().numpy().astype(np.float16)

        for tile, (r, c), frac in tiling.grid_tiles(rgb_n):
            buf.append(np.ascontiguousarray(tile))
            buf_meta.append({"image_idx": i, "row": r, "col": c, "tissue_fraction": frac})
            if len(buf) >= batch:
                flush()
    flush()
    return {
        "tiles": np.concatenate(tile_vecs, axis=0) if tile_vecs
                 else np.zeros((0, model.dim), dtype=np.float16),
        "tile_meta": pd.DataFrame(tile_meta),
        "global": global_vecs,
        "hand": pd.DataFrame(hand),
    }


def _shard_path(shard_dir: Path, lo: int) -> Path:
    return shard_dir / f"{lo:05d}.npz"


def _save_shard(path: Path, block: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp,
             tiles=block["tiles"],
             tile_meta=block["tile_meta"].to_records(index=False),
             glob=block["global"],
             hand=block["hand"].to_records(index=False))
    tmp.replace(path)  # atomic: a half-written shard is never mistaken for done


def _load_shard(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "tiles": z["tiles"],
        "tile_meta": pd.DataFrame.from_records(z["tile_meta"]),
        "global": z["glob"],
        "hand": pd.DataFrame.from_records(z["hand"]),
    }


def _shard_ok(path: Path) -> bool:
    """Whether a checkpoint can actually be read back.

    A shard truncated by a full disk still exists on disk, so resuming on file
    existence alone makes the next pass crash on the stitch instead of simply
    re-encoding the block. Validating the archive here keeps an interrupted run
    self-healing: an unreadable shard is deleted and its block redone.
    """
    if not path.exists():
        return False
    try:
        z = np.load(path, allow_pickle=False)
        for k in ("tiles", "tile_meta", "glob", "hand"):
            _ = z[k].shape
        return True
    except Exception:
        print(f"  discarding unreadable checkpoint {path.name}", flush=True)
        path.unlink(missing_ok=True)
        return False


def run(index_csv: Path, backbone: str, norm: str, outdir: Path,
        batch: int = 32, threads: int | None = None,
        keep_shards: bool = False) -> None:
    if threads:
        torch.set_num_threads(threads)
    idx = pd.read_csv(index_csv).reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)
    shard_dir = outdir / "_shards"
    shard_dir.mkdir(exist_ok=True)

    blocks = [(lo, min(lo + BLOCK, len(idx))) for lo in range(0, len(idx), BLOCK)]
    todo = [b for b in blocks if not _shard_ok(_shard_path(shard_dir, b[0]))]
    done_imgs = len(idx) - sum(hi - lo for lo, hi in todo)
    print(f"=== {backbone} / {norm} === {len(idx)} images, "
          f"{done_imgs} already encoded, {len(todo)} blocks to go", flush=True)

    model = normalize = None
    if todo:
        model = backbones.load(backbone)
        normalize = tiling.NORMALIZERS[norm]
        t0 = time.time()
        for n, (lo, hi) in enumerate(todo, 1):
            block = _encode_block(model, normalize, norm, idx, lo, hi, batch)
            _save_shard(_shard_path(shard_dir, lo), block)
            el = time.time() - t0
            print(f"  [{backbone}/{norm}] block {n}/{len(todo)} "
                  f"(images {lo}-{hi})  {el:.0f}s  eta {el/n*(len(todo)-n):.0f}s",
                  flush=True)
        dim = model.dim
    else:
        dim = _load_shard(_shard_path(shard_dir, 0))["tiles"].shape[1]

    # ---- stitch the shards into the arrays the experiments consume
    t1 = time.time()
    tiles, metas, hands = [], [], []
    global_vecs = np.zeros((len(idx), dim), dtype=np.float16)
    for lo, hi in blocks:
        sh = _load_shard(_shard_path(shard_dir, lo))
        tiles.append(sh["tiles"])
        metas.append(sh["tile_meta"])
        hands.append(sh["hand"])
        global_vecs[lo:hi] = sh["global"]
    tiles_arr = np.concatenate(tiles, axis=0)
    tm = pd.concat(metas, ignore_index=True)
    cols = ["path", "patient_id", "label7", "superclass", "subclass", "magnification", "grade"]
    tm = tm.join(idx[cols], on="image_idx")

    np.save(outdir / "tiles.npy", tiles_arr)
    tm.to_csv(outdir / "tiles.csv", index=False)
    np.save(outdir / "global.npy", global_vecs)
    pd.concat(hands, ignore_index=True).to_csv(outdir / "handcrafted.csv", index=False)
    (outdir / "meta.txt").write_text(
        f"backbone={backbone}\nnorm={norm}\ndim={dim}\n"
        f"n_images={len(idx)}\nn_tiles={len(tiles_arr)}\n"
        f"desc={backbones.REGISTRY[backbone][1]}\n")
    if not keep_shards:
        shutil.rmtree(shard_dir)
    print(f"[{backbone}/{norm}] {len(tiles_arr)} tiles, dim={dim}, "
          f"stitched in {time.time()-t1:.0f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--norm", default="raw", choices=list(tiling.NORMALIZERS))
    ap.add_argument("--outroot", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--keep-shards", action="store_true",
                    help="retain per-block checkpoints after stitching")
    a = ap.parse_args()
    run(a.index, a.backbone, a.norm, a.outroot / f"{a.backbone}__{a.norm}",
        a.batch, a.threads, a.keep_shards)
