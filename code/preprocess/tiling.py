"""Tissue detection, tiling and stain normalisation for LungHist700 fields.

Every image is 1600x1200, so a 224-px grid yields 7x5 = 35 candidate tiles.
Tiles that are mostly glass carry no diagnostic signal and only dilute a mean-
pooled representation, so they are dropped by a saturation-based tissue test.
"""
from __future__ import annotations

import cv2
import numpy as np

TILE = 224


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean tissue mask via Otsu on HSV saturation plus a brightness veto.

    H&E tissue is chromatic; glass is bright and nearly grey. Saturation alone
    mislabels faint shadows, so very bright pixels are excluded explicitly.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[..., 1], (0, 0), 2)
    thr, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(thr, 12, 60))  # Otsu collapses on near-uniform fields
    return (sat > thr) & (hsv[..., 2] < 235)


def grid_tiles(rgb: np.ndarray, tile: int = TILE, min_tissue: float = 0.35):
    """Yield ``(tile_rgb, (row, col), tissue_fraction)`` for tissue-bearing tiles.

    Falls back to the most tissue-rich tiles when the mask is too strict, so no
    image is ever represented by an empty bag.
    """
    H, W = rgb.shape[:2]
    mask = tissue_mask(rgb)
    cand = []
    for r, y in enumerate(range(0, H - tile + 1, tile)):
        for c, x in enumerate(range(0, W - tile + 1, tile)):
            frac = float(mask[y : y + tile, x : x + tile].mean())
            cand.append((frac, r, c, y, x))
    keep = [t for t in cand if t[0] >= min_tissue]
    if len(keep) < 4:  # degenerate field: keep the four richest tiles regardless
        keep = sorted(cand, reverse=True)[:4]
    for frac, r, c, y, x in keep:
        yield rgb[y : y + tile, x : x + tile], (r, c), frac


# --------------------------------------------------------------------------
# Stain normalisation
# --------------------------------------------------------------------------
# Macenko reference stain matrix and concentrations, as distributed with the
# original method; used so every image maps onto one common H&E appearance.
_REF_HE = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
_REF_MAX_C = np.array([1.9705, 1.0308])


def _od(rgb: np.ndarray, io: int = 240) -> np.ndarray:
    rgb = rgb.reshape(-1, 3).astype(np.float64)
    return -np.log10(np.maximum(rgb, 1.0) / io)


def macenko_normalize(rgb: np.ndarray, io: int = 240, beta: float = 0.15,
                      alpha: float = 1.0) -> np.ndarray:
    """Macenko stain normalisation; returns the input unchanged if the field
    carries too little optical density to estimate a stain basis."""
    h, w = rgb.shape[:2]
    od = _od(rgb, io)
    od_h = od[~np.any(od < beta, axis=1)]
    if od_h.shape[0] < 100:
        return rgb

    cov = np.cov(od_h.T)
    if not np.all(np.isfinite(cov)):
        return rgb
    _, evecs = np.linalg.eigh(cov)
    plane = od_h @ evecs[:, 1:3]            # two leading eigenvectors
    phi = np.arctan2(plane[:, 1], plane[:, 0])
    v_min = evecs[:, 1:3] @ np.array([np.cos(np.percentile(phi, alpha)),
                                      np.sin(np.percentile(phi, alpha))])
    v_max = evecs[:, 1:3] @ np.array([np.cos(np.percentile(phi, 100 - alpha)),
                                      np.sin(np.percentile(phi, 100 - alpha))])
    # haematoxylin is the heavier stain, so order by first-channel magnitude
    he = np.stack([v_min, v_max] if v_min[0] > v_max[0] else [v_max, v_min], axis=1)

    conc, *_ = np.linalg.lstsq(he, od.T, rcond=None)
    max_c = np.percentile(conc, 99, axis=1)
    if np.any(max_c <= 1e-6):
        return rgb
    conc = conc * (_REF_MAX_C / max_c)[:, None]

    out = io * np.exp(-_REF_HE @ conc)
    return np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)


def reinhard_normalize(rgb: np.ndarray,
                       target_mean=(72.0, 1.5, -6.5),
                       target_std=(16.0, 10.5, 6.0)) -> np.ndarray:
    """Reinhard colour transfer in LAB, the cheaper normalisation baseline."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    for i in range(3):
        m, s = lab[..., i].mean(), lab[..., i].std()
        if s > 1e-6:
            lab[..., i] = (lab[..., i] - m) / s * target_std[i] + target_mean[i]
    lab[..., 0] = np.clip(lab[..., 0], 0, 100) * 255.0 / 100.0
    lab[..., 1:] = np.clip(lab[..., 1:] + 128.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


NORMALIZERS = {
    "raw": lambda x: x,
    "macenko": macenko_normalize,
    "reinhard": reinhard_normalize,
}
