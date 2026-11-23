"""Graphical abstract, which this journal requires as a separate file.

Elsevier asks for at least 531 x 1328 px (h x w), legible at 5 x 13 cm. Every
number on it is read from results/tables, so the figure cannot drift from the
measurements it summarises.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = Path("results/tables")
BLUE, ORANGE, GREY = "#2D6A9F", "#C2571A", "#6B7280"

TASKS = [("super3", "3-class"), ("grade3", "grading"), ("full7", "7-class")]
BH_TASKS = [("bh_binary", "benign /\nmalignant"), ("bh_type8", "8 types")]


def _lh_pair(folds: pd.DataFrame, task: str) -> tuple[float, float]:
    d = folds[(folds.task == task) & (folds["head"] == "linear")
              & (folds.representation == "tilemean")
              & (folds.backbone == "resnet50_in1k")
              & (folds.magnification == "both")]
    return (d[d.protocol == "patient"].balanced_accuracy.mean(),
            d[d.protocol == "image"].balanced_accuracy.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/graphical_abstract.png"))
    a = ap.parse_args()

    lh = pd.read_csv(T / "e1_leakage_folds.csv")
    pairs = {t: _lh_pair(lh, t) for t, _ in TASKS}

    bh_pairs = {}
    bh_path = T / "breakhis_folds.csv"
    if bh_path.exists():
        bh = pd.read_csv(bh_path)
        bh = bh[(bh["head"] == "linear") & (bh.representation == "tilemean")]
        for t, _ in BH_TASKS:
            d = bh[bh.task == t]
            if len(d):
                bh_pairs[t] = (d[d.protocol == "patient"].balanced_accuracy.mean(),
                               d[d.protocol == "image"].balanced_accuracy.mean())

    fig = plt.figure(figsize=(13.4, 5.4), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.25], wspace=0.32,
                          left=0.045, right=0.985, top=0.80, bottom=0.11)

    # ---- panel 1: the same model, two splits, two different numbers ---------
    ax = fig.add_subplot(gs[0])
    pat, img = pairs["super3"]
    ax.bar([0, 1], [pat, img], color=[BLUE, ORANGE], width=0.6)
    for x, v in zip([0, 1], [pat, img]):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", (1, img + 0.06), (0, pat + 0.06),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(0.5, img + 0.17, f"{100 * (img - pat):+.1f} points",
            ha="center", fontsize=12.5, color=ORANGE, fontweight="bold")
    ax.set_xticks([0, 1], ["patient-\ndisjoint", "image-\nlevel"], fontsize=11)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("balanced accuracy", fontsize=11)
    ax.set_title("Same features, same classifier.\nOnly the split changes.",
                 fontsize=12.5, fontweight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- panel 2: the procedure --------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title("The split-sensitivity audit", fontsize=12.5,
                  fontweight="bold", loc="left")
    steps = [
        ("1", "Find the grouping variable.\nIf it cannot be recovered, stop."),
        ("2", "Encode once, frozen.\nNo training, runs on a CPU."),
        ("3", "Fit matched arms:\ngrouped vs ungrouped folds."),
        ("4", "Verify disjointness per fold,\ndo not assume it."),
        ("5", "Report the contrast, and an\ninterval over groups not folds."),
    ]
    y = 0.95
    for n, text in steps:
        ax2.add_patch(plt.Circle((0.045, y - 0.045), 0.032, color=BLUE,
                                 transform=ax2.transAxes, clip_on=False))
        ax2.text(0.045, y - 0.045, n, color="white", fontsize=10.5, ha="center",
                 va="center", fontweight="bold", transform=ax2.transAxes)
        ax2.text(0.115, y - 0.045, text, fontsize=10.2, va="center",
                 transform=ax2.transAxes)
        y -= 0.197

    # ---- panel 3: it replicates --------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    labels, vals, colors = [], [], []
    for t, lab in TASKS:
        p, i = pairs[t]
        labels.append(lab); vals.append(100 * (i - p)); colors.append(BLUE)
    for t, lab in BH_TASKS:
        if t in bh_pairs:
            p, i = bh_pairs[t]
            labels.append(lab); vals.append(100 * (i - p)); colors.append(ORANGE)
    ypos = np.arange(len(labels))
    ax3.barh(ypos, vals, color=colors, height=0.62)
    for yy, v in zip(ypos, vals):
        ax3.text(v + 0.6, yy, f"{v:+.1f}", va="center", fontsize=11,
                 fontweight="bold")
    ax3.set_yticks(ypos, labels, fontsize=10.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("inflation from an ungrouped split (points)", fontsize=10.5)
    ax3.set_xlim(0, max(vals) * 1.22 if vals else 1)
    ax3.set_title("Two cohorts, same effect", fontsize=12.5,
                  fontweight="bold", loc="left", pad=16)
    ax3.grid(axis="x", alpha=0.3)
    ax3.set_axisbelow(True)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)
    if bh_pairs:
        # above the axes, beside the left-aligned title, so no bar is covered
        ax3.text(1.0, 1.015, "blue: LungHist700 (lung, 45 patients)   "
                 "orange: BreakHis (breast, 81 patients)", fontsize=9,
                 color=GREY, ha="right", va="bottom", transform=ax3.transAxes)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100)
    fig.savefig(a.out.with_suffix(".pdf"))
    plt.close(fig)
    im = plt.imread(a.out)
    print(f"wrote {a.out}  {im.shape[1]}x{im.shape[0]} px "
          f"(Elsevier minimum 1328x531)")


if __name__ == "__main__":
    main()
