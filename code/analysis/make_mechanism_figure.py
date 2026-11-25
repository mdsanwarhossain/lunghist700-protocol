"""The mechanism figure: when the partition matters, and why.

(a) The inflation an ungrouped split produces, against how many images each
    group contributes. At one image per group the two protocols are the same
    partition, so the inflation must be zero; the curve shows how fast it
    appears as groups grow.
(b) Why: the fraction of test images whose nearest training neighbour in
    feature space comes from their own group. Under a grouped split that is
    zero by construction; under an ungrouped one it is where the inflation
    comes from.
(c) The control a reviewer should ask for: capping images per group also
    shrinks the dataset, so this panel holds the total roughly fixed and trades
    groups against images per group instead.
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


def _curve(d: pd.DataFrame) -> pd.DataFrame:
    """Per-cap inflation, with dispersion over seeds."""
    per_seed = (d.groupby(["cap", "seed", "protocol"])
                 .agg(ba=("balanced_accuracy", "mean"),
                      ipg=("images_per_group", "first")).reset_index())
    w = per_seed.pivot_table(index=["cap", "seed", "ipg"], columns="protocol",
                             values="ba").reset_index()
    w["infl"] = 100 * (w["image"] - w["patient"])
    return (w.groupby(["cap", "ipg"])
             .agg(infl=("infl", "mean"), sd=("infl", "std")).reset_index()
             .sort_values("ipg"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/fig_mechanism.pdf"))
    a = ap.parse_args()

    frames = {}
    for cohort, label in (("lunghist700", "LungHist700 (lung, 45 patients)"),
                          ("breakhis", "BreakHis (breast, 81 patients)")):
        p = T / f"mechanism_{cohort}.csv"
        if p.exists():
            frames[label] = pd.read_csv(p)

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 3.7))

    styles = ((BLUE, "o", "-"), (ORANGE, "s", "--"))
    for (label, d), (color, mk, ls) in zip(frames.items(), styles):
        c = _curve(d)
        ax.errorbar(c.ipg, c.infl, yerr=c.sd.fillna(0), marker=mk, ms=5,
                    lw=1.8, ls=ls, capsize=3, color=color, label=label)
    ax.axhline(0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("images per group (mean)")
    ax.set_ylabel("inflation from an ungrouped split (points)")
    ax.set_title("(a) The effect appears as groups grow", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])

    first = next(iter(frames.values())) if frames else None
    if first is not None:
        g = (first.groupby(["cap", "protocol"])
             .agg(ipg=("images_per_group", "first"),
                  nn=("nn_same_group", "mean")).reset_index())
        for proto, color, lab, mk, ls in (
                ("image", ORANGE, "ungrouped split", "s", "--"),
                ("patient", BLUE, "grouped split", "o", "-")):
            gg = g[g.protocol == proto].sort_values("ipg")
            ax2.plot(gg.ipg, gg.nn, marker=mk, ms=5, lw=1.8, ls=ls,
                     color=color, label=lab)
        ax2.set_xlabel("images per group (mean)")
        ax2.set_ylabel("nearest training neighbour\nis from the same group")
        ax2.set_title("(b) Why: the test image's twin is in training",
                      fontsize=10.5)
        ax2.set_ylim(-0.03, 1.0)
        ax2.legend(fontsize=8.5, loc="upper left", frameon=False)
        ax2.set_xscale("log")
        ax2.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])

    ctrl_p = T / "mechanism_control_lunghist700.csv"
    if ctrl_p.exists():
        c = pd.read_csv(ctrl_p)
        per = (c.groupby(["per_group", "n_groups", "seed", "protocol"])
               .agg(ba=("balanced_accuracy", "mean"),
                    n=("n_images", "first")).reset_index())
        w = per.pivot_table(index=["per_group", "n_groups", "seed", "n"],
                            columns="protocol", values="ba").reset_index()
        w["infl"] = 100 * (w["image"] - w["patient"])
        cc = (w.groupby(["per_group", "n_groups", "n"])
              .agg(infl=("infl", "mean"), sd=("infl", "std")).reset_index()
              .sort_values("per_group"))
        ax3.errorbar(cc.per_group, cc.infl, yerr=cc.sd.fillna(0), marker="s",
                     ms=5, lw=1.8, capsize=3, color=BLUE)
        # a linear axis here: the cells are chosen, not swept, and the group
        # count belongs under its own tick rather than floating in the panel
        ax3.set_xticks(cc.per_group.tolist(),
                       [f"{int(v)}\n{int(n)} groups"
                        for v, n in zip(cc.per_group, cc.n_groups)],
                       fontsize=8)
        ax3.set_xlabel(f"images per group (total held near {int(cc.n.mean())})")
        ax3.set_ylabel("inflation (points)")
        ax3.set_title("(c) Not a sample-size effect", fontsize=10.5)
        ax3.set_xlim(cc.per_group.min() - 2, cc.per_group.max() + 2)
        ax3.set_ylim(0, (cc.infl + cc.sd.fillna(0)).max() * 1.15)

    for axis in (ax, ax2, ax3):
        axis.grid(False)
        axis.set_axisbelow(True)
        for sp in ("top", "right"):
            axis.spines[sp].set_visible(False)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"wrote {a.out} ({len(frames)} cohort(s))")


if __name__ == "__main__":
    main()
