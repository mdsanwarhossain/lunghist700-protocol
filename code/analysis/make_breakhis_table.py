"""Render the BreakHis replication table: the audit's contrast on a second cohort.

Two tasks, two protocols, identical features and head throughout, so the
difference between the columns is the partition and nothing else.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

T = Path("results/tables")
TASK_LABEL = {"bh_binary": "Benign vs malignant", "bh_type8": "8 tumour types"}
CHANCE = {"bh_binary": 0.5, "bh_type8": 1 / 8}


def _ci(g: pd.DataFrame) -> tuple[float, float, float]:
    v = g.balanced_accuracy.dropna().values
    rng = np.random.default_rng(0)
    b = rng.choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("paper/tables/breakhis.tex"))
    a = ap.parse_args()

    d = pd.read_csv(T / "breakhis_folds.csv")
    d = d[(d["head"] == "linear") & (d.representation == "tilemean")]

    rows = []
    for task, label in TASK_LABEL.items():
        g = d[d.task == task]
        if not len(g):
            continue
        cells, means = [], {}
        for proto in ("patient", "image"):
            gg = g[g.protocol == proto]
            if not len(gg):
                cells.append("n.r."); continue
            m, lo, hi = _ci(gg)
            means[proto] = m
            cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
        infl = (f"\\textbf{{{100 * (means['image'] - means['patient']):+.1f}}}"
                if len(means) == 2 else "n.r.")
        overlap = int(g[g.protocol == "patient"].patient_overlap.max())
        rows.append(f"{label} & {CHANCE[task]:.3f} & {cells[0]} & {cells[1]} & "
                    f"{infl} & {overlap} \\\\")

    tex = "\n".join([
        r"\footnotesize",
        r"\begin{tabular}{@{}l c c c c c@{}}",
        r"\toprule",
        r"Task & Chance & Patient-disjoint & Image-level & Inflation (pp)"
        r" & Max shared patients \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tex + "\n")
    print(f"wrote {a.out} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
