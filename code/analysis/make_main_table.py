"""Render Table 2: the best patient-disjoint configuration for each task.

The manuscript's one results table. It carries, per task, the strongest
configuration found under the patient-disjoint protocol with its bootstrap
interval, the chance level so the seven-class and grading figures can be read
without arithmetic, and the dataset descriptor's patient-wise baseline, which
is the only prior figure on comparable footing.

The table deliberately does not carry an inflation column. The best
patient-disjoint configuration and the image-level figure come from different
encoders and representations, so differencing them would compare unmatched
arms; the matched inflation is reported in the text and in Figure 2, computed
on identical features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

T = Path("results/tables")
TASK_LABEL = {"super3": "3-class diagnosis", "full7": "7-class subtype + grade",
              "grade3": "3-class differentiation grade"}
CHANCE = {"super3": 1 / 3, "full7": 1 / 7, "grade3": 1 / 3}
DISPLAY = {"resnet50_in1k": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
           "phikon_vitb16": "Phikon ViT-B/16"}


def _esc(v: str) -> str:
    return str(v).replace("_", r"\_").replace("+", "$+$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/tables/main_results.tex"))
    a = ap.parse_args()

    s2 = pd.read_csv(T / "summary_e2_backbones.csv")
    # Diosdado et al. 2024, patient-wise 80/10/10, accuracy (not balanced).
    DESCRIPTOR = {"super3": "0.90 at 20$\\times$ / 0.82 at 40$\\times$*",
                  "full7": "not reported", "grade3": "not reported"}

    rows = []
    for task in ("super3", "full7", "grade3"):
        g = s2[(s2.task == task) & (s2.protocol == "patient") &
               (s2["head"] == "linear") & (s2.magnification == "both")]
        if not len(g):
            continue
        b = g.loc[g.balanced_accuracy_mean.idxmax()]
        rows.append(" & ".join([
            TASK_LABEL[task],
            f"{CHANCE[task]:.3f}",
            f"{DISPLAY.get(b.backbone, b.backbone)}, {_esc(b.representation)}",
            f"\\textbf{{{b.balanced_accuracy_mean:.3f}}} "
            f"[{b.balanced_accuracy_lo:.3f}, {b.balanced_accuracy_hi:.3f}]",
            DESCRIPTOR[task],
        ]) + r" \\")

    tex = "\n".join([
        r"\footnotesize",
        r"\begin{tabular}{@{}l c l c c@{}}",
        r"\toprule",
        r"Task & Chance & Best configuration & Patient-disjoint (this work) & Descriptor baseline \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tex + "\n")
    print(f"wrote {a.out} ({len(rows)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
