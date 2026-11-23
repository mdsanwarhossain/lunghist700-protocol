"""Supplementary table: the mechanism sweep and its sample-size control."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

T = Path("results/tables")


def _inflation(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ps = (d.groupby(keys + ["seed", "protocol"])
          .agg(ba=("balanced_accuracy", "mean")).reset_index())
    w = ps.pivot_table(index=keys + ["seed"], columns="protocol",
                       values="ba").reset_index()
    w["infl"] = 100 * (w["image"] - w["patient"])
    return (w.groupby(keys)
            .agg(patient=("patient", "mean"), image=("image", "mean"),
                 infl=("infl", "mean"), sd=("infl", "std")).reset_index())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("paper/tables/mechanism.tex"))
    a = ap.parse_args()
    rows = []

    for cohort, label in (("lunghist700", "LungHist700"),
                          ("breakhis", "BreakHis")):
        p = T / f"mechanism_{cohort}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        ipg = d.groupby("cap").images_per_group.first()
        g = _inflation(d, ["cap"])
        for _, r in g.sort_values("cap").iterrows():
            rows.append(f"{label} & sweep & {ipg[r.cap]:.1f} & "
                        f"{int(d[d.cap == r.cap].n_images.iloc[0])} & "
                        f"{r.patient:.3f} & {r.image:.3f} & "
                        f"{r.infl:+.1f} & {r.sd:.1f} \\\\")

    p = T / "mechanism_control_lunghist700.csv"
    if p.exists():
        d = pd.read_csv(p)
        g = _inflation(d, ["per_group", "n_groups"])
        for _, r in g.sort_values("per_group").iterrows():
            n = int(d[(d.per_group == r.per_group)].n_images.iloc[0])
            rows.append(f"LungHist700 & control & {r.per_group:.1f} & {n} & "
                        f"{r.patient:.3f} & {r.image:.3f} & "
                        f"{r.infl:+.1f} & {r.sd:.1f} \\\\")

    tex = "\n".join([
        r"\footnotesize",
        r"\begin{tabular}{@{}l l c c c c c c@{}}",
        r"\toprule",
        r"Cohort & Arm & Images/group & $n$ & Grouped & Ungrouped"
        r" & Inflation & sd \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tex + "\n")
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
