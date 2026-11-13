"""Render the protocol audit of published LungHist700 results as a LaTeX table.

The audit lives in results/tables/related_work.csv, which is maintained by hand
and records, per study, the split protocol and which of the reporting practices
this paper argues for are present. Keeping the manuscript table generated from
that file means the prose and the audit cannot drift apart.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# csv column -> (column heading, how to render a cell)
TICK = {"yes": r"\checkmark", "no": r"--", "none": r"--", "n/a": r"n/a"}


def _flag(v: str) -> str:
    s = str(v).strip().lower()
    if s in TICK:
        return TICK[s]
    if s in ("not reported", "not stated", ""):
        return r"n.r."
    if s.startswith("no"):
        return r"--"
    if s.startswith("yes"):
        return r"\checkmark"
    if s == "stain-conditioned, not normalised":
        return r"($\sim$)"
    if s.startswith("sd over") or s.endswith("sd"):
        return r"(sd)"
    # A qualified "yes" keeps the tick but carries its qualifier as a footnote
    # marker, so the table cannot overstate what a study actually did.
    if s.startswith("yes") or "paired t-test" in s:
        return r"\checkmark"
    return _esc(v)


def _esc(v: object) -> str:
    s = str(v)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


# The CSV spells protocols out in prose; the table column is narrow.
PROTOCOL_SHORT = [
    ("patient-wise single 80/10/10", "patient, single 80/10/10"),
    ("patient-level 70/15/15 + stratified 5-fold CV",
     "patient 70/15/15 $+$ image-level 5-fold"),
    ("patient-level single 70/15/15", "patient, single 70/15/15"),
    ("5-fold CV over the pooled, augmented image set",
     "image-level 5-fold, pooled w/ LC25000"),
    ("five-client non-IID federated simulation", "federated, 5 non-IID clients"),
    ("image-level single fixed split, no patient information",
     "image-level single split"),
    ("patient-level 70/15/15 hold-out; the 5-fold CV over train+val is "
     "stratified, NOT patient-grouped",
     "patient 70/15/15; 5-fold CV \\emph{not} patient-grouped"),
    ("image-level stratified single test split; no patient separation reported",
     "image-level stratified single split"),
    ("k-fold CV; number of folds and grouping not reported",
     "$k$-fold, grouping unreported"),
    ("stratified single split; grouping not determinable from the available text",
     "stratified single split; grouping not determinable"),
    ("5-fold CV over the pooled, augmented image set",
     "5-fold, pooled w/ LC25000"),
]


def _protocol(v: str) -> str:
    s = str(v).strip()
    for long, short in PROTOCOL_SHORT:
        if s == long:
            return _esc(short)
    return _esc(s)


# Where each row's protocol information came from. Rows audited from the full
# text are the only ones where an absence of reporting can be asserted; the
# others are limited by what the abstract or publisher preview discloses.
SOURCE_SHORT = {"full text": "F", "abstract": "A", "preview": "P"}


def _source(v: object) -> str:
    return SOURCE_SHORT.get(str(v).strip().lower(), _esc(v))


def _short(study: str) -> str:
    """First author plus 'et al.' where the row lists several."""
    s = study.split("(")[0].strip().rstrip(",")
    first = s.split("&")[0].split(",")[0].strip()
    return f"{_esc(first)} et al." if ("&" in s or "others" in s) else _esc(first)


def build(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        rows.append(" & ".join([
            f"{_short(r['study'])}~\\cite{{{r['citekey']}}}",
            _esc(r["year"]),
            _esc(r["n_classes_evaluated"]),
            _protocol(r["split_protocol"]),
            _flag(r["repeated_cv"]),
            _esc(r["reported_headline"]).replace("+-", "$\\pm$")
                                        .replace("20x", "20$\\times$")
                                        .replace("40x", "40$\\times$"),
            _flag(r["reports_7class"]),
            _flag(r["reports_ci_or_tests"]),
            _flag(r["stain_normalisation"]),
            _flag(r["calibration_or_abstention"]),
            _source(r.get("verification", "")),
        ]) + r" \\")

    return "\n".join([
        r"\footnotesize",
        r"\begin{tabular}{@{}l c c p{0.175\linewidth} c p{0.15\linewidth} ccccc@{}}",
        r"\toprule",
        r"Study & Year & \#cls & Split protocol & Rep.\ CV & Headline &"
        r" 7-cls & CI/test & Stain & Calib. & Source \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("results/tables/related_work.csv"))
    ap.add_argument("--out", type=Path, default=Path("paper/tables/related_work.tex"))
    a = ap.parse_args()

    df = pd.read_csv(a.csv, keep_default_na=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build(df) + "\n")
    print(f"wrote {a.out} ({len(df)} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
