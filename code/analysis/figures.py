"""Publication figures.

One visual language across every panel: a single categorical palette, direct
labelling instead of legends where a legend would cost a lookup, and intervals
drawn rather than implied.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Categorical palette: distinguishable in grayscale print and to the most
# common colour-vision deficiencies.
PALETTE = ["#2D6A9F", "#C2571A", "#4B8B3B", "#8E44AD", "#B03A48", "#6B7280"]
PROTOCOL_COLOR = {"patient": "#2D6A9F", "image": "#C2571A", "tile": "#B03A48"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure -> {out/name}.png")


def fig_leakage(folds: pd.DataFrame, out: Path, metric: str = "balanced_accuracy",
                head: str = "linear") -> None:
    """Identical features and classifier, three split protocols."""
    d = folds[(folds["head"] == head) & folds.protocol.isin(PROTOCOL_COLOR)]
    tasks = [t for t in ("super3", "full7", "grade3") if t in set(d.task)]
    titles = {"super3": "3-class diagnosis", "full7": "7-class subtype + grade",
              "grade3": "3-class differentiation grade"}
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.6 * len(tasks), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        sub = d[d.task == task]
        for i, proto in enumerate(["patient", "image", "tile"]):
            v = sub[sub.protocol == proto][metric].dropna().values
            if not len(v):
                continue
            ax.bar(i, v.mean(), 0.62, color=PROTOCOL_COLOR[proto],
                   edgecolor="white", linewidth=0.8)
            sd = v.std(ddof=1)
            ax.errorbar(i, v.mean(), yerr=sd, color="#1F2937", capsize=3, lw=1.1)
            # Place the label clear of the error bar, not of the bar top, or it
            # collides with the whisker whenever dispersion is large -- which is
            # exactly the patient-disjoint case the figure exists to show.
            ax.annotate(f"{v.mean():.3f}", (i, v.mean() + sd),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=8)
        n_cls = {"super3": 3, "full7": 7, "grade3": 3}[task]
        ax.axhline(1 / n_cls, ls=":", lw=1, color="#6B7280")
        ax.annotate("chance", (2.45, 1 / n_cls), fontsize=7, color="#6B7280",
                    va="bottom", ha="right")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["patient-\ndisjoint", "image-\nlevel", "tile-\nlevel"])
        ax.set_title(titles[task])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("balanced accuracy")
    fig.suptitle("The split, not the model: same features, same classifier",
                 fontsize=10.5, y=1.04)
    _save(fig, out, "fig_leakage")


def fig_backbones(summary: pd.DataFrame, out: Path,
                  metric: str = "balanced_accuracy") -> None:
    """Frozen-representation comparison under the patient-disjoint protocol."""
    d = summary[(summary.protocol == "patient") & (summary["head"] == "linear") &
                (summary.magnification == "both") &
                (summary.representation == "tilemean")]
    tasks = [t for t in ("super3", "full7", "grade3") if t in set(d.task)]
    order = (d[d.task == "full7"].sort_values(f"{metric}_mean", ascending=False)
             .backbone.tolist())
    order = order or sorted(d.backbone.unique())
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    w = 0.8 / len(tasks)
    for j, task in enumerate(tasks):
        xs, ys, los, his = [], [], [], []
        for i, b in enumerate(order):
            r = d[(d.task == task) & (d.backbone == b)]
            if r.empty:
                continue
            xs.append(i + j * w - 0.4 + w / 2)
            ys.append(float(r[f"{metric}_mean"].iloc[0]))
            los.append(ys[-1] - float(r[f"{metric}_lo"].iloc[0]))
            his.append(float(r[f"{metric}_hi"].iloc[0]) - ys[-1])
        ax.bar(xs, ys, w * 0.92, label=task, color=PALETTE[j],
               edgecolor="white", linewidth=0.7)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#1F2937",
                    capsize=2.5, lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([b.replace("_", "\n") for b in order], fontsize=8)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="task", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Frozen representations, patient-disjoint repeated CV", pad=24)
    _save(fig, out, "fig_backbones")


# Error types in the order a pathologist would rank their cost: calling a
# malignant field normal is worse than confusing two adjacent grades.
ERROR_ORDER = [("err_normal_vs_malignant", "normal vs malignant"),
               ("err_subtype", "subtype (ACA vs SCC)"),
               ("err_grade_distant", "grade, two steps"),
               ("err_grade_adjacent", "grade, one step")]


def fig_confusion(preds: pd.DataFrame, classes: list[str], out: Path,
                  name: str = "fig_confusion", title: str = "",
                  errors: dict[str, float] | None = None) -> None:
    """Row-normalised confusion over pooled out-of-fold predictions.

    With ``errors`` the figure gains a second panel decomposing the error mass
    by the kind of mistake, because on this task the headline number is
    dominated by one-step grade confusions whose clinical cost is low.
    """
    M = np.zeros((len(classes), len(classes)))
    for t, p in zip(preds.y_true, preds.y_pred):
        if t in classes and p in classes:
            M[classes.index(t), classes.index(p)] += 1
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    if errors:
        fig, (ax, axb) = plt.subplots(
            1, 2, figsize=(0.62 * len(classes) + 6.0, 0.62 * len(classes) + 1.7),
            gridspec_kw={"width_ratios": [1.35, 1.0]})
    else:
        axb = None
        fig, ax = plt.subplots(figsize=(0.62 * len(classes) + 2.0,
                                        0.62 * len(classes) + 1.7))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if M[i, j]:
                ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if Mn[i, j] > 0.55 else "#1F2937")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference")
    ax.grid(False)
    ax.set_title(("(a) " if axb is not None else "") +
                 (title or "Pooled out-of-fold confusion"), fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="row fraction")

    if axb is not None:
        keys = [(k, lab) for k, lab in ERROR_ORDER if k in errors]
        vals = [errors[k] for k, _ in keys]
        ypos = np.arange(len(keys))
        axb.barh(ypos, vals, color=["#B4472E", "#C2571A", "#5B8CB8", "#9CB8D1"][:len(keys)])
        for y, v in zip(ypos, vals):
            axb.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8)
        axb.set_yticks(ypos, [lab for _, lab in keys], fontsize=8)
        axb.invert_yaxis()
        axb.set_xlabel("share of all predictions", fontsize=8)
        axb.set_xlim(0, max(vals) * 1.28)
        axb.set_title("(b) Which mistakes the errors are", fontsize=9)
        axb.grid(axis="x", alpha=0.3)
        axb.set_axisbelow(True)
    _save(fig, out, name)


def fig_magnification(transfer: pd.DataFrame, out: Path,
                      metric: str = "balanced_accuracy") -> None:
    """Matched vs mismatched magnification, patient-disjoint throughout."""
    tasks = sorted(transfer.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.9 * len(tasks), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        d = transfer[transfer.task == task]
        cells, labels, colors = [], [], []
        for mtr in ("20x", "40x"):
            for mte in ("20x", "40x"):
                v = d[(d.train_mag == mtr) & (d.test_mag == mte)][metric].dropna()
                if not len(v):
                    continue
                cells.append(v)
                labels.append(f"{mtr}→{mte}")
                colors.append("#2D6A9F" if mtr == mte else "#C2571A")
        bp = ax.boxplot(cells, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#1F2937", lw=1.2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(task)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("balanced accuracy")
    fig.suptitle("Train→test magnification (blue matched, orange mismatched)",
                 fontsize=10, y=1.04)
    _save(fig, out, "fig_magnification")


def fig_calibration(rel: pd.DataFrame, rc: pd.DataFrame, out: Path,
                    label: str = "") -> None:
    """Reliability curve and the abstention trade-off it enables."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
    a1.plot([0, 1], [0, 1], ls=":", color="#6B7280", lw=1)
    m = rel.n > 0
    a1.plot(rel.mean_confidence[m], rel.empirical_accuracy[m], "o-",
            color="#2D6A9F", ms=4, lw=1.4)
    a1.set_xlabel("mean predicted confidence")
    a1.set_ylabel("empirical accuracy")
    a1.set_title("Reliability")
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.annotate("perfectly\ncalibrated", (0.72, 0.60), fontsize=7, color="#6B7280")

    a2.plot(rc.coverage, rc.selective_risk, "o-", color="#C2571A", ms=4, lw=1.4)
    a2.set_xlabel("coverage (fraction of fields reported)")
    a2.set_ylabel("error among reported fields")
    a2.set_title("Risk vs coverage")
    a2.xaxis.set_major_formatter(PercentFormatter(1.0))
    a2.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle(label or "Confidence quality", fontsize=10, y=1.05)
    _save(fig, out, "fig_calibration")


def fig_learning_curve(curve: pd.DataFrame, out: Path,
                       metric: str = "balanced_accuracy") -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for j, task in enumerate(sorted(curve.task.unique())):
        d = curve[curve.task == task]
        g = d.groupby("fraction").agg(
            n=("n_train_patients", "mean"), m=(metric, "mean"), s=(metric, "std"))
        ax.plot(g.n, g.m, "o-", color=PALETTE[j], ms=4, lw=1.5, label=task)
        ax.fill_between(g.n, g.m - g.s, g.m + g.s, color=PALETTE[j], alpha=0.15)
    ax.set_xlabel("training patients")
    ax.set_ylabel("balanced accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="task")
    ax.set_title("Accuracy is still patient-limited at n = 45")
    _save(fig, out, "fig_learning_curve")


# ---------------------------------------------------------------------------
# Figure 1: why the split matters on this dataset
# ---------------------------------------------------------------------------
CLASS_LABEL = {"aca_bd": "ACA\nwell", "aca_md": "ACA\nmod", "aca_pd": "ACA\npoor",
               "nor": "normal", "scc_bd": "SCC\nwell", "scc_md": "SCC\nmod",
               "scc_pd": "SCC\npoor"}


def _protocol_schematic(ax) -> None:
    """Draw what each splitter is allowed to separate.

    Two patients, three images each, four tiles per image. Colour marks the
    side of the split, so the reader sees directly that `patient` keeps a
    patient whole, `image` cuts between a patient's images, and `tile` cuts
    inside a single image.
    """
    train, test = PROTOCOL_COLOR["patient"], PROTOCOL_COLOR["image"]
    rows = {
        # protocol -> which side each of the 6 images (2 patients x 3) is on,
        # or, for `tile`, which side each tile of each image is on
        "patient": [0, 0, 0, 1, 1, 1],
        "image":   [0, 0, 1, 0, 1, 1],
    }
    y = 0.0
    for name in ("patient", "image", "tile"):
        ax.text(-0.9, y + 0.17, name, ha="right", va="center", fontsize=8.5,
                color="#374151", fontweight="bold")
        for img in range(6):
            x0 = img * 1.25 + (0.45 if img >= 3 else 0.0)
            for t in range(4):
                if name == "tile":
                    side = (img + t) % 2          # split runs inside each image
                else:
                    side = rows[name][img]
                ax.add_patch(plt.Rectangle((x0 + t * 0.26, y), 0.22, 0.34,
                                           facecolor=train if side == 0 else test,
                                           edgecolor="white", linewidth=0.6))
        y -= 0.62

    # patient brackets along the top
    for p, x0 in ((1, 0.0), (2, 4.2)):
        ax.plot([x0, x0 + 3.3], [0.52, 0.52], color="#374151", lw=0.9)
        ax.text(x0 + 1.65, 0.60, f"patient {p}", ha="center", fontsize=8,
                color="#374151")
    ax.text(0.0, -1.55, "each block is one tile; three images per patient",
            fontsize=7.5, color="#6B7280")
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in (train, test)]
    ax.legend(h, ["training side", "test side"], loc="lower right",
              fontsize=7.5, ncol=2, bbox_to_anchor=(1.02, -0.12))
    ax.set_xlim(-2.6, 8.2)
    ax.set_ylim(-1.7, 0.8)
    ax.axis("off")


def _pipeline_schematic(ax, n_encoders: int = 4) -> None:
    """The study design: encode once, then vary one factor at a time.

    The point the panel has to carry is that the grid hangs off a single cached
    embedding store, because that is what licenses attributing a difference
    between two arms to the factor named in the comparison.
    """
    box = dict(boxstyle="round,pad=0.42", linewidth=0.9)
    steps = [
        (0.5, "691 fields\n1600$\\times$1200", "#E8EEF4", "#2D6A9F"),
        (2.6, "tile grid at\nnative resolution", "#E8EEF4", "#2D6A9F"),
        (4.8, f"{n_encoders} frozen\nencoders", "#E8EEF4", "#2D6A9F"),
        (7.0, "cached embedding\nstore", "#FDF0E6", "#C2571A"),
    ]
    for x, text, fc, ec in steps:
        ax.text(x, 2.25, text, ha="center", va="center", fontsize=7.6,
                bbox={**box, "facecolor": fc, "edgecolor": ec})
    for x in (1.45, 3.62, 5.85):
        ax.annotate("", (x + 0.45, 2.25), (x, 2.25),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#6B7280"))
    ax.text(3.75, 2.88, "encoded once per (encoder, stain) pair",
            ha="center", fontsize=7, color="#6B7280", style="italic")

    # the grid fitted on top of the store
    factors = [("task", "super3 / full7 / grade3"),
               ("protocol", "patient / image / tile"),
               ("representation", "tile-mean / global / fused"),
               ("head", "linear / MLP / attention-MIL"),
               ("magnification", "20$\\times$ / 40$\\times$ / both")]
    y = 1.45
    for name, levels in factors:
        ax.text(5.25, y, name, ha="right", va="center", fontsize=7.4,
                color="#374151", fontweight="bold")
        ax.text(5.45, y, levels, ha="left", va="center", fontsize=7.4,
                color="#4B5563")
        y -= 0.30
    ax.annotate("", (7.0, 1.52), (7.0, 2.02),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#6B7280"))
    ax.text(2.0, 1.52, "cross-validated heads\nover the grid:", ha="center",
            va="top", fontsize=7.6, color="#374151")
    ax.add_patch(plt.Rectangle((5.3, -0.07), 4.0, 1.68, facecolor="none",
                               edgecolor="#9CA3AF", linewidth=0.8,
                               linestyle="--"))
    ax.set_xlim(-0.6, 9.8)
    ax.set_ylim(-0.3, 3.2)
    ax.axis("off")


def fig_design(incidence: pd.DataFrame, out: Path, n_encoders: int = 4) -> None:
    """Patient-class coupling, and what each protocol permits to be shared.

    ``n_encoders`` is passed in rather than fixed, so the schematic cannot drift
    from the number of encoders the benchmark actually ran.
    """
    d = incidence.set_index("patient_id")
    d = d[[c for c in CLASS_LABEL if c in d.columns]]

    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 1.0, 0.92])
    ax = fig.add_subplot(gs[0])

    m = d.values.astype(float)
    im = ax.imshow(np.where(m > 0, m, np.nan), aspect="auto", cmap="Blues",
                   vmin=0, vmax=np.nanmax(m))
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels([CLASS_LABEL[c] for c in d.columns], fontsize=7.5)
    ax.set_yticks(range(0, len(d), 4))
    ax.set_yticklabels(d.index[::4], fontsize=7)
    ax.set_ylabel("patient")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="images", fraction=0.03, pad=0.015)

    one = int((d.gt(0).sum(axis=1) == 1).sum())
    ax.set_title(f"(a)  {one} of {len(d)} patients contribute a single class, "
                 "so patient identity carries the label", fontsize=9.5, loc="left")

    ax2 = fig.add_subplot(gs[1])
    _protocol_schematic(ax2)
    ax2.set_title("(b)  what each protocol allows to be shared across the split",
                  fontsize=9.5, loc="left")

    ax3 = fig.add_subplot(gs[2])
    _pipeline_schematic(ax3, n_encoders)
    ax3.set_title("(c)  study design: one cached store, one factor varied at a time",
                  fontsize=9.5, loc="left")
    _save(fig, out, "fig_design")


# ---------------------------------------------------------------------------
# Figure 2: the paper's headline comparison, on one scale
# ---------------------------------------------------------------------------
def fig_protocol_vs_representation(folds: pd.DataFrame, summary: pd.DataFrame,
                                   out: Path, task: str = "full7",
                                   metric: str = "balanced_accuracy") -> None:
    """Protocol effect and representation effect side by side, shared y-axis.

    The two panels are only comparable because they share a scale; that shared
    scale is the argument, so it is set from the union of both panels rather
    than per-axis.
    """
    lo_hi = []
    d1 = folds[(folds.task == task) & (folds["head"] == "linear") &
               folds.protocol.isin(PROTOCOL_COLOR)]
    # E2 may not have run yet; treat a missing or empty summary as "no second
    # panel" rather than letting an attribute lookup fail on an empty frame.
    if summary is None or not len(summary) or "task" not in summary.columns:
        d2 = pd.DataFrame()
    else:
        d2 = summary[(summary.task == task) & (summary.protocol == "patient") &
                     (summary["head"] == "linear")]
    if len(d1):
        lo_hi += [d1[metric].min(), d1[metric].max()]
    if len(d2) and f"{metric}_lo" in d2.columns:
        lo_hi += [d2[f"{metric}_lo"].min(), d2[f"{metric}_hi"].max()]
    if not lo_hi:
        print("  fig_protocol_vs_representation: no data yet, skipped")
        return
    lo, hi = min(lo_hi), max(lo_hi)
    pad = 0.08 * (hi - lo + 1e-6)

    # The protocol panel is the paper's headline and depends only on E1, so it
    # is drawn as soon as that stage finishes rather than waiting for E2.
    solo = not len(d2)
    if solo:
        fig, axa = plt.subplots(figsize=(3.6, 3.4))
        axb = None
    else:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True,
                                       gridspec_kw={"width_ratios": [1.0, 1.25]})

    for i, proto in enumerate(["patient", "image", "tile"]):
        v = d1[d1.protocol == proto][metric].dropna().values
        if not len(v):
            continue
        axa.bar(i, v.mean(), 0.6, color=PROTOCOL_COLOR[proto], edgecolor="white")
        axa.scatter(np.full(len(v), i) + np.random.default_rng(0)
                    .uniform(-0.13, 0.13, len(v)), v, s=7, color="#1F2937",
                    alpha=0.55, zorder=3, linewidths=0)
        axa.annotate(f"{v.mean():.3f}", (i, v.mean()), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=8)
    axa.set_xticks(range(3))
    axa.set_xticklabels(["patient-\ndisjoint", "image-\nlevel", "tile-\nlevel"])
    axa.set_title("(a)  same features, same classifier", fontsize=9.5, loc="left")
    axa.set_ylabel("balanced accuracy")

    if solo:
        axa.set_ylim(max(0, lo - pad), min(1.0, hi + pad))
        axa.yaxis.set_major_formatter(PercentFormatter(1.0))
        fig.suptitle("Changing the protocol moves the score", fontsize=10.5, y=1.04)
        _save(fig, out, "fig_protocol_vs_representation")
        return

    b = d2.sort_values(f"{metric}_mean")
    for i, (_, r) in enumerate(b.iterrows()):
        axb.bar(i, r[f"{metric}_mean"], 0.6, color=PALETTE[i % len(PALETTE)],
                edgecolor="white")
        axb.errorbar(i, r[f"{metric}_mean"], color="#1F2937", capsize=3, lw=1.1,
                     yerr=[[r[f"{metric}_mean"] - r[f"{metric}_lo"]],
                           [r[f"{metric}_hi"] - r[f"{metric}_mean"]]])
    axb.set_xticks(range(len(b)))
    axb.set_xticklabels(b.backbone, rotation=20, ha="right", fontsize=7.5)
    axb.set_title("(b)  patient-disjoint throughout", fontsize=9.5, loc="left")

    for ax in (axa, axb):
        ax.set_ylim(max(0, lo - pad), min(1.0, hi + pad))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle("Changing the protocol moves the score further than changing "
                 "the encoder", fontsize=10.5, y=1.05)
    _save(fig, out, "fig_protocol_vs_representation")


# ---------------------------------------------------------------------------
# Figure 3: published figures, ordered against the protocol that produced them
# ---------------------------------------------------------------------------
SPLIT_STYLE = {
    "patient":  ("#2D6A9F", "o", "patient-wise"),
    "mixed":    ("#8E44AD", "s", "patient hold-out, CV not grouped"),
    "image":    ("#C2571A", "^", "image-level / pooled"),
    "unstated": ("#6B7280", "D", "grouping not stated"),
    "external": ("#4B8B3B", "v", "held-out external cohort"),
}


def fig_literature(related: pd.DataFrame, out: Path,
                   ours: tuple[float, float, float] | None = None) -> None:
    """Every published three-class figure, marked by the split that produced it.

    Ordering by reported value rather than by year is deliberate: the point is
    that the ranking tracks the protocol, which a chronological axis would hide.
    """
    d = related.dropna(subset=["headline_value"]).copy()
    d["headline_value"] = d.headline_value.astype(float)
    d = d.sort_values("headline_value")

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, (_, r) in enumerate(d.iterrows()):
        colour, marker, _ = SPLIT_STYLE.get(r.split_class, SPLIT_STYLE["unstated"])
        ax.plot([0, r.headline_value], [i, i], color=colour, lw=0.8, alpha=0.35)
        ax.scatter(r.headline_value, i, s=46, color=colour, marker=marker,
                   zorder=3, linewidths=0)
        base = r.citekey == "diosdado2024lunghist700"
        ax.annotate(f"{r.headline_value:.3f}" + ("  \u2190 comparable baseline"
                                                 if base else ""),
                    (r.headline_value, i), xytext=(7, 0),
                    textcoords="offset points", va="center",
                    fontsize=7.5, fontweight="bold" if base else "normal",
                    color="#111827" if base else "#374151")

    def _first(study: str) -> str:
        head = study.split("(")[0]
        return head.split("&")[0].split(",")[0].replace("et al.", "").strip()

    labels = [f"{_first(s)} et al. ({y})" for s, y in zip(d.study, d.year)]
    y = len(d)
    if ours is not None:
        m, lo, hi = ours
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], color="#111827", lw=1.6,
                    capsize=4, zorder=4)
        ax.scatter(m, y, s=80, color="#111827", marker="*", zorder=5)
        ax.annotate(f"{m:.3f}", (m, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, fontweight="bold")
        labels.append("this work (patient-disjoint)")
        ax.axhline(y - 0.5, color="#9CA3AF", lw=0.7, ls="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.6, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("reported three-class performance")
    ax.grid(axis="y", visible=False)

    seen, handles = [], []
    for k in ("patient", "mixed", "image", "unstated", "external"):
        if (d.split_class == k).any():
            c, mk, lab = SPLIT_STYLE[k]
            handles.append(plt.Line2D([], [], color=c, marker=mk, ls="",
                                      markersize=6, label=lab))
            seen.append(k)
    leg = ax.legend(handles=handles, fontsize=7.5, loc="upper left",
                    bbox_to_anchor=(0.005, 0.62), title="split protocol",
                    title_fontsize=7.5, frameon=True, framealpha=1.0)
    leg.get_frame().set_edgecolor("none")
    leg.set_zorder(6)
    ax.set_title("Reported performance tracks the evaluation protocol",
                 fontsize=10.5, loc="left")
    _save(fig, out, "fig_literature")


def fig_attribution(tables: dict[str, pd.DataFrame], out: Path) -> None:
    """Deletion curves in attention order against a random tile ordering.

    A map is faithful only if removing the tiles it ranks highest degrades the
    prediction faster than removing tiles at random; the gap between the two
    areas is the quantity, and a map no better than random would show none.
    """
    tasks = [t for t in ("super3", "full7") if t in tables]
    if not tasks:
        print("  fig_attribution: no data"); return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.2))
    axes = np.atleast_1d(axes)
    titles = {"super3": "3-class diagnosis", "full7": "7-class subtype + grade"}

    for ax, task in zip(axes, tables and tasks):
        d = tables[task]
        for mode, colour in (("attention", PALETTE[0]), ("random", "#9CA3AF")):
            v = d[d["mode"] == mode].auc_deletion.dropna().values
            ax.hist(v, bins=20, range=(0, 1), alpha=0.65, color=colour,
                    label=f"{mode} (mean {v.mean():.3f})")
        gap = d[d["mode"] == "attention"].faithfulness_gap
        ax.set_title(f"{titles[task]}\ngap {gap.mean():.3f}; "
                     f"attention wins on {100*(gap>0).mean():.0f}% of images",
                     fontsize=9)
        ax.set_xlabel("deletion AUC (lower = faster degradation)")
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("images")
    fig.suptitle("Attention identifies tiles the model relies on",
                 fontsize=10.5, y=1.04)
    _save(fig, out, "fig_attribution")
