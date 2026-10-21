"""Figures.

Every figure here has a companion table in the paper carrying the same numbers, which
is both good practice for a print venue and the relief required for the one palette
slot that sits below 3:1 contrast on a white surface.

Conventions: one measure per axis and never two y-scales; a single hue when the chart
compares models on one metric, with the proposed model emphasised by weight rather than
by a different colour; categorical hues assigned in fixed order and never cycled; axis
and grid recessive; text in ink tokens rather than in a series colour.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from src import config as C

FIG = C.PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-3 (all-pairs clean in light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8982"
GRID = "#e4e3de"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
})


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def _placeholder(name: str, reason: str) -> None:
    """Emit a visibly empty figure when its data has not been produced yet.

    The manuscript references every result figure, and `includegraphics` on a missing
    file is a hard error, so a partial run would otherwise fail to build at all. A
    placeholder keeps the document compiling and makes the gap obvious in the PDF
    rather than silent.
    """
    fig, ax = plt.subplots(figsize=(4.0, 1.4))
    ax.axis("off")
    ax.text(0.5, 0.5, f"[{name} not generated]\n{reason}", ha="center", va="center",
            color=MUTED, fontsize=8)
    _save(fig, name)


# ---------------------------------------------------------------------------

def fig_floor_sensitivity():
    """The central quantity as a function of where the detection floor is placed."""
    p = C.PROC / "floor_sensitivity.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    x = d.floor * 100
    # a zero floor cannot sit on a log axis; plot it as the left-most tick
    xs = np.where(x <= 0, 5e-4, x)
    for col, colour in (("both_rate", BLUE), ("gs_neg_rate", ORANGE)):
        ax.plot(xs, d[col] * 100, "-o", color=colour, lw=1.3, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.7)
    # Direct labels instead of a legend box: with two series this carries identity
    # without colour, and keeps the reader's eye on the curves.
    ax.annotate("detected by\nboth assays", (xs[3], d.both_rate.iloc[3] * 100),
                textcoords="offset points", xytext=(-10, -6), color=INK2, fontsize=7.5,
                ha="right", va="top")
    ax.annotate("cellular assay\nreported nothing", (xs[1], d.gs_neg_rate.iloc[1] * 100),
                textcoords="offset points", xytext=(8, 8), color=INK2, fontsize=7.5,
                va="bottom")
    ax.axvline(0.1, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.annotate("assay's published\nsensitivity (0.1%)", (0.1, 88),
                textcoords="offset points", xytext=(6, 0), color=MUTED, fontsize=7,
                ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("activity floor (% editing above matched control)")
    ax.set_ylabel("sites called active (%)")
    ax.set_ylim(-5, 108)
    ax.set_xlim(3e-4, 2.2)
    ax.set_xticks([5e-4, 1e-2, 1e-1, 1e0])
    ax.set_xticklabels(["0", "0.01", "0.1", "1"])
    _save(fig, "fig2_floor_sensitivity")


def _forest(ax, d, metric, title, highlight=("L3_assay",)):
    d = d[d.metric == metric].dropna(subset=["value"]).copy()
    if not len(d):
        return
    d = d.sort_values("value")
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        hot = any(h in str(r.model) for h in highlight)
        col = BLUE if hot else MUTED
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.6 if hot else 1.0,
                solid_capstyle="round", alpha=1.0 if hot else 0.8)
        ax.plot([r.value], [i], "o", color=col, ms=5.5 if hot else 4.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([str(m).replace("_", " ") for m in d.model])
    ax.set_xlabel(metric)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", visible=False)


def fig_protocol(tag: str, metric: str = "ap", name: str | None = None):
    p = C.RESULTS / f"metrics_{tag}.csv"
    out = name or f"fig_{tag}_{metric}"
    if not p.exists():
        return _placeholder(out, f"run `src.run {tag}`")
    d = pd.read_csv(p)
    n = d[d.metric == metric].model.nunique()
    fig, ax = plt.subplots(figsize=(4.6, 0.3 * max(4, n) + 1.0))
    _forest(ax, d, metric, f"{tag}: macro {metric} with 95% guide bootstrap CI")
    _save(fig, out)


def fig_identifiability():
    p = C.RESULTS / "identifiability.csv"
    if not p.exists():
        return _placeholder("fig5_identifiability", "run `src.run identify`")
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.scatter(d.report_loglik, d.activity_sd_across_seeds, s=55, color=BLUE,
               edgecolor="white", linewidth=0.8, zorder=3)
    for r in d.itertuples():
        ax.annotate(r.config, (r.report_loglik, r.activity_sd_across_seeds),
                    textcoords="offset points", xytext=(7, 4), fontsize=7.5,
                    color=INK2)
    ax.set_xlabel("log-likelihood of the observed reports (higher = better fit)")
    ax.set_ylabel("s.d. of recovered activity\nacross initialisations")
    ax.set_title("Equal fit, different answers = not identified", loc="left")
    _save(fig, "fig5_identifiability")


def fig_semisynth():
    p = C.RESULTS / "metrics_semisynth.csv"
    if not p.exists():
        return _placeholder("fig6_semisynth", "run `src.run semisynth`")
    d = pd.read_csv(p)
    d = d[d.metric == "ap"]
    if not len(d):
        return
    mechs = list(dict.fromkeys(d.mechanism))
    models = list(dict.fromkeys(d.model))
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    width = 0.34
    for j, m in enumerate(mechs):
        sub = d[d.mechanism == m].set_index("model").reindex(models)
        x = np.arange(len(models)) + (j - 0.5) * width
        colour = BLUE if j == 0 else ORANGE
        ax.errorbar(x, sub.value, yerr=[sub.value - sub.lo, sub.hi - sub.value],
                    fmt="o", color=colour, ms=5, lw=1.2, capsize=2.5,
                    markeredgecolor="white", markeredgewidth=0.7,
                    label=m.replace("_", " "))
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models])
    ax.set_ylabel("average precision for hidden positives")
    ax.set_title("Recovery of masked positives by masking mechanism", loc="left")
    ax.legend(loc="best")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig6_semisynth")


def fig_calibration():
    p = C.RESULTS / "calibration_validated.csv"
    if not p.exists():
        return _placeholder("fig7_calibration", "run `src.run validated`")
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)),
            label="perfect calibration")
    for colour, (_, r) in zip([BLUE, ORANGE, AQUA], d.iterrows()):
        ax.plot([r.mean_pred], [r.base_rate], "o", color=colour, ms=7,
                markeredgecolor="white", markeredgewidth=0.8,
                label=str(r.model).replace("_", " "))
    ax.set_xlabel("mean predicted activity")
    ax.set_ylabel("observed active fraction")
    ax.set_title("Calibration on the confirmatory assay", loc="left")
    ax.legend(loc="best")
    _save(fig, "fig7_calibration")


def fig_riskcoverage():
    """Error of the accepted set against how much is accepted.

    Both axes matter: an abstention rule that looks good only because it accepts very
    little is not an improvement, so coverage is on the x-axis rather than hidden.
    """
    p = C.RESULTS / "riskcoverage_validated.csv"
    if not p.exists():
        return _placeholder("fig8_risk_coverage", "run `src.run validated`")
    d = pd.read_csv(p)
    keep = [m for m in ("L1_supervised", "L2_nnpu", "L3_assay") if m in set(d.model)]
    if not keep:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    for colour, m in zip([BLUE, ORANGE, AQUA], keep):
        sub = d[d.model == m].sort_values("coverage")
        ax.plot(sub.coverage * 100, sub.brier, "-o", color=colour, lw=1.3, ms=3.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label=m.replace("_", " "))
        last = sub.iloc[-1]
        ax.annotate(m.replace("_", " "), (last.coverage * 100, last.brier),
                    textcoords="offset points", xytext=(-6, 7), color=INK2,
                    fontsize=7.5, ha="right")
    ax.set_xlabel("coverage: sites accepted (%)")
    ax.set_ylabel("Brier score on the accepted set")
    ax.set_title("Selective prediction on the confirmatory assay", loc="left")
    _save(fig, "fig8_risk_coverage")


def fig_candidate_space():
    if not C.CANDIDATES.exists():
        return
    # Both axes must count mismatches the same way. The candidate table's `n_mm_total`
    # is over the 22 informative positions (spacer + the two G's); the pair table
    # stores the spacer and PAM counts separately, so they are added here. Plotting
    # `n_mismatch` alone against `n_mm_total` compares two different quantities and
    # makes detections appear to exceed the candidate space that contains them.
    cand = pd.read_parquet(C.CANDIDATES, columns=["n_mm_total", "pam_class",
                                                  "source_set"])
    cand = cand[cand.source_set.astype(str) == "pilot"]
    pairs = pd.read_parquet(C.PAIRS, columns=["source_set", "detected_changeseq",
                                              "has_bulge", "n_mismatch",
                                              "n_mismatch_pam"])
    det = pairs[(pairs.source_set == "pilot") & pairs.detected_changeseq
                & ~pairs.has_bulge].copy()
    det["n_mismatch"] = det.n_mismatch + det.n_mismatch_pam.fillna(0)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    bins = np.arange(-0.5, 7.5)
    c = cand.n_mm_total.value_counts().reindex(range(7), fill_value=0)
    dd = det.n_mismatch.value_counts().reindex(range(7), fill_value=0)
    x = np.arange(7)
    ax.bar(x - 0.19, c, width=0.36, color=MUTED, label="enumerated candidates",
            edgecolor="white", linewidth=0.8)
    ax.bar(x + 0.19, dd, width=0.36, color=BLUE, label="detected in vitro",
            edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("substitutions relative to the sgRNA pattern")
    ax.set_ylabel("sites (log scale)")
    ax.set_title("Candidate space and detections", loc="left")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig3_candidate_space")


def main():
    print("figures ->", FIG)
    fig_candidate_space()
    fig_floor_sensitivity()
    for tag in ("guide", "validated", "study", "tech"):
        fig_protocol(tag)
    fig_identifiability()
    fig_semisynth()
    fig_calibration()
    fig_riskcoverage()


if __name__ == "__main__":
    main()
