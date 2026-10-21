"""LaTeX tables, written straight from the result files so the paper cannot drift
from the numbers.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from src import config as C

TAB = C.PAPER / "tables"
TAB.mkdir(parents=True, exist_ok=True)

PRETTY = {
    "L1_supervised": "L1 supervised (unreported = neg.)",
    "L2_nnpu": "L2 nnPU",
    "L3_assay": "L3 assay-aware (proposed)",
    "L3_assay_pilot_only": "L3 assay-aware, single training study",
    "A_no_obs_branch": "-- without the observation branch",
    "A_no_validation": "-- without the confirmatory anchor",
    "A_free_prop": "-- with an unconstrained propensity",
    "A_no_fp": "-- without the false-positive term",
    "baseline_mismatch_count": "mismatch count",
    "baseline_position_weighted": "position-weighted mismatches",
    "baseline_cfd": "CFD",
    "baseline_changeseq_reads": "in-vitro read depth",
}


#: Printed names for the metric columns.
METRIC_LABEL = {"ap": "AP", "auroc": "AUROC", "recall@10": "R@10",
                "recall@50": "R@50", "recall@100": "R@100",
                "precision@10": "P@10", "precision@50": "P@50",
                "precision@100": "P@100"}


def pretty(m: str) -> str:
    m = str(m)
    suffix = ""
    if m.endswith(" [report]"):
        m, suffix = m[:-9], r" \textit{(report)}"
    return PRETTY.get(m, m.replace("_", " ")) + suffix


def _write(name: str, body: str) -> None:
    (TAB / name).write_text(body)
    print(f"  {name}")


def _ci(r) -> str:
    if pd.isna(r.value):
        return "--"
    return f"{r.value:.3f} ({r.lo:.3f}--{r.hi:.3f})"


# ---------------------------------------------------------------------------

def table_data() -> None:
    """Composition of the harmonised benchmark."""
    pairs = pd.read_parquet(C.PAIRS)
    obs = pd.read_parquet(C.OBS)
    pilot = obs[obs.study_id == "Lazzarotto2020"]
    ext = obs[obs.study_id != "Lazzarotto2020"]

    def prow(assay, ctx, note):
        sub = pilot[pilot.assay_id == assay]
        return (assay, ctx, len(sub), sub.pair_id.nunique(), note)

    def erow(tier, ctx, note):
        sub = ext[ext.evidence_tier.eq(tier)]
        return (tier, ctx, len(sub), sub.pair_id.nunique(), note)

    groups = [
        (f"Pilot study (one laboratory, 110 sgRNAs)", [
            prow("CHANGEseq", "in vitro", "cell-free genomic DNA"),
            prow("GUIDEseq", "in cell", "run for 57 of the 110 sgRNAs"),
            prow("TagSeq", "in cell", "confirmatory; selection rule documented"),
        ]),
        (f"External database ({ext.study_id.nunique()} studies, "
         f"{ext.assay_type.nunique()} technologies)", [
            erow("discovery", "mixed", "assay detections, no confirmation"),
            erow("confirmed", "in cell", "targeted assay called it positive"),
            erow("rejected_or_failed", "in cell",
                 "flag cannot separate rejected from failed"),
        ]),
    ]
    label = {"CHANGEseq": "CHANGE-seq", "GUIDEseq": "GUIDE-seq",
             "TagSeq": "targeted tag seq.", "discovery": "discovery",
             "confirmed": "confirmed", "rejected_or_failed": "rejected or failed"}
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Composition of the harmonised benchmark. `Observations' counts rows "
        r"in the observation table; the same genomic pair may be observed by several "
        r"assays, so the columns do not sum. No absent report is recorded as a "
        r"biological negative.}",
        r"\label{tab:data}",
        r"\begin{tabular}{@{}llrrp{0.26\linewidth}@{}}\toprule",
        r"Assay & Context & Observations & Pairs & Note \\\midrule",
    ]
    for i, (title, rows) in enumerate(groups):
        if i:
            body.append(r"\addlinespace")
        body.append(r"\multicolumn{5}{@{}l}{\textit{" + title + r"}} \\")
        for a, c, n, p, note in rows:
            body.append(f"\\quad {label[a]} & {c} & {n:,} & {p:,} & "
                        f"\\raggedright {note} \\tabularnewline")
    body += [r"\midrule",
             f"\\textbf{{Total}} & & {len(obs):,} & {pairs.pair_id.nunique():,} & "
             f"\\raggedright {int(pairs.has_bulge.sum()):,} bulge pairs flagged and "
             r"excluded \tabularnewline",
             r"\bottomrule\end{tabular}\end{table}"]
    _write("table1_data.tex", "\n".join(body))


def table_floor() -> None:
    p = C.PROC / "floor_sensitivity.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{How much cellular activity hides among sites the cellular assay did "
        r"not report, as a function of the activity floor. `Cell assay silent' are the "
        r"sites the cell-free assay detected and the cellular assay did not "
        r"(selection classes A--D); `detected by both' is the comparison group. The "
        r"assay's published sensitivity is 0.1\%.}",
        r"\label{tab:floor}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Activity floor & \multicolumn{2}{c}{Cell assay silent} & "
        r"\multicolumn{2}{c}{Detected by both} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & active/$n$ & rate & active/$n$ & rate \\\midrule",
    ]
    for r in d.itertuples():
        lab = "any significant" if r.floor == 0 else f"{r.floor * 100:g}\\%"
        body.append(f"{lab} & {r.gs_neg_active}/{r.gs_neg_n} & "
                    f"{r.gs_neg_rate * 100:.1f}\\% & {r.both_active}/{r.both_n} & "
                    f"{r.both_rate * 100:.1f}\\% \\\\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    _write("table2_floor.tex", "\n".join(body))


def table_metrics(tag: str, metrics=("ap", "auroc", "recall@100"),
                  caption: str | None = None, name: str | None = None,
                  drop_ablations: bool = True) -> None:
    p = C.RESULTS / f"metrics_{tag}.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.metric.isin(metrics)]
    if drop_ablations:
        d = d[~d.model.astype(str).str.startswith("A_")]
    if not len(d):
        return
    piv = {m: g.set_index("model") for m, g in d.groupby("metric")}
    models = list(dict.fromkeys(d.model))
    cols = [m for m in metrics if m in piv]
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{" + (caption or f"Protocol {tag}: macro averages over held-out "
                        r"groups with 95\% bootstrap intervals over those groups.")
        + r"}",
        rf"\label{{tab:{tag}}}",
        r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}\toprule",
        "Model & " + " & ".join(METRIC_LABEL.get(c, c) for c in cols) + r" \\\midrule",
    ]
    for m in models:
        cells = []
        for c in cols:
            sub = piv[c]
            cells.append(_ci(sub.loc[m]) if m in sub.index else "--")
        body.append(f"{pretty(m)} & " + " & ".join(cells) + r" \\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    _write(name or f"table_{tag}.tex", "\n".join(body))


def table_ablation() -> None:
    p = C.RESULTS / "metrics_guide.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.metric.eq("ap") & ~d.model.astype(str).str.endswith("[report]")]
    d = d[d.model.astype(str).str.startswith(("A_", "L3"))]
    if not len(d):
        return
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Ablations on the unseen-sgRNA protocol, macro average precision with "
        r"95\% bootstrap intervals over held-out sgRNAs.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lr}\toprule",
        r"Configuration & Average precision \\\midrule",
    ]
    for r in d.sort_values("value", ascending=False).itertuples():
        body.append(f"{pretty(r.model)} & {_ci(r)} \\\\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    _write("table_ablation.tex", "\n".join(body))


def table_prior() -> None:
    p = C.RESULTS / "metrics_prior.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.metric.isin(["ap", "auroc"])]
    if not len(d):
        return
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Sensitivity of the PU baseline to the assumed class prior, on one "
        r"unseen-sgRNA fold. The prior is not identifiable from positive-unlabelled "
        r"data without further assumptions, so a PU result quoted at a single value of "
        r"it is quoting a choice.}",
        r"\label{tab:prior}",
        r"\begin{tabular}{@{}lrr@{}}\toprule",
        r"Assumed class prior $\pi$ & Average precision & AUROC \\\midrule",
    ]
    for pi, g in d.groupby("prior"):
        gi = g.set_index("metric")
        ap = _ci(gi.loc["ap"]) if "ap" in gi.index else "--"
        au = _ci(gi.loc["auroc"]) if "auroc" in gi.index else "--"
        body.append(f"{pi:g} & {ap} & {au} \\\\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    _write("table_prior.tex", "\n".join(body))


def table_identify() -> None:
    p = C.RESULTS / "identifiability.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Identifiability. Each configuration is refitted from several random "
        r"initialisations. A configuration that fits the observed reports equally well "
        r"but recovers materially different activity probabilities is not identified, "
        r"and its point estimates should not be read as cleavage probabilities.}",
        r"\label{tab:identify}",
        r"\begin{tabular}{lrrr}\toprule",
        r"Configuration & Report log-lik. & s.d.\ of activity & s.d.\ of report \\"
        r"\midrule",
    ]
    for r in d.itertuples():
        body.append(f"{r.config} & {r.report_loglik:.4f} & "
                    f"{r.activity_sd_across_seeds:.4f} & "
                    f"{r.report_sd_across_seeds:.4f} \\\\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    _write("table_identify.tex", "\n".join(body))


def table_semisynth() -> None:
    p = C.RESULTS / "metrics_semisynth.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.metric.eq("ap")]
    if not len(d):
        return
    mechs = list(dict.fromkeys(d.mechanism))
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Semi-synthetic masking: average precision for recovering positives "
        r"hidden either completely at random or preferentially where the in-vitro "
        r"signal is weak. The second mechanism is the one a detection limit actually "
        r"produces.}",
        r"\label{tab:semisynth}",
        r"\begin{tabular}{l" + "r" * len(mechs) + r"}\toprule",
        "Model & " + " & ".join(m.replace("_", " ") for m in mechs) + r" \\\midrule",
    ]
    for m in dict.fromkeys(d.model):
        cells = []
        for mech in mechs:
            sub = d[(d.model == m) & (d.mechanism == mech)]
            cells.append(_ci(sub.iloc[0]) if len(sub) else "--")
        body.append(f"{pretty(m)} & " + " & ".join(cells) + r" \\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    _write("table_semisynth.tex", "\n".join(body))


def main() -> None:
    print("tables ->", TAB)
    table_data()
    table_floor()
    table_metrics("guide", caption=(
        "Unseen-sgRNA protocol: recovery of the sites an independent cellular assay "
        "reported, across the enumerated candidate space, macro-averaged over held-out "
        "sgRNAs with 95\\% bootstrap intervals over sgRNAs. Rows marked "
        "\\textit{(report)} score the probability of a positive report rather than the "
        "activity probability. Precision-based quantities are lower bounds here, "
        "because unreported candidates are not confirmed negatives."))
    table_metrics("validated", metrics=("ap", "auroc"), caption=(
        "Confirmatory-assay protocol, leave-one-sgRNA-out over the six sgRNAs with "
        "targeted validation. Conclusions are restricted to this subpopulation: sites "
        "were selected for confirmation by a documented rule, not sampled at random."))
    table_metrics("study", metrics=("ap", "auroc"), caption=(
        "Unseen-study protocol on the external labelled rows."))
    table_metrics("tech", metrics=("ap", "auroc"), caption=(
        "Unseen-technology protocol on the external labelled rows."))
    table_prior()
    table_ablation()
    table_identify()
    table_semisynth()


if __name__ == "__main__":
    main()
