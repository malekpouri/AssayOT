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
    "I_constant_prop": "-- with a constant propensity",
    "C1_supervised_anchor": "C1 supervised $+$ confirmatory anchor (control)",
    "C2_anchor_unweighted": "-- with an unweighted confirmatory anchor",
    "R1_cnn_bilstm": "R1 CNN\\,$+$\\,BiLSTM (CRISPR-Net family)",
    "R2_gbt_elevation": "R2 boosted substitution indicators (Elevation family)",
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
    # The journal-neutral class and elsarticle have different text widths, so a table
    # that fits one can overflow the other. Wrapping the tabular in adjustbox shrinks
    # it only when it would not otherwise fit, which keeps both builds clean without
    # hand-tuning a font size per class.
    body = body.replace(r"\begin{tabular}",
                        r"\begin{adjustbox}{max width=\linewidth}" "\n" r"\begin{tabular}")
    body = body.replace(r"\end{tabular}",
                        r"\end{tabular}" "\n" r"\end{adjustbox}")
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
    # three value-plus-interval columns do not fit the text block at \small
    size = r"\footnotesize" if len(cols) > 2 else r"\small"
    body = [
        r"\begin{table}[htbp]\centering" + size,
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
    # fold-matched file if present, else fall back
    p = C.RESULTS / "metrics_guide_ablation.csv"
    if not p.exists():
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
        r"95\% bootstrap intervals over held-out sgRNAs. Every row is computed on the "
        r"same held-out fold, so the configurations are compared on identical sgRNAs; "
        r"the values therefore differ from Table~\ref{tab:guide}, which averages the "
        r"full-model rows over all five folds.}",
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


def table_reference() -> None:
    """The paper's models beside the reimplemented published architectures.

    Both protocols in one table, because the point is that the ordering differs between
    them and a reader should not have to hold two tables in mind to see it.
    """
    rows = []
    for tag, label in (("guide", "reports"), ("validated", "measured activity")):
        own = C.RESULTS / f"metrics_{tag}.csv"
        ref = C.RESULTS / f"metrics_{tag}_reference.csv"
        if not (own.exists() and ref.exists()):
            return
        d = pd.concat([pd.read_csv(own), pd.read_csv(ref)])
        d = d[d.metric.eq("ap") & ~d.model.astype(str).str.startswith("A_")
              & ~d.model.astype(str).str.contains(r"\[report\]")]
        rows.append((label, d.set_index("model")))
    models = ["C1_supervised_anchor", "L3_assay", "L1_supervised", "L2_nnpu",
              "R1_cnn_bilstm", "R2_gbt_elevation", "baseline_changeseq_reads",
              "baseline_cfd"]
    body = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Macro average precision against the reimplemented published "
        r"architectures, on both estimands. R1 is a convolutional-recurrent network of "
        r"the CRISPR-Net family, R2 gradient boosting over position-by-substitution "
        r"indicators of the Elevation family; both are fitted on these folds with the "
        r"labelling convention they are published with, so the comparison does not "
        r"confound another training corpus. Intervals are guide-level bootstraps; the "
        r"left column has 57 held-out sgRNAs and the right has 6.}",
        r"\label{tab:reference}",
        r"\begin{tabular}{@{}lrr@{}}\toprule",
        r"Model & AP, reports & AP, measured activity \\\midrule",
    ]
    for m in models:
        cells = []
        for _, d in rows:
            cells.append(_ci(d.loc[m]) if m in d.index else "--")
        body.append(f"{pretty(m)} & " + " & ".join(cells) + r" \\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    _write("table_reference.tex", "\n".join(body))


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
    table_reference()
    table_metrics("trueot", metrics=("ap", "auroc"), caption=(
        "External measured-activity benchmark (TrueOT): editing quantified directly in "
        "cells across 9 published studies, positive above the same 0.1\\% floor used "
        "here, macro-averaged over the 25 evaluable sgRNAs with 95\\% bootstrap "
        "intervals over sgRNAs. No TrueOT row is trained on: the pilot study's own 507 "
        "rows and a further 59 rows carrying pilot spacers are excluded, so every "
        "sgRNA scored here is unseen as a sequence."))
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
