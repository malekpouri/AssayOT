"""Harmonise CHANGE-seq supplementary tables and CRISPRoffT into a
provenance-aware two-table benchmark.

Table 1 (`pairs_v1.parquet`)        one row per biological entity: guide x genomic site.
Table 2 (`observations_v1.parquet`) one row per *observation* of a pair in one assay.
Table 3 (`validation_v1.parquet`)   targeted-tag-seq anchor with explicit activity calls.

The point of the split is that the same pair seen by CHANGE-seq and by GUIDE-seq is two
correlated observations of (possibly different) latent events, not two independent
samples (roadmap section 5, rule 1). Nothing here converts "not reported" into a
biological negative.

Run:  python -m src.prep
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    from . import config as C
except ImportError:  # executed as a script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from src import config as C

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

COORD_RE = re.compile(r"^(chr[\w.]+):(\d+)-(\d+)(?::([+-]))?$")


def parse_coord(s: str) -> tuple[str, int, int, str | None]:
    """'chr2:203870828-203870850:+' -> ('chr2', 203870828, 203870850, '+'), 1-based."""
    m = COORD_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"unparsable coordinate {s!r}")
    chrom, a, b, strand = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    return chrom, a, b, strand


def pair_key(chrom: str, start: int, end: int, strand: str, guide: str) -> str:
    """Identity of a biological entity: a site in the genome interrogated by one guide."""
    return f"{guide}|{chrom}:{start}-{end}:{strand}"


def mismatch_profile(site: str, target: str) -> tuple[int, int, str]:
    """Compare a 23-nt site against a 23-nt target pattern (N = wildcard).

    Returns (protospacer mismatches, PAM mismatches, per-position pattern string).
    The pattern uses '.' for a match, the site base for a mismatch and '-' where
    either sequence carries a gap, so bulge rows stay visible instead of silently
    shifting the alignment.
    """
    site = str(site).upper()
    target = str(target).upper()
    n = min(len(site), len(target))
    pat = []
    n_mm_proto = n_mm_pam = 0
    for i in range(n):
        s, t = site[i], target[i]
        if s == "-" or t == "-":
            pat.append("-")
            continue
        if t == "N" or s == t:
            pat.append(".")
            continue
        pat.append(s)
        if i < C.SPACER_LEN:
            n_mm_proto += 1
        else:
            n_mm_pam += 1
    return n_mm_proto, n_mm_pam, "".join(pat)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------
# CHANGE-seq supplementary tables
# ---------------------------------------------------------------------------

@dataclass
class ChangeSeqTables:
    guides: pd.DataFrame
    changeseq: pd.DataFrame
    guideseq: pd.DataFrame
    validation: pd.DataFrame
    ontarget: pd.DataFrame


def _sheet(xls: pd.ExcelFile, name: str, header: int) -> pd.DataFrame:
    return xls.parse(name, header=header, dtype=object)


def load_changeseq() -> ChangeSeqTables:
    log(f"\n## CHANGE-seq supplementary tables\n\nSource: `{C.CHANGESEQ_XLSX.name}` "
        "(Lazzarotto et al. 2020, Nat Biotechnol 38:1317, Supplementary Tables 2-8).")
    xls = pd.ExcelFile(C.CHANGESEQ_XLSX)

    # --- Table 3: CHANGE-seq detected sites (in vitro) ----------------------
    cs = _sheet(xls, "CHANGE-seq_Supp_Table_3", 0)
    cs = cs.rename(columns={
        "chrom": "chrom", "chromStart": "start0", "chromEnd": "end1",
        "name": "sample", "CHANGEseq_reads": "reads", "strand": "strand",
        "offtarget_sequence": "site_seq", "distance": "distance",
        "target": "target_pattern",
    })
    cs = cs.dropna(subset=["chrom", "sample"]).copy()
    for c in ("start0", "end1", "reads", "distance"):
        cs[c] = pd.to_numeric(cs[c], errors="coerce")
    cs["start1"] = cs["start0"].astype("Int64") + 1
    cs["span"] = cs["end1"] - cs["start0"]
    # 48 released rows carry coordinates, reads and a mismatch distance but an empty
    # sequence cell; their spans are 24 nt, i.e. they are bulge-containing sites. They
    # are kept and flagged rather than dropped, and the substitution-only primary
    # analysis excludes them like any other bulge row.
    cs["seq_missing"] = cs["site_seq"].isna()
    cs["site_seq"] = cs["site_seq"].fillna("")
    log(f"- Table 3: {len(cs)} CHANGE-seq detected sites "
        f"({cs['sample'].nunique()} sgRNAs), cell-free genomic DNA from human "
        "CD4+/CD8+ T-cells, hg38.")
    log(f"  - {int(cs['seq_missing'].sum())} rows have an empty sequence cell "
        f"(spans {sorted(cs.loc[cs.seq_missing, 'span'].unique().tolist())} nt, i.e. "
        "bulge sites); kept and flagged, excluded from the primary analysis.")
    log(f"  - reported mismatch distance ranges "
        f"{int(cs['distance'].min())}-{int(cs['distance'].max())}; the distance is "
        "counted over the 22 informative positions of the 23-nt pattern, so sites "
        "with a non-NGG PAM are present.")

    # --- the sgRNA table ----------------------------------------------------
    # Built from the sites tables, which cover all 110 sgRNAs, rather than from
    # Table 4, which lists 109 (TRBC1_site_2 is absent there).
    g = (cs[["sample", "target_pattern"]].drop_duplicates("sample")
         .dropna().reset_index(drop=True))
    g["target_pattern"] = g["target_pattern"].str.upper().str.strip()
    g["protospacer"] = g["target_pattern"].str[: C.SPACER_LEN]
    g["locus"] = g["sample"].str.replace(r"_site_?\d+$", "", regex=True)
    assert g["protospacer"].str.len().eq(C.SPACER_LEN).all(), "spacer length"
    assert g["target_pattern"].str.len().eq(C.SITE_LEN).all(), "pattern length"

    t4 = _sheet(xls, "CHANGE-seq_Supp_Table_4", 0)
    t4.columns = ["sample", "protospacer_t4", "specificity_ratio"]
    t4 = t4.dropna(subset=["sample"]).copy()
    t4["specificity_ratio"] = pd.to_numeric(t4["specificity_ratio"], errors="coerce")
    g = g.merge(t4[["sample", "specificity_ratio", "protospacer_t4"]],
                on="sample", how="left")
    mism = g.protospacer_t4.notna() & (
        g.protospacer_t4.str.upper().str.strip() != g.protospacer)
    assert not mism.any(), f"spacer disagreement for {g.loc[mism, 'sample'].tolist()}"
    g = g.drop(columns=["protospacer_t4"])
    log(f"- sgRNA table: {len(g)} sgRNAs over {g['locus'].nunique()} loci; "
        f"specificity ratio available for {int(g.specificity_ratio.notna().sum())} "
        "(Table 4 omits TRBC1_site_2). Spacers agree with Table 4 wherever both exist.")

    # --- Table 6: GUIDE-seq detected sites (in cell) ------------------------
    gs = _sheet(xls, "CHANGE-seq_Supp_Table_6", 0)
    gs = gs.rename(columns={
        "#chrom": "chrom", "chromStart": "start0", "chromEnd": "end1",
        "name": "sample", "GUIDEseq_reads": "reads", "strand": "strand",
        "offtarget_sequence": "site_seq", "genomic_coordinate": "coord",
        "distance": "distance", "target": "target_pattern", "run": "run",
    })
    gs = gs.dropna(subset=["chrom", "sample", "site_seq"]).copy()
    for c in ("start0", "end1", "reads", "distance"):
        gs[c] = pd.to_numeric(gs[c], errors="coerce")
    gs["start1"] = gs["start0"].astype("Int64") + 1
    log(f"- Table 6: {len(gs)} GUIDE-seq detected sites "
        f"({gs['sample'].nunique()} sgRNAs), in cell.")

    # --- Tables 7+8: targeted tag sequencing validation ---------------------
    v7 = _sheet(xls, "CHANGE-seq_Supp_Table_7", 0)
    v8 = _sheet(xls, "CHANGE-seq_Supp_Table_8", 0)
    v7["src_table"] = "SuppTable7"
    v8["src_table"] = "SuppTable8"
    val = pd.concat([v7, v8], ignore_index=True)
    val = val.dropna(subset=["Genomic Coordinate", "Target Name"]).copy()
    log(f"- Tables 7+8: {len(val)} sites assayed by targeted tag sequencing over "
        f"{val['Target Name'].nunique()} sgRNAs; selection classes "
        f"{dict(Counter(val['Class']))}.")

    # --- Table 5: on-target indel rates -------------------------------------
    ot = _sheet(xls, "CHANGE-seq_Supp_Table_5", 1)
    ot = ot.dropna(subset=["Target"]).copy()
    log(f"- Table 5: on-target indel rates for {len(ot)} sites.")

    return ChangeSeqTables(g, cs, gs, val, ot)


# ---------------------------------------------------------------------------
# validation activity calls
# ---------------------------------------------------------------------------

REP = (1, 2, 3)


def _pool(df: pd.DataFrame, prefix_num: str, prefix_den: str) -> tuple[pd.Series, pd.Series]:
    num = sum(pd.to_numeric(df[f"{prefix_num}{i}"], errors="coerce").fillna(0) for i in REP)
    den = sum(pd.to_numeric(df[f"{prefix_den}{i}"], errors="coerce").fillna(0) for i in REP)
    return num, den


def call_validation(val: pd.DataFrame) -> pd.DataFrame:
    """Call cellular activity at each targeted-tag-seq site.

    Two readouts are available per site, each with three treated and three control
    replicates: indel frequency and dsODN tag integration. Replicates are pooled into
    one 2x2 table and tested one-sided against the matched control with Fisher's exact
    test; q-values are BH-corrected over all sites within a readout. A site is called
    active when it is significant *and* clears a 0.1% effect floor, which is the
    sensitivity the assay was published with. Keeping the depth attached means a
    "negative" is a negative at a stated limit of detection, not an absolute one.
    """
    out = val.copy()
    readouts = {
        "indel": ("Indel_RNP_", "Total_RNP_", "Indel_control_", "Total_control_"),
        "tag": ("Integration_RNP_dsODN_", "Total_RNP_dsODN_",
                "Integration_control_dsODN_", "Total_control_dsODN_"),
    }
    for name, (tn, td, cn, cd) in readouts.items():
        t_num, t_den = _pool(out, tn, td)
        c_num, c_den = _pool(out, cn, cd)
        out[f"{name}_treated_count"] = t_num
        out[f"{name}_treated_total"] = t_den
        out[f"{name}_control_count"] = c_num
        out[f"{name}_control_total"] = c_den
        t_frac = np.where(t_den > 0, t_num / t_den.replace(0, np.nan), np.nan)
        c_frac = np.where(c_den > 0, c_num / c_den.replace(0, np.nan), np.nan)
        out[f"{name}_treated_frac"] = t_frac
        out[f"{name}_control_frac"] = c_frac
        out[f"{name}_delta"] = t_frac - c_frac
        p = np.full(len(out), np.nan)
        for i in range(len(out)):
            a, b = t_num.iat[i], t_den.iat[i] - t_num.iat[i]
            c, d = c_num.iat[i], c_den.iat[i] - c_num.iat[i]
            if min(t_den.iat[i], c_den.iat[i]) <= 0 or b < 0 or d < 0:
                continue
            p[i] = fisher_exact([[a, b], [c, d]], alternative="greater")[1]
        out[f"{name}_p"] = p
        out[f"{name}_q"] = bh_qvalues(p)
        out[f"{name}_active"] = (
            (out[f"{name}_q"] < C.VALIDATION_FDR)
            & (out[f"{name}_delta"] >= C.VALIDATION_MIN_DELTA)
        )
        # depth-aware interpretability: a site with too little depth to see the floor
        # is not evidence of absence.
        out[f"{name}_powered"] = t_den >= 1.0 / C.VALIDATION_MIN_DELTA

    out["is_active"] = out["tag_active"] | out["indel_active"]
    out["is_interpretable"] = out["tag_powered"] | out["indel_powered"]
    log(f"\n### Validation calls (FDR {C.VALIDATION_FDR}, effect floor "
        f"{C.VALIDATION_MIN_DELTA:.1%})\n")
    log(f"- active by tag integration: {int(out['tag_active'].sum())}; "
        f"by indel: {int(out['indel_active'].sum())}; "
        f"either: {int(out['is_active'].sum())} of {len(out)}.")
    log(f"- sufficiently powered to see the floor: {int(out['is_interpretable'].sum())}.")
    tab = out.groupby("Class")["is_active"].agg(["size", "sum", "mean"])
    log("\n| selection class | sites | active | active rate |\n|---|---|---|---|")
    for cls, r in tab.iterrows():
        log(f"| {cls} | {int(r['size'])} | {int(r['sum'])} | {r['mean']:.1%} |")

    floor_sensitivity(out)
    return out


def floor_sensitivity(out: pd.DataFrame) -> pd.DataFrame:
    """How much of the headline depends on where the detection floor is put?

    The quantity the project hinges on is: among sites CHANGE-seq detected but
    GUIDE-seq did not report, what fraction is actually active in cells? That number
    is not a property of the data alone, it is a property of the data *and* the
    effect size one is willing to call activity. Reporting it as a curve rather than a
    point is the honest form (roadmap sections 6 and 9).
    """
    gs_neg = out[out["Class"].isin(list("ABCD"))]
    both = out[out["Class"].eq("both")]
    rows = []
    for floor in (0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2):
        def rate(df):
            act = ((df["tag_q"] < C.VALIDATION_FDR) & (df["tag_delta"] >= floor)) | \
                  ((df["indel_q"] < C.VALIDATION_FDR) & (df["indel_delta"] >= floor))
            return act.sum(), len(df), act.mean()
        a1, n1, r1 = rate(gs_neg)
        a2, n2, r2 = rate(both)
        rows.append(dict(floor=floor, gs_neg_active=int(a1), gs_neg_n=n1,
                         gs_neg_rate=r1, both_active=int(a2), both_n=n2,
                         both_rate=r2))
    df = pd.DataFrame(rows)
    log("\n### Sensitivity of the central quantity to the activity floor\n")
    log("`GUIDE-seq-negative` = CHANGE-seq-detected sites in selection classes A-D, "
        "which GUIDE-seq did not report.\n")
    log("| activity floor | GUIDE-seq-negative active | rate | detected-by-both active | rate |")
    log("|---|---|---|---|---|")
    for r in rows:
        log(f"| {r['floor']:.2%} | {r['gs_neg_active']}/{r['gs_neg_n']} | "
            f"{r['gs_neg_rate']:.1%} | {r['both_active']}/{r['both_n']} | "
            f"{r['both_rate']:.1%} |")
    df.to_csv(C.PROC / "floor_sensitivity.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# build the two tables
# ---------------------------------------------------------------------------

def build(tabs: ChangeSeqTables, val_called: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    guides = tabs.guides.set_index("sample")
    pat = guides["target_pattern"].to_dict()
    spacer = guides["protospacer"].to_dict()
    # The released tables are not consistent about sgRNA names: the GUIDE-seq sheet
    # contains `CTLA4site_2`, `LAG_site_3` and two mislabelled CBLB sites that appear
    # nowhere in the detection sheet. Each one carries its target sequence in the same
    # row, and each resolves to exactly one known sgRNA by that sequence, so identity
    # is taken from the sequence and the name is only a fallback. Matching on the name
    # alone silently discards 17 in-cell detections, which are the scarcest rows here.
    by_pattern: dict[str, str] = {}
    for s, p in pat.items():
        by_pattern.setdefault(str(p).upper().strip(), s)

    aliases: dict[str, str] = {}

    def resolve(sample, target_pattern=None):
        if sample in spacer:
            return sample
        if sample in aliases:
            return aliases[sample]
        alias = by_pattern.get(str(target_pattern).upper().strip())
        if alias is not None:
            aliases[sample] = alias
        return alias

    pair_rows: dict[str, dict] = {}
    obs_rows: list[dict] = []

    def register(chrom, start1, end1, strand, sample, site_seq,
                 target_pattern=None) -> str:
        sample = resolve(sample, target_pattern)
        if sample is None:
            return ""
        guide = spacer[sample]
        key = pair_key(chrom, int(start1), int(end1), strand, guide)
        if key not in pair_rows:
            n_proto, n_pam, mmpat = mismatch_profile(site_seq, pat[sample])
            pair_rows[key] = dict(
                pair_id=key, guide_sample=sample, guide_sequence=guide,
                target_pattern=pat[sample], site_sequence=str(site_seq).upper(),
                protospacer_observed=str(site_seq).upper()[: C.SPACER_LEN],
                pam_observed=str(site_seq).upper()[C.SPACER_LEN:],
                chromosome=chrom, start=int(start1), end=int(end1), strand=strand,
                genome_build=C.GENOME_BUILD,
                n_mismatch=n_proto, n_mismatch_pam=n_pam,
                mismatch_pattern=mmpat,
                # A bulge shows up three ways in the released tables and all three must
                # be caught: a gap character in the sequence (DNA bulge), a genomic
                # window wider than 23 nt (RNA bulge, which carries no gap character),
                # or an empty sequence cell. Flagging only the gap character leaves the
                # RNA-bulge rows looking like substitution sites, where a
                # position-by-position comparison against a 23-nt pattern reads out as
                # 7 or more spurious mismatches.
                has_bulge=(len(str(site_seq)) != C.SITE_LEN
                           or "-" in str(site_seq)
                           or (int(end1) - int(start1) + 1) != C.SITE_LEN),
                span=int(end1) - int(start1) + 1,
                locus=guides.at[sample, "locus"],
                specificity_ratio=guides.at[sample, "specificity_ratio"],
            )
        return key

    # --- CHANGE-seq: in vitro, reported positives ---------------------------
    cs = tabs.changeseq
    for r in cs.itertuples(index=False):
        key = register(r.chrom, r.start1, r.end1, r.strand, r.sample, r.site_seq,
                       getattr(r, "target_pattern", None))
        if not key:
            continue
        obs_rows.append(dict(
            pair_id=key, study_id="Lazzarotto2020", assay_id="CHANGEseq",
            assay_type="CHANGE-seq", context=C.CTX_INVITRO,
            cell_type="CD4+/CD8+ T-cell gDNA", donor_id="donor1",
            nuclease=C.NUCLEASE, replicate_id="pooled",
            status=C.STATUS_POSITIVE, evidence_tier="discovery",
            genome_build=C.GENOME_BUILD, signal_value=float(r.reads),
            signal_units="CHANGE-seq reads", coverage=np.nan, qc_status="pass",
            reported_distance=float(r.distance) if pd.notna(r.distance) else np.nan,
        ))

    # --- GUIDE-seq: in cell, reported positives -----------------------------
    gs = tabs.guideseq
    for r in gs.itertuples(index=False):
        key = register(r.chrom, r.start1, r.end1, r.strand, r.sample, r.site_seq,
                       getattr(r, "target_pattern", None))
        if not key:
            continue
        obs_rows.append(dict(
            pair_id=key, study_id="Lazzarotto2020", assay_id="GUIDEseq",
            assay_type="GUIDE-seq", context=C.CTX_INCELL,
            cell_type="CD4+/CD8+ T-cell", donor_id="donor1",
            nuclease=C.NUCLEASE, replicate_id=str(getattr(r, "run", "")),
            status=C.STATUS_POSITIVE, evidence_tier="discovery",
            genome_build=C.GENOME_BUILD, signal_value=float(r.reads),
            signal_units="GUIDE-seq reads", coverage=np.nan, qc_status="pass",
            reported_distance=float(r.distance) if pd.notna(r.distance) else np.nan,
        ))

    if aliases:
        log(f"\n- {len(aliases)} sgRNA names in the GUIDE-seq sheet are absent from "
            "the detection sheet and were resolved by target sequence, each to exactly "
            "one known sgRNA: "
            + ", ".join(f"`{k}` -> `{v}`" for k, v in sorted(aliases.items()))
            + ". Matching on the name alone would silently discard these in-cell "
              "detections.")

    # Which sgRNAs were actually run through GUIDE-seq? Only for those sgRNAs is a
    # non-detection informative at all; for every other sgRNA the site is unobserved.
    guideseq_samples = {s for s in (resolve(x, p) for x, p in
                        zip(gs["sample"], gs.get("target_pattern", gs["sample"])))
                        if s is not None}
    log(f"\n- GUIDE-seq was run for {len(guideseq_samples)} of {len(guides)} sgRNAs; "
        "for the remaining sgRNAs a site is unobserved in cell, not negative.")

    # --- targeted tag sequencing: the anchor --------------------------------
    val_rows = []
    for d in val_called.to_dict("records"):
        name = d["Target Name"]
        chrom, a, b, strand = parse_coord(d["Genomic Coordinate"])
        if strand is None:
            strand = "+"
        key = register(chrom, a, b, strand, name, d["Off-target Sequence"])
        if not key:
            continue
        status = C.STATUS_POSITIVE if d["is_active"] else (
            C.STATUS_TESTED_NEG if d["is_interpretable"] else C.STATUS_UNINTERPRETABLE)
        obs_rows.append(dict(
            pair_id=key, study_id="Lazzarotto2020", assay_id="TagSeq",
            assay_type="targeted tag sequencing", context=C.CTX_INCELL,
            cell_type="CD4+/CD8+ T-cell", donor_id="donor1",
            nuclease=C.NUCLEASE, replicate_id="pooled3",
            status=status,
            evidence_tier=("confirmed" if d["is_active"] else
                           "rejected" if d["is_interpretable"] else "failed"),
            genome_build=C.GENOME_BUILD,
            signal_value=float(d["tag_delta"]) if pd.notna(d["tag_delta"]) else np.nan,
            signal_units="dsODN tag integration fraction above control",
            coverage=float(d["tag_treated_total"]), qc_status="pass",
            reported_distance=np.nan,
        ))
        row = dict(d)
        row["pair_id"] = key
        row["selection_class"] = d["Class"]
        val_rows.append(row)

    pairs = pd.DataFrame(list(pair_rows.values()))
    obs = pd.DataFrame(obs_rows)
    validation = pd.DataFrame(val_rows)

    # cross-assay bookkeeping on the pair table
    pos_cs = set(obs.loc[(obs.assay_id == "CHANGEseq"), "pair_id"])
    pos_gs = set(obs.loc[(obs.assay_id == "GUIDEseq"), "pair_id"])
    pairs["detected_changeseq"] = pairs.pair_id.isin(pos_cs)
    pairs["detected_guideseq"] = pairs.pair_id.isin(pos_gs)
    pairs["guideseq_run"] = pairs.guide_sample.isin(guideseq_samples)
    pairs["in_validation"] = pairs.pair_id.isin(set(validation.pair_id))

    log(f"\n- pairs: {len(pairs)}; observations: {len(obs)}; "
        f"validated pairs: {int(pairs['in_validation'].sum())}.")
    nb = pairs["has_bulge"].fillna(False)
    log(f"- pairs with a bulge (excluded from the primary analysis): {int(nb.sum())} "
        f"= {int((pairs.site_sequence.str.contains('-', na=False)).sum())} with a gap "
        f"character, {int((pairs.span.fillna(C.SITE_LEN) > C.SITE_LEN).sum())} with a "
        f"window wider than {C.SITE_LEN} nt, "
        f"{int((pairs.site_sequence.fillna('') == '').sum())} with an empty sequence.")

    # The key asymmetry the paper is about.
    both = pairs.detected_changeseq & pairs.detected_guideseq
    cs_only = pairs.detected_changeseq & ~pairs.detected_guideseq & pairs.guideseq_run
    gs_only = ~pairs.detected_changeseq & pairs.detected_guideseq
    log(f"- among sgRNAs with matched GUIDE-seq: {int(both.sum())} pairs detected by "
        f"both, {int(cs_only.sum())} by CHANGE-seq only, {int(gs_only.sum())} by "
        "GUIDE-seq only.")
    return pairs, obs, validation


# ---------------------------------------------------------------------------
# CRISPRoffT: external studies
# ---------------------------------------------------------------------------

def load_offt() -> pd.DataFrame:
    log("\n## CRISPRoffT\n\nSource: `allframe_update_addEpige.txt` "
        "(Zhang et al. 2025, Nucleic Acids Res 53:D914).")
    df = pd.read_csv(C.OFFT_ALL, sep="\t", dtype=str, low_memory=False,
                     on_bad_lines="warn")
    n0 = len(df)
    # Two rows in the release are shifted by one field; they are identifiable by a
    # Year that is not a year, and are dropped rather than guessed at.
    bad = ~df["Year"].fillna("").str.fullmatch(r"\d{4}")
    log(f"- {n0} rows; dropping {int(bad.sum())} rows whose fields are shifted "
        "(non-numeric Year).")
    df = df[~bad].copy()
    for c in df.columns:
        df[c] = df[c].replace({"NULL": np.nan, "": np.nan})
    df["n_mismatch"] = pd.to_numeric(df["Mismach"], errors="coerce")
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    for c in ("Indel_control%", "Indel_treatment%", "Indel_accu%"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = (
        df["Species"].eq(C.SPECIES)
        & df["Cas9_type"].eq(C.NUCLEASE)
        & df["Assembly"].isin(["hg38", "hg19"])
    )
    log(f"- scope filter (SpCas9, {C.SPECIES}, hg38 or hg19): "
        f"{int(keep.sum())} of {len(df)} rows retained. hg19 rows are kept because "
        "the model consumes sequence pairs, not coordinates; they carry their build "
        "and are excluded from every coordinate-dependent step (candidate "
        "enumeration, epigenomic features, positional de-duplication).")
    df = df[keep].copy()
    log(f"- builds retained: {dict(Counter(df['Assembly']))}")
    if C.EXCLUDE_BULGES:
        nb = df["Bulge"].isna()
        log(f"- substitution-only filter: dropping {int((~nb).sum())} bulge rows.")
        df = df[nb].copy()

    df["context"] = np.where(
        df["Reaction_location"].fillna("").str.contains("Cell-free"),
        C.CTX_INVITRO, C.CTX_INCELL)
    # `Validation == FALSE` is ambiguous in the source: it can mean "tested and
    # rejected" or "PCR failed" (CRISPRoffT help page). It is therefore never mapped
    # to a training negative; it becomes an explicitly uninterpretable status unless
    # an indel measurement is attached.
    has_measure = df["Indel_treatment%"].notna()
    df["status"] = np.where(
        df["Validation"].eq("TRUE"), C.STATUS_POSITIVE,
        np.where(df["Validation"].eq("FALSE") & has_measure, C.STATUS_TESTED_NEG,
                 np.where(df["Validation"].eq("FALSE"), C.STATUS_UNINTERPRETABLE,
                          C.STATUS_POSITIVE)))
    # Evidence tier is kept separate from status: a site confirmed by a targeted
    # confirmatory assay is stronger evidence than a discovery-assay detection, and
    # the two must not be pooled into one "positive" class.
    df["evidence_tier"] = np.where(
        df["Validation"].eq("TRUE"), "confirmed",
        np.where(df["Validation"].isna(), "discovery", "rejected_or_failed"))
    log(f"- status mapping: {dict(Counter(df['status']))}")
    log(f"- evidence tier: {dict(Counter(df['evidence_tier']))}")
    nfalse = int(df["Validation"].eq("FALSE").sum())
    nmeas = int((df["Validation"].eq("FALSE") & has_measure).sum())
    log(f"- of {nfalse} rows marked `Validation == FALSE`, only {nmeas} carry an indel "
        "measurement. The CRISPRoffT documentation states that `FALSE` can mean either "
        "a rejected off-target or a failed PCR, so the remainder are recorded as "
        "uninterpretable and never used as training negatives.")
    log(f"- technologies: {df['Technology'].nunique()}; studies (PMID): "
        f"{df['PMID'].nunique()}; guides: {df['Guide_sequence'].nunique()}.")
    return df


def offt_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map CRISPRoffT onto the same two-table schema."""
    d = df.copy()
    # CRISPRoffT stores the guide as spacer+PAM (23 nt for SpCas9); the pilot tables
    # store the 20-nt spacer. Normalise to the spacer so the same site reported by
    # both sources collapses to one pair.
    raw = d["Guide_sequence"].str.upper().str.strip()
    d["guide_sequence"] = np.where(raw.str.len() >= C.SPACER_LEN,
                                   raw.str[: C.SPACER_LEN], raw)
    log(f"- guide length before normalisation: "
        f"{dict(Counter(raw.str.len().dropna().astype(int)))}")
    d["site_sequence"] = d["Target_sequence"].fillna(
        d["Protospacer_sequence"].fillna("") + d["PAM"].fillna("")).str.upper()
    d = d.dropna(subset=["chr", "start", "end", "guide_sequence"])
    d["start"] = pd.to_numeric(d["start"], errors="coerce")
    d["end"] = pd.to_numeric(d["end"], errors="coerce")
    d = d.dropna(subset=["start", "end"])
    d["strand"] = d["Strand"].fillna("+")
    d["pair_id"] = [
        pair_key(c, int(s), int(e), st, g)
        for c, s, e, st, g in zip(d["chr"], d["start"], d["end"], d["strand"],
                                  d["guide_sequence"])
    ]
    pairs = (d.groupby("pair_id", as_index=False)
             .agg(guide_sequence=("guide_sequence", "first"),
                  site_sequence=("site_sequence", "first"),
                  chromosome=("chr", "first"), start=("start", "first"),
                  end=("end", "first"), strand=("strand", "first"),
                  n_mismatch=("n_mismatch", "first"),
                  gene=("Gene_OR_Locus", "first"),
                  on_target_site=("On_target_site", "first"),
                  epigenetic=("Epigenetic_markers_combind", "first"),
                  genome_build=("Assembly", "first")))
    obs = pd.DataFrame(dict(
        pair_id=d["pair_id"],
        study_id="PMID" + d["PMID"].astype(str),
        assay_id=d["Technology"].astype(str) + "|PMID" + d["PMID"].astype(str),
        assay_type=d["Technology"],
        context=d["context"],
        cell_type=d["Source"].fillna("unknown"),
        donor_id=np.nan,
        nuclease=d["Cas9_type"],
        replicate_id=np.nan,
        status=d["status"],
        evidence_tier=d["evidence_tier"],
        genome_build=d["Assembly"],
        signal_value=pd.to_numeric(d["Score"], errors="coerce"),
        signal_units="source-specific discovery score",
        indel_treatment_pct=d["Indel_treatment%"],
        indel_control_pct=d["Indel_control%"],
        coverage=np.nan,
        qc_status="pass",
        targeted_method=d["Targeted_method"],
        year=d["year"],
    ))
    log(f"- CRISPRoffT mapped to {len(pairs)} pairs and {len(obs)} observations.")
    return pairs, obs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    log("# Provenance log\n")
    log("Built by `src/prep.py`. Every row records which assay saw a site, in which "
        "context, and what the report means. No step converts an absent report into a "
        "biological negative.")

    tabs = load_changeseq()
    val_called = call_validation(tabs.validation)
    pairs, obs, validation = build(tabs, val_called)

    offt = load_offt()
    offt_pairs, offt_obs = offt_observations(offt)

    # Overlap between the pilot and the external set (roadmap section 4: de-duplicate).
    shared = set(pairs.pair_id) & set(offt_pairs.pair_id)
    log(f"\n- {len(shared)} pairs appear in both the CHANGE-seq pilot and CRISPRoffT "
        "(CRISPRoffT re-publishes the CHANGE-seq release); flagged as "
        "`in_pilot` so external tests can exclude them.")
    offt_pairs["in_pilot"] = offt_pairs.pair_id.isin(shared)
    offt_obs["in_pilot"] = offt_obs.pair_id.isin(shared)

    pairs["source_set"] = "pilot"
    offt_pairs["source_set"] = "external"
    all_pairs = pd.concat([pairs, offt_pairs], ignore_index=True)
    all_pairs = all_pairs.drop_duplicates("pair_id", keep="first")
    all_obs = pd.concat([obs, offt_obs], ignore_index=True)

    # Concatenating frames with different columns leaves boolean flags as `object`
    # holding None. `~` on an object Series is a bitwise not, so `~True` is -2 and
    # `~False` is -1 -- both truthy -- which silently turns any "drop the bulges" mask
    # into a no-op. Coerce the flags to real booleans here so that cannot happen
    # downstream.
    for col in ("has_bulge", "detected_changeseq", "detected_guideseq",
                "guideseq_run", "in_validation", "in_pilot"):
        if col in all_pairs:
            all_pairs[col] = all_pairs[col].fillna(False).astype(bool)
    for col in ("in_pilot",):
        if col in all_obs:
            all_obs[col] = all_obs[col].fillna(False).astype(bool)

    # A sequence model cannot tell two genomic loci with an identical (guide, site)
    # 23-mer apart, so the sequence pair -- not the locus -- is the unit that must not
    # straddle a train/test boundary. Repeat families make this a real risk.
    all_pairs["seq_key"] = (all_pairs.guide_sequence.fillna("")
                            + "|" + all_pairs.site_sequence.fillna(""))
    n_loci, n_seq = len(all_pairs), all_pairs.seq_key.nunique()
    dup = all_pairs.seq_key.duplicated(keep=False)
    log(f"\n- {n_loci} genomic pairs collapse to {n_seq} distinct (guide, site) "
        f"sequence pairs; {int(dup.sum())} pairs share a sequence with another locus "
        "(repeat families). Splits are made on `seq_key`, never on the locus.")

    all_pairs.to_parquet(C.PAIRS, index=False)
    all_obs.to_parquet(C.OBS, index=False)
    validation.to_parquet(C.VALID, index=False)
    log(f"\n## Written\n\n- `{C.PAIRS.name}`: {len(all_pairs)} pairs")
    log(f"- `{C.OBS.name}`: {len(all_obs)} observations")
    log(f"- `{C.VALID.name}`: {len(validation)} validation rows")
    C.PROVENANCE_LOG.write_text("\n".join(LOG) + "\n")
    print(f"\nprovenance log -> {C.PROVENANCE_LOG}")


if __name__ == "__main__":
    main()
