"""TrueOT: an external benchmark of directly measured off-target editing.

Kota et al. curated 1,903 guide-site pairs from 11 published studies under three
inclusion rules: the experiment was done in living cells, editing was quantified by
directly measuring modification of the target sequence, and a chromosomal position is
available in hg38. A site is positive when measured editing exceeds 0.1%.

That makes TrueOT a measurement of the same estimand this paper calls the confirmatory
target -- what a nuclease does, read out by a targeted assay at a stated floor -- and
it is the reason the benchmark is used here. Our own confirmatory protocol has six
sgRNAs from one laboratory; TrueOT supplies an order of magnitude more, from
laboratories that had no part in the training corpus.

Two exclusions make it genuinely external, and both matter:

`2020_Lazzarotto` is the pilot study this paper trains on. Scoring it would be scoring
training data, the same trap the CRISPRoffT re-publication sets (Results, "The external
database is dominated by a re-publication of the pilot"). All 507 of its rows go.

Seven further sgRNAs appear in TrueOT under other studies but are also in the pilot, so
their spacers have been seen in training even though these particular measurements have
not. Those 59 rows go too, leaving a test on sgRNAs that are unseen as sequences.

What remains is 1,074 rows, 192 of them positive, over 28 sgRNAs and 9 studies; 25 of
those sgRNAs carry both classes and so admit a ranking metric.

Scope is held to the paper's frozen one: substitutions only, at most six mismatches
over the informative positions. Bulge-containing rows are dropped rather than scored,
as everywhere else here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:  # pragma: no cover
    from src import config as C

#: The pilot study, under the name TrueOT gives it.
PILOT_DATASET = "2020_Lazzarotto"

TRUEOT_XLSX = C.DATA / "crispr_trueot" / "Table_S2_TrueOT_v1_1.xlsx"


def load_trueot(drop_pilot_guides: bool = True) -> pd.DataFrame:
    """Harmonise TrueOT into the column names the rest of this project uses.

    Returns one row per measured guide-site pair with `guide_sequence` (20-nt spacer),
    `site_sequence` (23-nt protospacer plus PAM), `y` (1 if editing exceeded 0.1%),
    `guide_sample`, `study_id`, `cell_type`, `n_mismatch`.
    """
    if not TRUEOT_XLSX.exists():
        raise FileNotFoundError(
            f"{TRUEOT_XLSX} is absent; see datasets/MANIFEST.md for the download")
    d = pd.read_excel(TRUEOT_XLSX, sheet_name=0, header=2)
    d = d.dropna(subset=["Dataset"]).copy()
    for c in ("TrueOT", "#MM", "#Bulge"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["guide_sequence"] = d["Wildtype Sequence"].astype(str).str.upper().str[:20]
    d["site_sequence"] = d["OT sequence w/o bulge info"].astype(str).str.upper()
    d["study_id"] = "TrueOT:" + d["Dataset"].astype(str)
    d["cell_type"] = d["Cell type"].astype(str)
    d["y"] = d["TrueOT"].astype(np.float32)
    d["n_mismatch"] = d["#MM"]

    # frozen scope: substitutions only, within the enumerated mismatch budget, and a
    # site of exactly the 23-nt pattern length (two rows are 22 and 24 nt, which means
    # an alignment this analysis does not model)
    keep = (d["#Bulge"].eq(0)
            & d["#MM"].le(C.MAX_MM_PRIMARY)
            & d.site_sequence.str.len().eq(C.SITE_LEN)
            & d.guide_sequence.str.len().eq(20))
    d = d[keep].copy()

    # the pilot study is training data
    d = d[d.Dataset.ne(PILOT_DATASET)].copy()

    if drop_pilot_guides:
        pairs = pd.read_parquet(C.PAIRS, columns=["guide_sequence", "source_set"])
        pilot = set(pairs[pairs.source_set.eq("pilot")].guide_sequence.str.upper())
        d = d[~d.guide_sequence.isin(pilot)].copy()

    d["guide_sample"] = d.guide_sequence
    cols = ["guide_sample", "guide_sequence", "site_sequence", "y", "study_id",
            "cell_type", "n_mismatch"]
    return d[cols].reset_index(drop=True)


def evaluable(d: pd.DataFrame) -> pd.DataFrame:
    """Rows belonging to sgRNAs that carry both classes, so a ranking is defined."""
    g = d.groupby("guide_sample").y.agg(["size", "sum"])
    ok = g[(g["sum"] > 0) & (g["sum"] < g["size"])].index
    return d[d.guide_sample.isin(ok)].copy()
