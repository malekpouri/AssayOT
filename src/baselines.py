"""Reference scorers, so that any gain from the proposed model can be attributed to
the formulation rather than to having more parameters (roadmap section 11).

`cfd` is the published Cutting Frequency Determination score of Doench et al. 2016.
The two lookup tables are shipped as JSON rather than as the pickles they circulate in,
and are checked against the published values on load.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .encode import BASES, SPACER_LEN, encode_seqs, mismatch_matrix

CFD_DIR = C.DATA / "cfd"
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A", "N": "N"}


def load_cfd() -> tuple[dict, dict]:
    mm = json.loads((CFD_DIR / "cfd_mismatch_scores.json").read_text())
    pam = json.loads((CFD_DIR / "cfd_pam_scores.json").read_text())
    # published anchors; if these drift the tables are not the ones we think they are
    assert len(mm) == 240 and len(pam) == 16, (len(mm), len(pam))
    assert abs(pam["GG"] - 1.0) < 1e-9
    assert abs(pam["AG"] - 0.259259259) < 1e-6
    assert abs(pam["TT"]) < 1e-12
    return mm, pam


def cfd_score(guide_seqs, site_seqs) -> np.ndarray:
    """CFD for 20-nt guides against 23-nt sites.

    Follows the reference implementation: the guide is read as RNA, the site base is
    complemented to give the DNA base paired to it, and the per-position penalties
    multiply, times a PAM factor taken from the final two site positions. A mismatch
    pair absent from the table contributes a factor of 1, as in the original code.
    """
    pen, pam = _cfd_tables()
    g = np.clip(encode_seqs(guide_seqs, SPACER_LEN).astype(np.int64), 0, 4)
    s = np.clip(encode_seqs(site_seqs, 23).astype(np.int64), 0, 4)
    j = np.arange(SPACER_LEN)[None, :]
    out = pen[g, s[:, :SPACER_LEN], j].prod(axis=1) * pam[s[:, 21], s[:, 22]]
    # a site with no usable sequence carries no score
    out[(s[:, :SPACER_LEN] >= 4).all(axis=1)] = np.nan
    return out


#: RNA base order aligned to the encoder's ACGT codes: the guide is read as RNA, so a
#: T at a guide position is U in the published lookup key.
_RNA = ("A", "C", "G", "U")


def _cfd_tables():
    """Compile the CFD lookups into arrays indexed by base codes.

    `pen[g, s, j]` is the penalty for guide base `g` against site base `s` at spacer
    position `j`; a pair absent from the published table contributes 1, as in the
    reference implementation, and a match is never penalised. `pam[p1, p2]` is the PAM
    factor, defaulting to 0 for a PAM the table does not list. Compiling once turns a
    per-row Python loop over 20 dictionary lookups into one array gather, which matters
    because the baseline is scored over millions of candidate rows.
    """
    mm_s, pam_s = load_cfd()
    pen = np.ones((5, 5, SPACER_LEN), dtype=np.float64)
    for gi, gb in enumerate(_RNA):
        for si, sb in enumerate(BASES):
            if gb == sb.replace("T", "U"):
                continue
            for jj in range(SPACER_LEN):
                pen[gi, si, jj] = mm_s.get(f"r{gb}:d{_COMP[sb]},{jj + 1}", 1.0)
    pam = np.zeros((5, 5), dtype=np.float64)
    for i, b1 in enumerate(BASES):
        for k, b2 in enumerate(BASES):
            pam[i, k] = pam_s.get(b1 + b2, 0.0)
    return pen, pam


def mismatch_count_score(guide_seqs, site_seqs) -> np.ndarray:
    """Fewer mismatches ranks higher. The simplest defensible ordering."""
    g = encode_seqs(guide_seqs, SPACER_LEN)
    s = encode_seqs(site_seqs, 23)
    return -mismatch_matrix(g, s).sum(1).astype(np.float64)


def position_weighted_score(guide_seqs, site_seqs) -> np.ndarray:
    """Mismatches weighted linearly by proximity to the PAM."""
    g = encode_seqs(guide_seqs, SPACER_LEN)
    s = encode_seqs(site_seqs, 23)
    mm = mismatch_matrix(g, s)
    w = np.linspace(0.2, 1.0, SPACER_LEN)
    return -(mm * w[None, :]).sum(1)


def changeseq_reads_score(df: pd.DataFrame, obs: pd.DataFrame) -> np.ndarray:
    """In-vitro read depth as a direct predictor of cellular activity.

    This is the baseline a practitioner actually uses: rank by the CHANGE-seq read
    count. It is the thing a cross-context model has to beat, and because it needs an
    assay to have been run it is only available for already-measured sites.
    """
    reads = (obs[obs.assay_id == "CHANGEseq"]
             .groupby("pair_id")["signal_value"].max())
    return df.pair_id.map(reads).fillna(0.0).to_numpy(dtype=np.float64)


ALL = {
    "mismatch_count": mismatch_count_score,
    "position_weighted": position_weighted_score,
    "cfd": cfd_score,
}


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, fn in ALL.items():
        out[f"score_{name}"] = fn(df.guide_sequence.tolist(),
                                  df.site_sequence.tolist())
    return out
