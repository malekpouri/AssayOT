"""Sequence-pair encoding and pre-assay tabular features.

Encoding happens on the fly rather than in a cached matrix: the candidate space runs to
millions of rows, and a 23-character string converts to codes with one `frombuffer`.

Only quantities knowable *before* an experiment is run may enter the activity branch
(roadmap section 5, rule 3). Read depth, assay identity and anything else produced by
running an assay belongs to the observation branch, and the split is enforced here by
keeping the two feature builders separate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASES = "ACGT"
#: 0-3 = ACGT, 4 = N/other, 5 = gap
CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    CODE[ord(_b)] = _i
    CODE[ord(_b.lower())] = _i
CODE[ord("-")] = 5
N_BASE = 6

SPACER_LEN = 20
SITE_LEN = 23
PAM_WILDCARD_IDX = 20


def encode_seqs(seqs, length: int) -> np.ndarray:
    """(n,) strings -> (n, length) uint8 base codes, right-padded with N.

    Fully vectorised: a fixed-width bytes array pads short strings with NUL, which
    `CODE` already maps to the N code, so the whole batch converts in one view. This is
    on the hot path for both training and the baselines, where it runs over millions of
    rows per epoch.
    """
    s = pd.Series(seqs, copy=False)
    # A missing value must not be stringified: `nan` would become the bases N, A, N
    # because 'a' is a valid base character. Replace it with an empty string, which the
    # NUL padding then maps to all-N. The check is vectorised, so the common all-valid
    # case costs nothing.
    if s.isna().any():
        s = s.fillna("")
    arr = np.asarray(s.to_numpy(), dtype=f"S{length}")
    raw = arr.view(np.uint8).reshape(len(arr), length)
    return CODE[raw]


def pair_tokens(guide: np.ndarray, site: np.ndarray) -> np.ndarray:
    """Per-position joint token of (guide base, site base).

    The pair is the unit the nuclease sees: a G:T mismatch at position 3 is a different
    object from a C:A mismatch at position 3, and a joint token lets the encoder learn
    that directly instead of having to recombine two one-hot channels.

    The guide is 20 nt and the site 23 nt; the three PAM positions have no guide
    counterpart and are given the gap code.
    """
    n = site.shape[0]
    g = np.full((n, SITE_LEN), 5, dtype=np.uint8)
    g[:, :SPACER_LEN] = guide[:, :SPACER_LEN]
    return (g.astype(np.int16) * N_BASE + site.astype(np.int16)).astype(np.int16)


N_PAIR_TOKEN = N_BASE * N_BASE


def mismatch_matrix(guide: np.ndarray, site: np.ndarray) -> np.ndarray:
    """(n, 20) bool: a protospacer position differs between guide and site."""
    return guide[:, :SPACER_LEN] != site[:, :SPACER_LEN]


def pam_class_codes(site: np.ndarray) -> np.ndarray:
    """NGG=0, NAG=1, NGA=2, other=3, using the two informative PAM positions."""
    g, a = BASES.index("G"), BASES.index("A")
    p1, p2 = site[:, 21], site[:, 22]
    out = np.full(site.shape[0], 3, dtype=np.int64)
    out[(p1 == g) & (p2 == g)] = 0
    out[(p1 == a) & (p2 == g)] = 1
    out[(p1 == g) & (p2 == a)] = 2
    return out


#: PAM-proximal positions. Position 19 of the spacer abuts the PAM, so the
#: "seed" is the last 8 spacer positions in this 0-based layout.
SEED_SLICE = slice(12, 20)


def tabular_pre_assay(guide: np.ndarray, site: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Features available before any assay is run.

    Deliberately small and interpretable: these double as the feature set for the
    mismatch-count baseline and as the only sequence information the constrained
    propensity branch is allowed to see.
    """
    mm = mismatch_matrix(guide, site)
    n = site.shape[0]
    n_mm = mm.sum(1).astype(np.float32)
    n_seed = mm[:, SEED_SLICE].sum(1).astype(np.float32)
    # position of the mismatch nearest the PAM (20 when there is none)
    idx = np.where(mm, np.arange(SPACER_LEN)[None, :], -1)
    nearest = np.where(n_mm > 0, SPACER_LEN - 1 - idx.max(1), SPACER_LEN).astype(np.float32)
    furthest = np.where(n_mm > 0,
                        SPACER_LEN - 1 - np.where(mm, np.arange(SPACER_LEN)[None, :],
                                                  SPACER_LEN).min(1),
                        SPACER_LEN).astype(np.float32)
    gc = np.isin(site[:, :SPACER_LEN], [BASES.index("G"), BASES.index("C")]
                 ).sum(1).astype(np.float32)
    pam = pam_class_codes(site)
    pam_oh = np.zeros((n, 4), dtype=np.float32)
    pam_oh[np.arange(n), pam] = 1.0
    cols = np.column_stack([
        n_mm, n_seed, nearest, furthest, gc,
        mm[:, :4].sum(1).astype(np.float32),    # PAM-distal quarter
        mm[:, 4:12].sum(1).astype(np.float32),  # middle
        pam_oh,
    ]).astype(np.float32)
    names = ["n_mm", "n_mm_seed", "nearest_mm_to_pam", "furthest_mm_to_pam",
             "gc_spacer", "n_mm_distal", "n_mm_mid",
             "pam_NGG", "pam_NAG", "pam_NGA", "pam_other"]
    assert cols.shape[1] == len(names)
    return cols, names


def encode_pairs(guide_seqs, site_seqs) -> dict[str, np.ndarray]:
    """Full encoding bundle for a batch of (guide, site) sequence pairs."""
    g = encode_seqs(guide_seqs, SPACER_LEN)
    s = encode_seqs(site_seqs, SITE_LEN)
    tab, names = tabular_pre_assay(g, s)
    return dict(guide=g, site=s, tokens=pair_tokens(g, s),
                mismatch=mismatch_matrix(g, s), tabular=tab, tabular_names=names)
