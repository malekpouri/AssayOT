"""Assert the invariants the paper's claims rest on.

    python -m src.selfcheck

Each check corresponds to a statement made in the manuscript. They run against the
built tables, so a change to the harmonisation that quietly breaks one of them fails
here rather than in a reviewer's reading.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import encode as E
    from . import metrics as M
    from . import baselines as B
except ImportError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from src import config as C, encode as E, metrics as M, baselines as B

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    print("Invariants\n")

    # --- encoding -----------------------------------------------------------
    b = E.encode_pairs(["GTCACCAATCCTGTCCCTAG"], ["ATCACCTATCCTATCCCTAAGGG"])
    check("mismatch count reproduces the published distance",
          b["tabular"][0, 0] == 4, "expected 4")
    nan_row = E.encode_seqs([float("nan")], 23)
    check("a missing sequence encodes as all-N, never as bases",
          bool((nan_row == 4).all()),
          "str(nan) is 'nan' and 'a' is a valid base")

    # --- CFD ----------------------------------------------------------------
    check("CFD scores a perfect NGG match as 1.0",
          abs(B.cfd_score(["A" * 20], ["A" * 20 + "AGG"])[0] - 1.0) < 1e-12)
    check("CFD scores an unlisted PAM as 0",
          abs(B.cfd_score(["A" * 20], ["A" * 20 + "ATT"])[0]) < 1e-12)

    # --- metrics refuse to invent negatives ---------------------------------
    try:
        M.log_loss_brier(np.zeros(3), np.full(3, 0.5), None)
        ok = False
    except ValueError:
        ok = True
    check("probability metrics refuse to run without a labelled mask", ok)

    if not C.PAIRS.exists():
        print("\n(build the tables with `python -m src.run prep` for the rest)")
        return

    pairs = pd.read_parquet(C.PAIRS)
    obs = pd.read_parquet(C.OBS)

    # --- status vocabulary --------------------------------------------------
    allowed = {C.STATUS_POSITIVE, C.STATUS_TESTED_NEG, C.STATUS_UNOBSERVED,
               C.STATUS_UNINTERPRETABLE}
    check("every observation carries a status from the fixed vocabulary",
          set(obs.status.dropna().unique()) <= allowed,
          str(sorted(set(obs.status.dropna().unique()))))
    check("confirmed evidence is tiered apart from discovery detections",
          {"confirmed", "discovery"} <= set(obs.evidence_tier.dropna().unique()))
    amb = obs[obs.evidence_tier.eq("rejected_or_failed")]
    check("ambiguous validation rows are never a tested negative without a measurement",
          bool((amb[amb.indel_treatment_pct.isna()].status
                == C.STATUS_UNINTERPRETABLE).all()),
          f"{int(amb.indel_treatment_pct.isna().sum())} rows lack a measurement")

    # --- the pilot must not re-enter through the external set ---------------
    dup = obs[(obs.study_id != "Lazzarotto2020")
              & obs.in_pilot.fillna(False).astype(bool)]
    check("the database's re-publication of the pilot is flagged",
          len(dup) > 0, f"{len(dup):,} external rows duplicate the pilot")
    try:
        from . import run as R
    except ImportError:
        from src import run as R
    lab = R.external_labelled(obs, pairs)
    check("the external labelled set excludes the re-published pilot",
          not lab.in_pilot.fillna(False).astype(bool).any(),
          f"{len(lab):,} rows across {lab.study_id.nunique()} studies")

    # --- boolean flags are real booleans ------------------------------------
    check("boolean flags are bool dtype, so `~mask` is not a bitwise not",
          pairs.has_bulge.dtype == bool, str(pairs.has_bulge.dtype))

    # --- bulges -------------------------------------------------------------
    pil = pairs[pairs.source_set == "pilot"]
    gap = pil.site_sequence.str.contains("-", na=False)
    wide = pil.span.fillna(C.SITE_LEN) > C.SITE_LEN
    empty = pil.site_sequence.fillna("") == ""
    check("all three shapes of bulge row are flagged",
          bool((pil.has_bulge | ~(gap | wide | empty)).all()),
          f"gap={int(gap.sum())}, wide={int(wide.sum())}, empty={int(empty.sum())}")

    # --- candidate space ----------------------------------------------------
    if C.CANDIDATES.exists():
        cand = pd.read_parquet(C.CANDIDATES, columns=["pair_id"])
        det = pil[pil.detected_changeseq & ~pil.has_bulge]
        cov = det.pair_id.isin(set(cand.pair_id)).mean()
        check("candidate space contains every substitution-only detection",
              cov == 1.0, f"coverage {cov:.6%} of {len(det):,}")

    # --- sgRNA counts -------------------------------------------------------
    # The cellular sheet carries 60 distinct *labels* for 57 distinct sgRNAs: three
    # are typographic variants of sgRNAs also present under correct labels, and one
    # (LAG_site_3 -> LAG3_site_3) is a misspelling of an sgRNA appearing nowhere else.
    # The 60 has been quoted in error twice, so it is pinned here.
    pil_g = pairs[pairs.source_set.eq("pilot")]
    n_cell = pil_g[pil_g.detected_guideseq.fillna(False).astype(bool)].guide_sample.nunique()
    check("57 sgRNAs have cellular detections, not the 60 distinct labels",
          n_cell == 57 and pil_g.guide_sample.nunique() == 110,
          f"{pil_g.guide_sample.nunique()} pilot sgRNAs, {n_cell} with GUIDE-seq")

    # --- splits -------------------------------------------------------------
    check("the split key is the sequence pair, not the locus",
          "seq_key" in pairs.columns
          and pairs.seq_key.nunique() < pairs.pair_id.nunique(),
          f"{pairs.pair_id.nunique():,} loci -> {pairs.seq_key.nunique():,} sequences")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all invariants hold")


if __name__ == "__main__":
    main()
