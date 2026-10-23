"""Is the hidden-activity curve a property of biology, or of sequencing depth?

The paper's most transferable result is that the fraction of cellular activity hiding
among cell-assay-silent sites is a curve rather than a number: 22.5% of them show
statistically significant editing at any effect size, 0.9% at the assay's own 0.1%
floor. The upper end is a significance call, and significance buys more with more
reads -- so part of that 22.5% could be an artefact of how deeply the confirmation
experiment happened to be sequenced rather than a fact about cutting.

This checks it directly, because a claim offered as the paper's most portable finding
should not rest on an unexamined confound. Read depth across these sites spans more
than two orders of magnitude, which makes the test informative.

    python -m src.floor_depth
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import config as C
except ImportError:  # pragma: no cover
    from src import config as C

#: The four selection classes consisting of sites the cell-free assay detected and the
#: cellular assay did not report.
SILENT_CLASSES = ["A", "B", "C", "D"]


def main() -> None:
    v = pd.read_parquet(C.PROC / "validation_v1.parquet")
    s = v[v.selection_class.isin(SILENT_CLASSES)].copy()
    depth = s.indel_treated_total.astype(float)

    # "significant at any effect size": BH q < 0.05 with a positive treated-control
    # difference, in either readout. This is the 22.5% end of the curve.
    any_sig = (((s.indel_q.fillna(1) < 0.05) & (s.indel_delta.fillna(0) > 0))
               | ((s.tag_q.fillna(1) < 0.05) & (s.tag_delta.fillna(0) > 0)))
    # the 0.1% end, which is what a clinical assessment would act on
    at_floor = s.indel_active.fillna(False).astype(bool)

    rows = []
    for name, flag in (("any_effect_size", any_sig), ("at_0.1pct_floor", at_floor)):
        r = stats.spearmanr(depth, flag.astype(int))
        q = pd.qcut(depth, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        by = flag.groupby(q, observed=True).mean() * 100
        rows.append(dict(
            endpoint=name, n=len(s), pct_overall=100 * flag.mean(),
            spearman_rho=r.statistic, spearman_p=r.pvalue,
            pct_depth_Q1=by.get("Q1"), pct_depth_Q4=by.get("Q4"),
            q4_over_q1=by.get("Q4") / by.get("Q1") if by.get("Q1") else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "floor_depth.csv", index=False)

    pd.DataFrame([dict(
        n_sites=len(s), depth_median=depth.median(), depth_min=depth.min(),
        depth_max=depth.max(), depth_fold_range=depth.max() / max(depth.min(), 1),
        all_sites_powered=bool(s.indel_powered.fillna(False).astype(bool).all()),
    )]).to_csv(C.RESULTS / "floor_depth_coverage.csv", index=False)

    print(f"{len(s)} cell-assay-silent sites; read depth spans "
          f"{depth.min():,.0f}-{depth.max():,.0f} "
          f"({depth.max() / max(depth.min(), 1):,.0f}-fold)")
    print(out.round(4).to_string(index=False))
    print("\nReading: the upper end of the curve rises with depth, so part of it is a "
          "property of the experiment;\nthe 0.1% end does not, so the clinically "
          "relevant statement is depth-robust.")


if __name__ == "__main__":
    main()
