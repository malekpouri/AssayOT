"""Benjamini--Hochberg across every model comparison the paper reports.

The analysis plan fixed a multiple-testing procedure for the detection-floor analysis
and not for comparisons between models. By the end of the study 23 paired contrasts
were being reported, a family in which roughly one spurious result at alpha=0.05 is
expected -- enough to manufacture a conclusion. This recomputes the adjusted values
from the contrast files so the paper's qualifications cannot drift from them.

    python -m src.multiplicity
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:  # pragma: no cover
    from src import config as C


def collect() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(C.RESULTS / "metrics_*contrasts*.csv"))):
        d = pd.read_csv(f)
        if not {"a", "b", "p"} <= set(d.columns):
            continue
        d["source"] = (f.split("/")[-1].replace("metrics_", "")
                       .replace("_contrasts.csv", ""))
        frames.append(d[["source", "a", "b", "delta", "p"]])
    d = pd.concat(frames, ignore_index=True).dropna(subset=["p"])
    return d.drop_duplicates(["source", "a", "b"]).sort_values("p").reset_index(drop=True)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted values for p sorted ascending.

    The cumulative minimum runs from the LARGEST p-value backwards. Taking it forwards
    is a silent and total failure -- it drives every value to the first one, so a
    contrast at p=0.90 reports as significant.
    """
    m = len(p)
    raw = p * m / np.arange(1, m + 1)
    return np.minimum.accumulate(raw[::-1])[::-1].clip(max=1.0)


def main() -> None:
    d = collect()
    d["rank"] = np.arange(1, len(d) + 1)
    d["bh_q"] = benjamini_hochberg(d.p.to_numpy())
    d["sig_raw"] = d.p < 0.05
    d["sig_bh"] = d.bh_q < 0.05
    d.to_csv(C.RESULTS / "metrics_multiplicity.csv", index=False)
    lost = d[d.sig_raw & ~d.sig_bh]
    print(f"{len(d)} contrasts; {int(d.sig_bh.sum())} survive Benjamini-Hochberg at q<0.05")
    print(f"\n{len(lost)} nominally significant contrasts do not survive:")
    if len(lost):
        print(lost[["source", "a", "b", "delta", "p", "bh_q"]].round(4).to_string(index=False))
    assert not benjamini_hochberg(np.array([0.001, 0.9]))[1] < 0.05, \
        "BH direction regression: a p=0.9 contrast must never be significant"


if __name__ == "__main__":
    main()
