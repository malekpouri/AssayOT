"""Compact digest of every result file, for writing the paper against.

    python -m src.summary
"""
from __future__ import annotations

import sys

import pandas as pd

try:
    from . import config as C
    from . import metrics as M
except ImportError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from src import config as C
    from src import metrics as M

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def show_metrics(tag: str, metrics=("ap", "auroc", "recall@100")) -> None:
    p = C.RESULTS / f"metrics_{tag}.csv"
    if not p.exists():
        print(f"\n### {tag}: not run")
        return
    d = pd.read_csv(p)
    d = d[d.metric.isin(metrics)]
    print(f"\n### {tag}")
    piv = d.pivot_table(index="model", columns="metric", values="value")
    lo = d.pivot_table(index="model", columns="metric", values="lo")
    hi = d.pivot_table(index="model", columns="metric", values="hi")
    for m in piv.columns:
        piv[m] = [f"{v:.3f} ({a:.3f}-{b:.3f})" if pd.notna(v) else "--"
                  for v, a, b in zip(piv[m], lo[m], hi[m])]
    print(piv.to_string())
    ng = d.groupby("model").n_guides.max()
    print("groups per model:", ng.to_dict())


def paired(tag: str, a: str, b: str, metric: str = "ap", guide_col="guide") -> None:
    fa = C.RESULTS / f"perguide_{tag}_{a}.csv"
    fb = C.RESULTS / f"perguide_{tag}_{b}.csv"
    if not (fa.exists() and fb.exists()):
        return
    r = M.paired_guide_bootstrap(pd.read_csv(fa), pd.read_csv(fb), metric)
    print(f"  paired {a} - {b} [{metric}]: delta={r['delta']:+.4f} "
          f"({r['lo']:+.4f}, {r['hi']:+.4f})  n={r['n_guides']}  p={r['p_two_sided']:.3f}")


def main() -> None:
    print("=" * 78)
    print("FLOOR SENSITIVITY")
    p = C.PROC / "floor_sensitivity.csv"
    if p.exists():
        print(pd.read_csv(p).to_string(index=False))

    for tag in ("guide", "validated", "study", "tech"):
        show_metrics(tag)

    print("\n### paired comparisons (positive delta favours the first model)")
    pairs_to_test = (("L3_assay", "L1_supervised"), ("L3_assay", "L2_nnpu"),
                     ("L1_supervised", "L2_nnpu"), ("L3_assay", "score_cfd"),
                     ("L1_supervised", "score_cfd"))
    for tag in ("guide", "validated", "study", "tech"):
        done = False
        for a, b in pairs_to_test:
            if (C.RESULTS / f"perguide_{tag}_{a}.csv").exists() and \
               (C.RESULTS / f"perguide_{tag}_{b}.csv").exists():
                if not done:
                    print(f"  -- {tag}")
                    done = True
                paired(tag, a, b)

    for name, title in (("identifiability", "IDENTIFIABILITY"),
                        ("metrics_semisynth", "SEMI-SYNTHETIC MASKING"),
                        ("calibration_validated", "CALIBRATION (confirmatory)"),
                        ("selection_strata", "SELECTION STRATA")):
        f = C.RESULTS / f"{name}.csv"
        if f.exists():
            print(f"\n### {title}")
            d = pd.read_csv(f)
            if name == "metrics_semisynth":
                d = d[d.metric.eq("ap")]
                print(d.pivot_table(index="model", columns="mechanism",
                                    values="value").round(4).to_string())
            else:
                print(d.round(5).to_string(index=False))

    print("\n### ablations (unseen sgRNA, AP)")
    f = C.RESULTS / "metrics_guide.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.metric.eq("ap") & ~d.model.astype(str).str.endswith("[report]")]
        print(d[["model", "value", "lo", "hi", "n_guides"]]
              .sort_values("value", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
