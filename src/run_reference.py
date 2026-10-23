"""Reference-architecture comparison, run outside the main protocol driver.

Kept in its own module so it can be added to the paper without touching `run.py`,
whose protocols are the pre-registered ones. It reuses the *same* folds, the same
training guides and the same evaluation rows, so the numbers drop straight into the
existing tables:

    python -m src.run_reference guide
    python -m src.run_reference validated
    python -m src.run_reference all
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import torch

from . import baselines as B
from . import baselines_ref as R
from . import config as C
from . import experiment as X
from . import metrics as M
from .run import EPOCHS, RATIO, _frame


def protocol_guide_reference(epochs: int = EPOCHS) -> None:
    """The unseen-sgRNA protocol, for the reimplemented published architectures."""
    fr, cand, obs = _frame()
    folds = pd.read_csv(C.RESULTS / "folds_guide.csv")
    gs_rows = fr.obs[fr.obs.assay_id == "GUIDEseq"]
    preds = []
    for fold in sorted(folds.fold.unique()):
        tr_g = folds.loc[folds.fold != fold, "guide_sample"]
        te_g = set(folds.loc[folds.fold == fold, "guide_sample"])
        te = gs_rows[gs_rows.guide_sample.isin(te_g)].copy()
        if not len(te) or te.s.sum() == 0:
            continue
        for name, fn in R.REFERENCE_MODELS.items():
            t0 = time.time()
            score = fn(fr, tr_g, te, seed=C.SEED + fold)
            p = te[["guide_sample", "s"]].copy()
            p["p_activity"] = score
            p["level"] = name
            p["fold"] = fold
            preds.append(p)
            print(f"  fold {fold} / {name}: {len(te):,} rows "
                  f"({time.time() - t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
    allp = pd.concat(preds, ignore_index=True)
    rows = []
    for level, sub in allp.groupby("level"):
        pg = M.per_guide_ranking(sub, "p_activity", "s")
        pg.to_csv(C.RESULTS / f"perguide_guide_{level}.csv", index=False)
        s = M.macro_summary(pg)
        s["model"] = level
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.RESULTS / "metrics_guide_reference.csv", index=False)
    allp.to_parquet(C.RESULTS / "pred_guide_reference.parquet", index=False)
    print("\n== unseen-sgRNA protocol, reference architectures ==")
    print(out[out.metric.isin(["ap", "auroc"])].pivot(
        index="model", columns="metric", values="value").round(4).to_string())


def protocol_validated_reference(epochs: int = EPOCHS) -> None:
    """The confirmatory protocol, for the reimplemented published architectures."""
    fr, cand, obs = _frame()
    vg = sorted(fr.val.guide_sample.unique())
    preds = []
    for held in vg:
        tr_g = [g for g in fr.obs.guide_sample.unique() if g != held]
        te = fr.val[fr.val.guide_sample == held].copy()
        if te.y.nunique() < 2:
            continue
        for name, fn in R.REFERENCE_MODELS.items():
            t0 = time.time()
            score = fn(fr, tr_g, te, seed=C.SEED)
            p = te[["guide_sample", "y"]].copy()
            p["p_activity"] = score
            p["level"] = name
            p["held_guide"] = held
            preds.append(p)
            print(f"  {held} / {name}: {len(te)} sites "
                  f"({time.time() - t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
    allp = pd.concat(preds, ignore_index=True)
    rows = []
    for level, sub in allp.groupby("level"):
        pg = M.per_guide_ranking(sub, "p_activity", "y", guide="held_guide")
        pg.to_csv(C.RESULTS / f"perguide_validated_{level}.csv", index=False)
        s = M.macro_summary(pg)
        s["model"] = level
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.RESULTS / "metrics_validated_reference.csv", index=False)
    allp.to_parquet(C.RESULTS / "pred_validated_reference.parquet", index=False)
    print("\n== confirmatory protocol, reference architectures ==")
    print(out[out.metric.isin(["ap", "auroc"])].pivot(
        index="model", columns="metric", values="value").round(4).to_string())


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("guide", "all"):
        protocol_guide_reference()
    if what in ("validated", "all"):
        protocol_validated_reference()


if __name__ == "__main__":
    main()
