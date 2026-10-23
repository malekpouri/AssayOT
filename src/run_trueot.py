"""External validation on TrueOT: 25 sgRNAs, 9 laboratories, measured editing.

This is the protocol the paper's confirmatory claim most needed and could not previously
support. The confirmatory experiment inside the pilot gives six sgRNAs from one
laboratory, selected by a documented but numerically unspecified rule. TrueOT gives
measurements of the same kind -- editing of the target sequence quantified directly in
cells, positive above 0.1% -- for sgRNAs and laboratories that are nowhere in the
training corpus.

Every model is trained on the pilot exactly as in the other protocols and then scored,
without refitting, on TrueOT. No TrueOT row is ever trained on, so there is no split to
choose and no selection rule to model: the comparison is a straight transfer test.

    python -m src.run_trueot
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from . import baselines as B
from . import baselines_ref as R
from . import config as C
from . import experiment as X
from . import metrics as M
from . import trueot as T
from .run import EPOCHS, RATIO, _frame

#: Everything the paper compares, on one external benchmark.
LEVELS = ["L1_supervised", "L2_nnpu", "L3_assay", "C1_supervised_anchor",
          "C2_anchor_unweighted", "A_no_validation"]


def main(epochs: int = EPOCHS) -> None:
    fr, cand, obs = _frame()
    d = T.evaluable(T.load_trueot())
    print(f"TrueOT: {len(d):,} rows, {int(d.y.sum())} positive, "
          f"{d.guide_sample.nunique()} sgRNAs, {d.study_id.nunique()} studies")

    # every pilot sgRNA is available for training; no TrueOT sgRNA is among them
    tr_g = fr.obs.guide_sample.unique()
    assert not set(d.guide_sample) & set(tr_g), "TrueOT sgRNA present in training set"

    te = B.score_all(d)
    rows, preds = [], []

    for level in LEVELS:
        t0 = time.time()
        model = X.train(fr, tr_g, level=level, epochs=epochs, ratio=RATIO,
                        seed=C.SEED, verbose=False)
        p = X.predict(model, fr, te, context=C.CTX_INCELL, assay="TagSeq")
        p["level"] = level
        preds.append(p)
        del model
        torch.cuda.empty_cache()
        print(f"  {level} ({time.time() - t0:.0f}s)", flush=True)

    for name, fn in R.REFERENCE_MODELS.items():
        t0 = time.time()
        score = fn(fr, tr_g, te, seed=C.SEED)
        p = te.copy()
        p["p_activity"] = score
        p["level"] = name
        preds.append(p)
        print(f"  {name} ({time.time() - t0:.0f}s)", flush=True)

    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(C.RESULTS / "pred_trueot.parquet", index=False)

    for level, sub in allp.groupby("level"):
        pg = M.per_guide_ranking(sub, "p_activity", "y")
        pg.to_csv(C.RESULTS / f"perguide_trueot_{level}.csv", index=False)
        s = M.macro_summary(pg)
        s["model"] = level
        rows.append(s)

    base = allp[allp.level == allp.level.iloc[0]]
    for name in ["score_mismatch_count", "score_position_weighted", "score_cfd"]:
        pg = M.per_guide_ranking(base, name, "y")
        pg.to_csv(C.RESULTS / f"perguide_trueot_{name}.csv", index=False)
        s = M.macro_summary(pg)
        s["model"] = name.replace("score_", "baseline_")
        rows.append(s)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.RESULTS / "metrics_trueot.csv", index=False)

    # the base rate a random ranker would reach, for reading the AP column against
    br = d.groupby("guide_sample").y.mean().mean()
    pd.DataFrame([dict(macro_base_rate=br, n_guides=d.guide_sample.nunique(),
                       n_rows=len(d), n_pos=int(d.y.sum()),
                       n_studies=d.study_id.nunique())]).to_csv(
        C.RESULTS / "trueot_summary.csv", index=False)

    print(f"\n== TrueOT (macro base rate {br:.3f}) ==")
    print(out[out.metric.isin(["ap", "auroc"])].pivot(
        index="model", columns="metric", values="value").round(4).to_string())


if __name__ == "__main__":
    main()
