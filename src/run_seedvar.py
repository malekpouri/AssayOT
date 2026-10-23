"""How much of a reported difference is the optimiser rather than the model?

Rerunning the confirmatory protocol with identical code and an identical seed moved
macro average precision by -0.011 for `L3_assay` and +0.039 for `L2_nnpu`, while
`L1_supervised` and `A_no_validation` reproduced to four decimal places. Whatever its
source -- non-deterministic GPU reductions amplified by the clamped logarithms in the
nnPU risk and the observation likelihood -- a point estimate that moves by 0.04 between
identical runs cannot carry a comparison of 0.035.

So the headline contrast is re-estimated as a distribution rather than a number: each
level is refitted from several seeds, the whole leave-one-sgRNA-out protocol is scored
per seed, and the spread across seeds is reported alongside the guide-level bootstrap.
Two sources of uncertainty, both of which belong in the paper.

    python -m src.run_seedvar            # default 3 seeds
    N_SEEDS=5 python -m src.run_seedvar
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from . import baselines as B
from . import config as C
from . import experiment as X
from . import metrics as M
from .run import EPOCHS, RATIO, _frame

#: The three that decide the paper's claim: no anchor, anchor with the observation
#: model, anchor without it.
LEVELS = ["L1_supervised", "L3_assay", "C1_supervised_anchor"]


def main(n_seeds: int | None = None, epochs: int = EPOCHS) -> None:
    n_seeds = n_seeds or int(os.environ.get("N_SEEDS", 3))
    fr, cand, obs = _frame()
    vg = sorted(fr.val.guide_sample.unique())
    rows = []
    for seed_i in range(n_seeds):
        seed = C.SEED + 1000 * seed_i
        for held in vg:
            tr_g = [g for g in fr.obs.guide_sample.unique() if g != held]
            te = fr.val[fr.val.guide_sample == held].copy()
            if te.y.nunique() < 2:
                continue
            te = B.score_all(te)
            for level in LEVELS:
                t0 = time.time()
                model = X.train(fr, tr_g, level=level, epochs=epochs, ratio=RATIO,
                                seed=seed, verbose=False)
                p = X.predict(model, fr, te, context=C.CTX_INCELL, assay="TagSeq")
                p["level"] = level
                p["held_guide"] = held
                p["seed_i"] = seed_i
                pg = M.per_guide_ranking(p, "p_activity", "y", guide="held_guide")
                rows.append(dict(seed_i=seed_i, seed=seed, level=level, guide=held,
                                 ap=float(pg.ap.iloc[0]), auroc=float(pg.auroc.iloc[0]),
                                 n=int(len(te)), n_pos=int(te.y.sum())))
                del model
                torch.cuda.empty_cache()
                print(f"  seed {seed_i} / {held} / {level}: ap="
                      f"{rows[-1]['ap']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(C.RESULTS / "seedvar_validated_perguide.csv", index=False)

    # macro AP per (level, seed), then the spread of that macro across seeds
    macro = d.groupby(["level", "seed_i"]).ap.mean().reset_index()
    summ = macro.groupby("level").ap.agg(["mean", "std", "min", "max", "count"])
    summ.to_csv(C.RESULTS / "seedvar_validated.csv")

    # paired contrasts within seed, so the optimiser draw is held fixed
    piv = macro.pivot(index="seed_i", columns="level", values="ap")
    con = []
    for a, b in [("C1_supervised_anchor", "L3_assay"),
                 ("C1_supervised_anchor", "L1_supervised"),
                 ("L3_assay", "L1_supervised")]:
        if a in piv.columns and b in piv.columns:
            dd = (piv[a] - piv[b]).to_numpy()
            con.append(dict(a=a, b=b, mean_delta=float(dd.mean()),
                            sd_delta=float(dd.std(ddof=1)) if len(dd) > 1 else np.nan,
                            min_delta=float(dd.min()), max_delta=float(dd.max()),
                            n_seeds=len(dd),
                            sign_stable=bool((dd > 0).all() or (dd < 0).all())))
    pd.DataFrame(con).to_csv(C.RESULTS / "seedvar_contrasts.csv", index=False)

    print("\n== macro AP across seeds ==")
    print(summ.round(4).to_string())
    print("\n== within-seed contrasts ==")
    print(pd.DataFrame(con).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
