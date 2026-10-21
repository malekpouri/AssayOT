"""Protocol driver.

    python -m src.run prep          harmonise the source tables
    python -m src.run scan          enumerate the candidate space (GPU, needs hg38)
    python -m src.run guide         unseen-sgRNA folds: all model levels + baselines
    python -m src.run validated     leave-one-guide-out on the confirmatory assay
    python -m src.run study         unseen study, on the external labelled set
    python -m src.run tech          unseen technology
    python -m src.run semisynth     masking experiment with a known mechanism
    python -m src.run prior         PU class-prior sensitivity sweep
    python -m src.run identify      identifiability / sensitivity analysis
    python -m src.run all

Each protocol writes per-row predictions to `results/pred_<protocol>.parquet` and a
metric table to `results/metrics_<protocol>.csv`, so the analysis step never refits.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import baselines as B
from . import config as C
from . import experiment as X
from . import metrics as M

PRIMARY_LEVELS = ["L1_supervised", "L2_nnpu", "L3_assay"]
ABLATIONS = ["A_no_obs_branch", "A_no_validation", "A_free_prop", "A_no_fp"]


def _frame():
    t0 = time.time()
    fr = X.build_frame()
    cand = pd.read_parquet(C.CANDIDATES)
    obs = pd.read_parquet(C.OBS)
    fr.val, sel_tab = X.selection_weights(fr, cand, obs)
    sel_tab.to_csv(C.RESULTS / "selection_strata.csv")
    print(f"frame: {len(fr.obs):,} observation rows, {len(fr.val):,} confirmatory rows "
          f"({time.time() - t0:.0f}s)")
    return fr, cand, obs


# ---------------------------------------------------------------------------
# protocol: unseen sgRNA
# ---------------------------------------------------------------------------

#: One fit is roughly 8 epochs at ~19 s. Ablations are run on a single fold, because
#: their purpose is to show which component carries the effect, not to estimate that
#: component's effect size to three decimals.
EPOCHS = 6
RATIO = 4
ABLATION_FOLDS = (0,)
MAX_HELDOUT_UNITS = 5


def protocol_guide(levels=None, epochs: int = EPOCHS) -> None:
    """Hold out sgRNA clusters. The evaluation question is whether the model recovers
    the sites an independent *cellular* assay reported, among the full candidate space
    for sgRNAs it has never seen.

    GUIDE-seq non-reports are not treated as confirmed negatives: AP and recall@K
    measure recovery of reported sites, and precision@K is reported as a lower bound,
    labelled as such in the output.
    """
    levels = levels or (PRIMARY_LEVELS + ABLATIONS)
    fr, cand, obs = _frame()
    folds = X.split_by_guide(fr)
    folds.to_csv(C.RESULTS / "folds_guide.csv", index=False)
    print(f"{folds.guide_sample.nunique()} sgRNAs in "
          f"{folds.cluster.nunique()} clusters over {folds.fold.nunique()} folds")

    gs_rows = fr.obs[(fr.obs.assay_id == "GUIDEseq")]
    preds = []
    for fold in sorted(folds.fold.unique()):
        tr_g = folds.loc[folds.fold != fold, "guide_sample"]
        te_g = set(folds.loc[folds.fold == fold, "guide_sample"])
        te = gs_rows[gs_rows.guide_sample.isin(te_g)].copy()
        if not len(te) or te.s.sum() == 0:
            print(f"  fold {fold}: no evaluable in-cell rows, skipped")
            continue
        te = B.score_all(te)
        for level in levels:
            if level in ABLATIONS and fold not in ABLATION_FOLDS:
                continue
            print(f"  fold {fold} / {level}", flush=True)
            model = X.train(fr, tr_g, level=level, epochs=epochs, ratio=RATIO,
                            seed=C.SEED + fold, verbose=False)
            p = X.predict(model, fr, te, context=C.CTX_INCELL, assay="GUIDEseq")
            p["level"] = level
            p["fold"] = fold
            preds.append(p)
            del model
            torch.cuda.empty_cache()
    if not preds:
        print("no predictions produced")
        return
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(C.RESULTS / "pred_guide.parquet", index=False)
    _summarise_guide(allp)


def _summarise_guide(allp: pd.DataFrame) -> None:
    rows = []
    for level, sub in allp.groupby("level"):
        # Both quantities are scored. `p_activity` is the quantity of interest;
        # `q_report` is the observable the likelihood was actually fitted to. Reporting
        # them side by side shows whether separating them buys anything, instead of
        # asserting that it does.
        for col, tag in (("p_activity", ""), ("q_report", " [report]")):
            pg = M.per_guide_ranking(sub, col, "s")
            s = M.macro_summary(pg)
            s["model"] = level + tag
            rows.append(s)
            pg.to_csv(C.RESULTS / f"perguide_guide_{level}{tag.strip(' []')}.csv",
                      index=False)
    # baselines, computed once on the union of folds
    base = allp[allp.level == allp.level.iloc[0]]
    for name in ["score_mismatch_count", "score_position_weighted", "score_cfd"]:
        pg = M.per_guide_ranking(base, name, "s")
        s = M.macro_summary(pg)
        s["model"] = name.replace("score_", "baseline_")
        rows.append(s)
        pg.to_csv(C.RESULTS / f"perguide_guide_{name}.csv", index=False)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.RESULTS / "metrics_guide.csv", index=False)
    print("\n== unseen-sgRNA protocol: macro over held-out sgRNAs ==")
    piv = out[out.metric.isin(["ap", "auroc", "recall@100"])].pivot(
        index="model", columns="metric", values="value")
    print(piv.round(4).to_string())


# ---------------------------------------------------------------------------
# protocol: confirmatory assay, leave one guide out
# ---------------------------------------------------------------------------

def protocol_validated(epochs: int = EPOCHS) -> None:
    """The external test that matters: can the model rank sites by whether a cellular
    confirmatory assay will call them active, for an sgRNA it has not seen?

    Only six sgRNAs have confirmatory data, so this is leave-one-guide-out over six
    folds and the conclusion is explicitly limited to that subpopulation: the sites
    sent for confirmation were chosen by a documented rule, not sampled at random.
    """
    fr, cand, obs = _frame()
    vg = sorted(fr.val.guide_sample.unique())
    print(f"confirmatory sgRNAs: {vg}")
    preds = []
    for held in vg:
        tr_g = [g for g in fr.obs.guide_sample.unique() if g != held]
        te = fr.val[fr.val.guide_sample == held].copy()
        if te.y.nunique() < 2:
            print(f"  {held}: single-class outcome ({int(te.y.sum())}/{len(te)} "
                  "active), ranking undefined, skipped")
            continue
        te = B.score_all(te)
        te["score_changeseq_reads"] = B.changeseq_reads_score(te, obs)
        for level in PRIMARY_LEVELS + ["A_no_validation"]:
            model = X.train(fr, tr_g, level=level, epochs=epochs, ratio=RATIO,
                            seed=C.SEED, verbose=False)
            p = X.predict(model, fr, te, context=C.CTX_INCELL, assay="TagSeq")
            p["level"] = level
            p["held_guide"] = held
            preds.append(p)
            del model
            torch.cuda.empty_cache()
        print(f"  {held}: {len(te)} sites, {int(te.y.sum())} active")
    if not preds:
        print("no evaluable confirmatory folds")
        return
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(C.RESULTS / "pred_validated.parquet", index=False)

    rows = []
    for level, sub in allp.groupby("level"):
        pg = M.per_guide_ranking(sub, "p_activity", "y", guide="held_guide")
        s = M.macro_summary(pg)
        s["model"] = level
        rows.append(s)
    base = allp[allp.level == allp.level.iloc[0]]
    for name in ["score_mismatch_count", "score_position_weighted", "score_cfd",
                 "score_changeseq_reads"]:
        pg = M.per_guide_ranking(base, name, "y", guide="held_guide")
        s = M.macro_summary(pg)
        s["model"] = name.replace("score_", "baseline_")
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.RESULTS / "metrics_validated.csv", index=False)

    # Calibration and selective prediction against a real label. This is the only
    # protocol where either is meaningful: elsewhere the negatives are unlabelled
    # candidates, and scoring a probability against them would be scoring it against
    # an assumption.
    cal, rc = [], []
    for level, sub in allp.groupby("level"):
        y = sub.y.to_numpy()
        p = sub.p_activity.to_numpy()
        lab = np.ones(len(sub), dtype=bool)
        d = M.log_loss_brier(y, p, lab)
        d["ece"] = M.expected_calibration_error(y, p, lab)
        d["model"] = level
        cal.append(d)
        # confidence = distance from the decision boundary
        conf = np.abs(p - 0.5)
        t = M.risk_coverage(y, p, conf, lab)
        t["model"] = level
        rc.append(t)
    pd.DataFrame(cal).to_csv(C.RESULTS / "calibration_validated.csv", index=False)
    pd.concat(rc, ignore_index=True).to_csv(
        C.RESULTS / "riskcoverage_validated.csv", index=False)
    print("\n== confirmatory-assay protocol (leave-one-sgRNA-out) ==")
    print(out[out.metric.isin(["ap", "auroc"])].pivot(
        index="model", columns="metric", values="value").round(4).to_string())


# ---------------------------------------------------------------------------
# protocol: external studies
# ---------------------------------------------------------------------------

def external_labelled(obs: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows from other studies where the label means something.

    `confirmed` = a targeted assay called the site positive. `rejected` = a targeted
    assay measured it and did not. Rows whose `Validation == FALSE` carries no
    measurement are excluded, because CRISPRoffT states that value can also mean a
    failed PCR.
    """
    e = obs[(obs.study_id != "Lazzarotto2020")].copy()
    # exclude the database's re-publication of the pilot, so an "external" test is
    # not scored on rows the model trained on under a different study id
    e = e[~e.in_pilot.fillna(False).astype(bool)]
    e = X.drop_editor_assays(e)
    lab = e[e.evidence_tier.eq("confirmed")
            | (e.evidence_tier.eq("rejected_or_failed") & e.indel_treatment_pct.notna())]
    lab = lab.assign(y=lab.evidence_tier.eq("confirmed").astype(np.float32))
    lab = lab.merge(pairs[["pair_id", "guide_sequence", "site_sequence",
                           "genome_build"]], on="pair_id", how="left",
                    suffixes=("", "_p"))
    lab = lab.dropna(subset=["guide_sequence", "site_sequence"])
    lab["guide_sample"] = lab.guide_sequence
    return lab


def protocol_study(epochs: int = EPOCHS, by: str = "study_id") -> None:
    """Hold out a whole study (or technology) and test on its labelled rows.

    The held-out unit is removed from training entirely, including its unlabelled
    candidate rows, so nothing about it is seen. Each level is fitted once per held-out
    unit; the pilot-only variant is also fitted so the value of having other studies in
    training is visible rather than assumed.
    """
    obs = pd.read_parquet(C.OBS)
    pairs = pd.read_parquet(C.PAIRS)
    lab = external_labelled(obs, pairs)
    usable = (lab.groupby(by, observed=True)["y"]
              .agg(["size", "nunique", "sum"]).query("nunique > 1 and size >= 20")
              .sort_values("size", ascending=False).head(MAX_HELDOUT_UNITS))
    print(f"{by}: {len(usable)} held-out units with both classes and >=20 rows")
    print(usable.to_string())
    if not len(usable):
        print("no usable held-out unit; external labels are too sparse for this split")
        return

    cand = pd.read_parquet(C.CANDIDATES)
    fr_pilot, _, _ = _frame()   # fitted once; it does not depend on the held-out unit
    preds = []
    for unit in usable.index:
        te = lab[lab[by] == unit].copy()
        te = B.score_all(te)
        # studies to exclude from training: the held-out unit itself
        if by == "study_id":
            drop = {str(unit)}
        else:
            drop = set(lab.loc[lab[by] == unit, "study_id"].astype(str).unique())
        fr_multi = X.build_frame_multistudy(exclude_studies=drop)
        fr_multi.val, _ = X.selection_weights(fr_multi, cand, obs)
        for level in PRIMARY_LEVELS:
            for tag, fr_use in (("multi", fr_multi), ("pilot_only", fr_pilot)):
                if tag == "pilot_only" and level != "L3_assay":
                    continue
                model = X.train(fr_use, fr_use.obs.guide_sample.unique(), level=level,
                                epochs=epochs, ratio=RATIO, seed=C.SEED,
                                verbose=False)
                p = X.predict(model, fr_use, te, context=C.CTX_INCELL, assay="TagSeq")
                p["level"] = level if tag == "multi" else f"{level}_pilot_only"
                p["held_unit"] = str(unit)
                preds.append(p)
                del model
                torch.cuda.empty_cache()
        print(f"  {unit}: {len(te)} labelled rows, {int(te.y.sum())} confirmed")
    allp = pd.concat(preds, ignore_index=True)
    tag = "study" if by == "study_id" else "tech"
    allp.to_parquet(C.RESULTS / f"pred_{tag}.parquet", index=False)
    rows = []
    for level, sub in allp.groupby("level"):
        pg = M.per_guide_ranking(sub, "p_activity", "y", guide="held_unit")
        s = M.macro_summary(pg)
        s["model"] = level
        rows.append(s)
    base = allp[allp.level == allp.level.iloc[0]]
    for name in ["score_mismatch_count", "score_cfd"]:
        pg = M.per_guide_ranking(base, name, "y", guide="held_unit")
        s = M.macro_summary(pg)
        s["model"] = name.replace("score_", "baseline_")
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.RESULTS / f"metrics_{tag}.csv", index=False)
    print(f"\n== unseen-{tag} protocol ==")
    print(out[out.metric.isin(["ap", "auroc"])].pivot(
        index="model", columns="metric", values="value").round(4).to_string())


# ---------------------------------------------------------------------------
# protocol: semi-synthetic masking
# ---------------------------------------------------------------------------

def protocol_semisynth(epochs: int = EPOCHS, hide_frac: float = 0.5) -> None:
    """Hide known positives under a mechanism we control, then see which formulation
    recovers them.

    Two mechanisms: completely at random, which is the assumption standard PU learning
    makes, and signal-dependent, where low-read sites are hidden preferentially, which
    is how a detection limit actually behaves. The contrast is the point; this is a
    controlled check, not a substitute for real validation.
    """
    fr, cand, obs = _frame()
    cs = fr.obs[fr.obs.assay_id == "CHANGEseq"].copy()
    reads = (obs[obs.assay_id == "CHANGEseq"].groupby("pair_id")["signal_value"].max())
    cs["reads"] = cs.pair_id.map(reads).fillna(0.0)
    pos = cs[cs.s > 0]
    rng = np.random.default_rng(C.SEED)
    out_rows = []
    for mech in ("random", "signal_dependent"):
        if mech == "random":
            hide = rng.random(len(pos)) < hide_frac
        else:
            # hide preferentially where the in-vitro signal is weak
            r = pos.reads.rank(pct=True).to_numpy()
            prob = np.clip(2 * hide_frac * (1 - r), 0, 1)
            hide = rng.random(len(pos)) < prob
        hidden_ids = set(pos.pair_id[hide])
        print(f"  {mech}: hiding {len(hidden_ids):,} of {len(pos):,} positives")
        tr = fr.obs.copy()
        mask = tr.pair_id.isin(hidden_ids) & (tr.assay_id == "CHANGEseq")
        tr.loc[mask, "s"] = 0.0
        fr2 = X.Frame(tr, fr.val, fr.assay_index, fr.study_index)
        te = cs[cs.pair_id.isin(hidden_ids) | (cs.s == 0)].copy()
        te["y"] = te.pair_id.isin(hidden_ids).astype(np.float32)
        te = te.sample(n=min(len(te), 400_000), random_state=0)
        for level in PRIMARY_LEVELS:
            model = X.train(fr2, fr2.obs.guide_sample.unique(), level=level,
                            epochs=epochs, ratio=RATIO, seed=C.SEED, verbose=False)
            p = X.predict(model, fr2, te, context=C.CTX_INVITRO, assay="CHANGEseq")
            pg = M.per_guide_ranking(p, "p_activity", "y")
            s = M.macro_summary(pg)
            s["model"] = level
            s["mechanism"] = mech
            out_rows.append(s)
            del model
            torch.cuda.empty_cache()
    out = pd.concat(out_rows, ignore_index=True)
    out.to_csv(C.RESULTS / "metrics_semisynth.csv", index=False)
    print("\n== semi-synthetic masking ==")
    print(out[out.metric.eq("ap")].pivot(index="model", columns="mechanism",
                                         values="value").round(4).to_string())


# ---------------------------------------------------------------------------
# identifiability
# ---------------------------------------------------------------------------

def protocol_prior(epochs: int = EPOCHS,
                   priors=(0.005, 0.01, 0.02, 0.05, 0.10)) -> None:
    """Sensitivity of the PU baseline to the assumed class prior.

    The prior is not identifiable from positive-unlabelled data without further
    assumptions, so a PU result reported at one value of it is reporting a choice.
    Fitted on a single unseen-sgRNA fold, which is enough to show the direction and
    size of the dependence.
    """
    fr, cand, obs = _frame()
    folds = X.split_by_guide(fr)
    tr_g = folds.loc[folds.fold != 0, "guide_sample"]
    te_g = set(folds.loc[folds.fold == 0, "guide_sample"])
    gs = fr.obs[(fr.obs.assay_id == "GUIDEseq")
                & fr.obs.guide_sample.isin(te_g)].copy()
    if not len(gs) or gs.s.sum() == 0:
        print("no evaluable in-cell rows on fold 0")
        return
    rows = []
    for prior in priors:
        model = X.train(fr, tr_g, level="L2_nnpu", epochs=epochs, ratio=RATIO,
                        prior=prior, seed=C.SEED, verbose=False)
        p = X.predict(model, fr, gs, context=C.CTX_INCELL, assay="GUIDEseq")
        pg = M.per_guide_ranking(p, "p_activity", "s")
        s = M.macro_summary(pg)
        s["prior"] = prior
        rows.append(s)
        ap = s.loc[s.metric == "ap", "value"].iloc[0]
        print(f"  prior={prior:.3f}  AP={ap:.4f}  mean p={p.p_activity.mean():.4f}")
        del model
        torch.cuda.empty_cache()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.RESULTS / "metrics_prior.csv", index=False)
    print("\n== PU class-prior sensitivity (unseen sgRNA, fold 0) ==")
    print(out[out.metric.isin(["ap", "auroc"])]
          .pivot(index="prior", columns="metric", values="value").round(4).to_string())


def protocol_identify(epochs: int = EPOCHS, n_seeds: int = 3) -> None:
    """Is the activity/detection split pinned down by the data, or only their product?

    Each configuration is fitted from several random initialisations. For each we
    record the fit to the observed reports and the spread, across initialisations, of
    the recovered activity probabilities on a fixed evaluation set. A configuration
    that fits equally well but lands on very different activity estimates is not
    identified, and its point estimates should not be reported as cleavage
    probabilities (roadmap section 6).
    """
    fr, cand, obs = _frame()
    eva = fr.obs[fr.obs.assay_id == "GUIDEseq"].sample(
        n=min(200_000, int((fr.obs.assay_id == "GUIDEseq").sum())),
        random_state=0).copy()
    configs = {
        "constrained + anchor": "L3_assay",
        "constrained, no anchor": "A_no_validation",
        "free propensity + anchor": "A_free_prop",
        "constant propensity": "A_no_obs_branch",
    }
    rec = []
    for label, level in configs.items():
        P, Q = [], []
        for seed in range(n_seeds):
            model = X.train(fr, fr.obs.guide_sample.unique(), level=level,
                            epochs=epochs, ratio=RATIO, seed=1000 + seed,
                            verbose=False)
            p = X.predict(model, fr, eva, context=C.CTX_INCELL, assay="GUIDEseq")
            P.append(p.p_activity.to_numpy())
            Q.append(p.q_report.to_numpy())
            del model
            torch.cuda.empty_cache()
        P = np.stack(P)
        Q = np.stack(Q)
        y = eva.s.to_numpy(dtype=float)
        ll = np.mean([
            (y * np.log(np.clip(q, 1e-7, 1)) +
             (1 - y) * np.log(np.clip(1 - q, 1e-7, 1))).mean() for q in Q])
        rec.append(dict(
            config=label, level=level,
            report_loglik=float(ll),
            activity_sd_across_seeds=float(P.std(0).mean()),
            activity_range_across_seeds=float((P.max(0) - P.min(0)).mean()),
            report_sd_across_seeds=float(Q.std(0).mean()),
            activity_mean=float(P.mean()),
        ))
        print(f"  {label}: loglik(report)={ll:.5f} "
              f"sd(activity)={P.std(0).mean():.4f} sd(report)={Q.std(0).mean():.4f}")
    out = pd.DataFrame(rec)
    out.to_csv(C.RESULTS / "identifiability.csv", index=False)
    print("\n== identifiability ==")
    print(out.round(5).to_string(index=False))


# ---------------------------------------------------------------------------

def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if what in ("prep", "all"):
        from . import prep
        prep.main()
    if what in ("scan", "all"):
        from . import scan
        scan.main()
    if what in ("guide", "all"):
        protocol_guide()
    if what in ("validated", "all"):
        protocol_validated()
    if what in ("study", "all"):
        protocol_study(by="study_id")
    if what in ("tech", "all"):
        protocol_study(by="assay_type")
    if what in ("semisynth", "all"):
        protocol_semisynth()
    if what in ("prior", "all"):
        protocol_prior()
    if what in ("identify", "all"):
        protocol_identify()
    print(f"\n[{what}] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
