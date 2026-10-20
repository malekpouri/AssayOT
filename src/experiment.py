"""Splits, the validation-selection model, and the training loop.

The five evaluation protocols of roadmap section 9 are implemented here as `SPLITS`.
Every one of them holds out a *group*, never a random set of rows, because rows that
share an sgRNA, a locus or a sequence are not independent.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from . import config as C
    from . import encode as E
    from .model import (AssayAwareModel, nnpu_loss, observation_nll,
                        validation_nll)
except ImportError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from src import config as C
    from src import encode as E
    from src.model import (AssayAwareModel, nnpu_loss, observation_nll,
                           validation_nll)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
#: bfloat16 autocast for the encoder is available and about 1.5x faster, but it shifted
#: held-out average precision by ~0.008 in a side-by-side check. That is well inside the
#: bootstrap intervals reported in the paper, but it is an avoidable confound, so the
#: reported results are full float32. Set ASSAYOT_AMP=1 to trade it back for speed.
AMP = (os.environ.get("ASSAYOT_AMP") == "1"
       and torch.cuda.is_available() and torch.cuda.is_bf16_supported())

CTX = {C.CTX_INVITRO: 0, C.CTX_INCELL: 1}


# ---------------------------------------------------------------------------
# guide clustering
# ---------------------------------------------------------------------------

def guide_clusters(spacers: list[str], max_mm: int = None) -> np.ndarray:
    """Single-linkage clusters of sgRNAs within `max_mm` substitutions.

    Without this, two sgRNAs differing at one position land in different folds and the
    "unseen guide" test measures almost nothing (roadmap section 9, test 1).
    """
    max_mm = C.GUIDE_CLUSTER_MAX_MM if max_mm is None else max_mm
    n = len(spacers)
    arr = E.encode_seqs(spacers, E.SPACER_LEN)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = (arr[i + 1:] != arr[i]).sum(1)
        for off in np.nonzero(d <= max_mm)[0]:
            j = i + 1 + int(off)
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return np.array([find(i) for i in range(n)])


# ---------------------------------------------------------------------------
# training frame
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """Long observation table ready for the model."""
    obs: pd.DataFrame                  # pair_id, guide_sample, sequences, assay, context, s
    val: pd.DataFrame                  # confirmatory-assay rows with y and weight
    assay_index: dict = field(default_factory=dict)
    study_index: dict = field(default_factory=dict)


def build_frame(guides: set[str] | None = None) -> Frame:
    """Assemble the long observation table.

    `guides` restricts the candidate space to a set of sgRNA samples, which keeps the
    frame small when a protocol only needs part of it.
    """
    cand = pd.read_parquet(C.CANDIDATES)
    if guides is not None:
        cand = cand[cand.guide_sample.astype(str).isin(guides)]
    obs = pd.read_parquet(C.OBS)
    pairs = pd.read_parquet(C.PAIRS)
    val = pd.read_parquet(C.VALID)

    pilot = obs[obs.study_id == "Lazzarotto2020"]
    cs_det = set(pilot.loc[pilot.assay_id == "CHANGEseq", "pair_id"])
    gs_det = set(pilot.loc[pilot.assay_id == "GUIDEseq", "pair_id"])
    gs_guides = set(pairs.loc[pairs.pair_id.isin(gs_det), "guide_sample"].dropna())

    cand = cand[~cand.site_sequence.str.contains("-", na=False)].copy()
    # Only the pilot sgRNAs were ever put through CHANGE-seq and GUIDE-seq. Candidates
    # belonging to external sgRNAs must not appear here with s = 0: that would assert
    # "this assay did not report the site" for an assay that never examined it, which
    # is precisely the error this work is about. External sgRNAs enter training only
    # through `build_frame_multistudy`, paired with the assay that did examine them.
    pilot_guides = set(pairs.loc[pairs.source_set == "pilot", "guide_sample"].dropna())
    cand = cand[cand.guide_sample.astype(str).isin(pilot_guides)].copy()
    base = cand[["pair_id", "guide_sample", "guide_sequence", "site_sequence",
                 "n_mm_total", "pam_class"]]

    # CHANGE-seq was run for every pilot sgRNA: in vitro reports over the whole space.
    a = base.copy()
    a["assay_id"] = "CHANGEseq"
    a["context"] = C.CTX_INVITRO
    a["s"] = a.pair_id.isin(cs_det).astype(np.float32)

    # GUIDE-seq was run for 60 sgRNAs only. For the other 50 a site is *unobserved*
    # in cell, so those rows are simply absent rather than present with s = 0.
    b = base[base.guide_sample.isin(gs_guides)].copy()
    b["assay_id"] = "GUIDEseq"
    b["context"] = C.CTX_INCELL
    b["s"] = b.pair_id.isin(gs_det).astype(np.float32)

    long = pd.concat([a, b], ignore_index=True)
    long["study_id"] = "Lazzarotto2020"
    # Several million rows of repeated short strings; categoricals keep this in a
    # couple of hundred megabytes instead of a couple of gigabytes.
    for col in ("guide_sample", "assay_id", "context", "study_id", "pam_class"):
        long[col] = long[col].astype("category")

    # confirmatory assay rows
    v = val[["pair_id", "Target Name", "Off-target Sequence", "selection_class",
             "is_active", "is_interpretable", "tag_delta", "indel_delta",
             "tag_treated_total"]].rename(
        columns={"Target Name": "guide_sample",
                 "Off-target Sequence": "site_sequence"})
    v = v.merge(pairs[["pair_id", "guide_sequence", "n_mismatch"]],
                on="pair_id", how="left")
    v = v[v.is_interpretable].copy()
    v["y"] = v.is_active.astype(np.float32)
    v["context"] = C.CTX_INCELL
    v["assay_id"] = "TagSeq"
    v["study_id"] = "Lazzarotto2020"

    assays = sorted(set(long.assay_id) | set(v.assay_id))
    studies = sorted(set(long.study_id) | set(v.study_id))
    ai = {a: i for i, a in enumerate(assays)}
    si = {s: i for i, s in enumerate(studies)}
    return Frame(long, v, ai, si)


def drop_editor_assays(df: pd.DataFrame, col: str = "assay_type") -> pd.DataFrame:
    """Remove base- and prime-editor assays.

    Filtering on the nuclease field is not sufficient: some base-editor rows carry
    `SpCas9` for the nickase component, so an assay whose read-out is an editor rather
    than a double-strand break would otherwise enter a study scoped to conventional
    SpCas9 cleavage (roadmap section 3).
    """
    import re
    pat = "|".join(C.EXCLUDED_ASSAY_PATTERNS)
    bad = df[col].astype(str).str.contains(pat, case=False, regex=True, na=False)
    if bad.any():
        print(f"  dropping {int(bad.sum())} rows from editor assays: "
              f"{sorted(df.loc[bad, col].astype(str).unique())}")
    return df[~bad]


def build_frame_multistudy(exclude_studies: set[str] = frozenset(),
                           min_reports: int = 20,
                           max_cand_per_study: int = 400_000) -> Frame:
    """Observation rows for the pilot *and* for external studies.

    For each external (study, assay) pair we take the sgRNAs that study used, pull their
    enumerated candidate space, and mark the sites that study reported. Without this the
    model only ever sees one study and the claim that modelling study differences helps
    transfer cannot be tested at all.

    An external study contributes only if it reported at least `min_reports` sites with
    an enumerable candidate space, and its unlabelled rows are capped so that one large
    study cannot dominate the likelihood.
    """
    base = build_frame()
    cand = pd.read_parquet(C.CANDIDATES)
    obs = pd.read_parquet(C.OBS)
    pairs = pd.read_parquet(C.PAIRS)

    gmap = pairs.set_index("pair_id")["guide_sequence"].to_dict()
    ext = obs[(obs.study_id != "Lazzarotto2020")
              & obs.status.eq(C.STATUS_POSITIVE)].copy()
    ext = ext[~ext.study_id.isin(exclude_studies)]
    ext = drop_editor_assays(ext)
    ext["guide_sequence"] = ext.pair_id.map(gmap)
    ext = ext.dropna(subset=["guide_sequence"])

    cand_by_guide = dict(tuple(cand.groupby(cand.guide_sample.astype(str))))
    rng = np.random.default_rng(C.SEED)
    blocks = []
    for (study, assay), g in ext.groupby(["study_id", "assay_id"], observed=True):
        guides = sorted(set(g.guide_sequence) & set(cand_by_guide))
        if not guides:
            continue
        reported = set(g.pair_id)
        sub = pd.concat([cand_by_guide[x] for x in guides], ignore_index=True)
        sub = sub[["pair_id", "guide_sample", "guide_sequence", "site_sequence",
                   "n_mm_total", "pam_class"]].copy()
        sub["s"] = sub.pair_id.isin(reported).astype(np.float32)
        if sub.s.sum() < min_reports:
            continue
        if len(sub) > max_cand_per_study:
            pos = sub[sub.s > 0]
            neg = sub[sub.s == 0]
            k = max(0, max_cand_per_study - len(pos))
            keep = rng.choice(len(neg), size=min(k, len(neg)), replace=False)
            sub = pd.concat([pos, neg.iloc[keep]], ignore_index=True)
        ctx = g.context.mode()
        sub["context"] = ctx.iloc[0] if len(ctx) else C.CTX_INCELL
        sub["assay_id"] = str(assay)
        sub["study_id"] = str(study)
        blocks.append(sub)

    if not blocks:
        print("no external study met the inclusion rule; falling back to the pilot")
        return base
    extra = pd.concat(blocks, ignore_index=True)
    long = pd.concat([base.obs, extra], ignore_index=True)
    for col in ("guide_sample", "assay_id", "context", "study_id", "pam_class"):
        long[col] = long[col].astype("category")
    assays = sorted(set(long.assay_id.astype(str)) | {"TagSeq"})
    studies = sorted(set(long.study_id.astype(str)))
    print(f"multi-study frame: {len(long):,} rows, {len(studies)} studies, "
          f"{len(assays)} assays "
          f"({extra.study_id.nunique()} external studies contributed)")
    return Frame(long, base.val,
                 {a: i for i, a in enumerate(assays)},
                 {s: i for i, s in enumerate(studies)})


# ---------------------------------------------------------------------------
# validation-selection model (inverse-probability weights)
# ---------------------------------------------------------------------------

def selection_weights(frame: Frame, cand: pd.DataFrame,
                      obs: pd.DataFrame) -> pd.DataFrame:
    """Probability that a CHANGE-seq-detected site was sent for confirmation.

    The published rule assigned sites to classes using mismatch count, CHANGE-seq read
    depth and GUIDE-seq status, but the read thresholds separating "low", "medium" and
    "high" were not stated numerically, so the rule cannot be replayed exactly.
    Instead the selection probability is estimated non-parametrically on the three
    covariates the rule is documented to have used, within the sgRNAs that were
    validated at all. Sites in a stratum from which nothing was selected get no weight
    because they carry no information about the confirmatory outcome.
    """
    v = frame.val
    val_guides = sorted(v.guide_sample.unique())
    pilot = obs[(obs.study_id == "Lazzarotto2020")]
    cs = pilot[pilot.assay_id == "CHANGEseq"][["pair_id", "signal_value"]]
    gs_det = set(pilot.loc[pilot.assay_id == "GUIDEseq", "pair_id"])

    pop = cand[cand.guide_sample.isin(val_guides)].merge(cs, on="pair_id", how="inner")
    pop["gs"] = pop.pair_id.isin(gs_det)
    pop["mm_bin"] = pd.cut(pop.n_mm_total, [-1, 3, 4, 6], labels=["0-3", "4", "5-6"])
    pop["read_bin"] = pop.groupby("guide_sample", observed=True)["signal_value"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"]))
    pop["stratum"] = (pop.gs.astype(str) + "|" + pop.mm_bin.astype(str)
                      + "|" + pop.read_bin.astype(str))

    sel = pop.pair_id.isin(set(v.pair_id))
    tab = pd.DataFrame({"n_pop": pop.groupby("stratum", observed=True).size(),
                        "n_sel": pop[sel].groupby("stratum", observed=True).size()})
    tab["n_sel"] = tab.n_sel.fillna(0)
    tab["p_select"] = tab.n_sel / tab.n_pop
    smap = pop.set_index("pair_id")["stratum"].to_dict()
    v = v.copy()
    v["stratum"] = v.pair_id.map(smap)
    v = v.merge(tab[["p_select"]], left_on="stratum", right_index=True, how="left")
    # stabilised weights, trimmed at the 95th percentile so one rare stratum cannot
    # dominate the confirmatory term
    w = 1.0 / v.p_select.replace(0, np.nan)
    cap = np.nanpercentile(w, 95) if np.isfinite(w).any() else 1.0
    v["weight"] = np.clip(w.fillna(1.0), 0, cap)
    v["weight"] = v.weight / v.weight.mean()
    return v, tab


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def split_by_guide(frame: Frame, n_folds: int = None, seed: int = None):
    """Unseen-sgRNA folds over guide clusters."""
    n_folds = n_folds or C.N_FOLDS
    rng = np.random.default_rng(seed or C.SEED)
    g = (frame.obs[["guide_sample", "guide_sequence"]].drop_duplicates("guide_sample")
         .reset_index(drop=True))
    cl = guide_clusters(g.guide_sequence.tolist())
    g["cluster"] = cl
    clusters = g.cluster.unique()
    rng.shuffle(clusters)
    assign = {c: i % n_folds for i, c in enumerate(clusters)}
    g["fold"] = g.cluster.map(assign)
    return g[["guide_sample", "cluster", "fold"]]


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

def encode_rows(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    bund = E.encode_pairs(df.guide_sequence.tolist(), df.site_sequence.tolist())
    return dict(
        tokens=torch.from_numpy(bund["tokens"].astype(np.int64)),
        mismatch=torch.from_numpy(bund["mismatch"]),
        tab=torch.from_numpy(bund["tabular"]),
    )


class Batcher:
    """Each epoch: every reported-positive row, plus a fresh random sample of the
    unreported rows at a fixed ratio. Resampling rather than fixing the negative set
    keeps the whole candidate space in play without a 40-million-row epoch.

    Everything is held as plain numpy arrays rather than a DataFrame. Slicing a
    multi-million-row frame once per epoch and rebuilding the categorical-to-integer
    maps once per *batch* dominated the training time; with arrays both become fancy
    indexing.
    """

    def __init__(self, obs: pd.DataFrame, assay_index: dict, study_index: dict,
                 ratio: int = 5, batch: int = 8192, seed: int = 0):
        self.guide = obs.guide_sequence.to_numpy(dtype=object)
        self.site = obs.site_sequence.to_numpy(dtype=object)
        self.s = obs.s.to_numpy(dtype=np.float32)
        self.ctx = np.array([CTX[c] for c in obs.context.astype(str)], dtype=np.int64)
        self.assay = np.array([assay_index[a] for a in obs.assay_id.astype(str)],
                              dtype=np.int64)
        self.study = np.array([study_index[s] for s in obs.study_id.astype(str)],
                              dtype=np.int64)
        self.pos_idx = np.nonzero(self.s > 0)[0]
        self.unl_idx = np.nonzero(self.s == 0)[0]
        self.ratio = ratio
        self.batch = batch
        self.rng = np.random.default_rng(seed)

    def epoch(self):
        k = min(len(self.unl_idx), self.ratio * max(1, len(self.pos_idx)))
        pick = self.rng.choice(self.unl_idx, size=k, replace=False)
        idx = np.concatenate([self.pos_idx, pick])
        self.rng.shuffle(idx)
        #: fraction of the candidate space this sample represents, so the loss can be
        #: reweighted back to the true unlabelled mass
        w_unl = len(self.unl_idx) / max(1, k)
        for i in range(0, len(idx), self.batch):
            j = idx[i:i + self.batch]
            yield dict(guide=self.guide[j], site=self.site[j], s=self.s[j],
                       ctx=self.ctx[j], assay=self.assay[j], study=self.study[j]), w_unl


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

LEVELS = {
    # name                  propensity        validation  false-pos  loss
    "L1_supervised":  dict(mode="constant",   use_val=False, fp=False, loss="bce"),
    "L2_nnpu":        dict(mode="constant",   use_val=False, fp=False, loss="nnpu"),
    "L3_assay":       dict(mode="constrained", use_val=True,  fp=True,  loss="obs"),
    # ablations
    "A_no_obs_branch": dict(mode="constant",   use_val=True,  fp=False, loss="obs"),
    "A_no_validation": dict(mode="constrained", use_val=False, fp=True,  loss="obs"),
    "A_free_prop":     dict(mode="free",       use_val=True,  fp=True,  loss="obs"),
    "A_no_fp":         dict(mode="constrained", use_val=True,  fp=False, loss="obs"),
}


def train(frame: Frame, train_guides, level: str = "L3_assay",
          epochs: int = 12, lr: float = 2e-3, ratio: int = 5,
          batch: int = 8192, prior: float = 0.02, lam_val: float = 1.0,
          lam_hier: float = 1e-3, seed: int = 0, verbose: bool = True):
    cfg = LEVELS[level]
    torch.manual_seed(seed)
    tr = frame.obs[frame.obs.guide_sample.isin(train_guides)]
    val = frame.val[frame.val.guide_sample.isin(train_guides)] if cfg["use_val"] else None

    n_tab = len(E.tabular_pre_assay(
        E.encode_seqs(["A" * 20], 20), E.encode_seqs(["A" * 23], 23))[1])
    model = AssayAwareModel(
        n_assay=len(frame.assay_index), n_study=len(frame.study_index), n_tab=n_tab,
        propensity_mode=cfg["mode"], model_false_positive=cfg["fp"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batcher = Batcher(tr, frame.assay_index, frame.study_index,
                      ratio=ratio, batch=batch, seed=seed)
    # An epoch is every positive plus `ratio` times as many sampled unlabelled rows,
    # not the whole training frame, so the schedule has to be sized from that.
    n_pos = len(batcher.pos_idx)
    rows_per_epoch = n_pos + min(len(batcher.unl_idx), ratio * max(1, n_pos))
    steps = max(1, epochs * (rows_per_epoch // batch + 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    val_enc = None
    if val is not None and len(val):
        val_enc = encode_rows(val)
        val_enc = {k: v.to(DEV) for k, v in val_enc.items()}
        val_y = torch.tensor(val.y.to_numpy(np.float32), device=DEV)
        val_w = torch.tensor(val.weight.to_numpy(np.float32), device=DEV)
        val_ctx = torch.tensor([CTX[c] for c in val.context], device=DEV)
        val_as = torch.tensor([frame.assay_index[a] for a in val.assay_id], device=DEV)
        val_st = torch.tensor([frame.study_index[s] for s in val.study_id], device=DEV)

    for ep in range(epochs):
        model.train()
        tot = n = 0
        t0 = time.time()
        for b, w_unl in batcher.epoch():
            bund = E.encode_pairs(b["guide"], b["site"])
            tokens = torch.from_numpy(bund["tokens"].astype(np.int64)).to(DEV)
            mism = torch.from_numpy(bund["mismatch"]).to(DEV)
            tab = torch.from_numpy(bund["tabular"]).to(DEV)
            s = torch.from_numpy(b["s"]).to(DEV)
            ctx = torch.from_numpy(b["ctx"]).to(DEV)
            asy = torch.from_numpy(b["assay"]).to(DEV)
            sty = torch.from_numpy(b["study"]).to(DEV)
            # weight unlabelled rows back up to the mass they stand for
            w = torch.where(s > 0, torch.ones_like(s), torch.full_like(s, w_unl))
            w = w / w.mean()

            # The encoder runs in bfloat16; the likelihood is evaluated in float32,
            # because the mixture p*e + (1-p)*f and the nnPU risk both involve small
            # probabilities whose logs are not safe at reduced precision.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                out = model(tokens, mism, tab, ctx, asy, sty)
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            out["q_report"] = out["q_report"].clamp(1e-7, 1 - 1e-7)
            if cfg["loss"] == "bce":
                # level 1: treat every unreported candidate as a biological negative
                loss = F.binary_cross_entropy_with_logits(
                    out["activity_logit"], s, weight=w)
            elif cfg["loss"] == "nnpu":
                loss = nnpu_loss(out["activity_logit"], s, prior=prior)
            else:
                loss = observation_nll(out["q_report"], s, w)
            if val_enc is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                    vo = model(val_enc["tokens"], val_enc["mismatch"],
                               val_enc["tab"], val_ctx, val_as, val_st)
                loss = loss + lam_val * validation_nll(
                    vo["activity_logit"].float(), val_y, val_w)
            loss = loss + lam_hier * model.hierarchical_penalty()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            nb = len(b["s"])
            tot += loss.detach().item() * nb
            n += nb
        if verbose:
            print(f"    [{level}] epoch {ep + 1}/{epochs} loss={tot / max(1, n):.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def predict(model: AssayAwareModel, frame: Frame, df: pd.DataFrame,
            context: str = C.CTX_INCELL, assay: str = "TagSeq",
            batch: int = 4096) -> pd.DataFrame:
    """Score rows. `context` chooses which activity head is read, so the in-vitro and
    in-cell predictions are never silently interchanged."""
    model.eval()
    guides = df.guide_sequence.to_numpy(dtype=object)
    sites = df.site_sequence.to_numpy(dtype=object)
    ps, qs, es = [], [], []
    for i in range(0, len(df), batch):
        g, s = guides[i:i + batch], sites[i:i + batch]
        bund = E.encode_pairs(g, s)
        n = len(g)
        ctx = torch.full((n,), CTX[context], device=DEV, dtype=torch.long)
        asy = torch.full((n,), frame.assay_index.get(assay, 0), device=DEV,
                         dtype=torch.long)
        # An unseen study has no embedding of its own; index 0 is held at the shrunk
        # grand mean. The activity head does not read the study effect at all, so
        # `p_activity` is unaffected by this choice; only `q_report` depends on it.
        sty = torch.zeros(n, device=DEV, dtype=torch.long)
        # Deliberately NOT under autocast. bfloat16 carries ~3 significant digits, so
        # scores spread over orders of magnitude collapse into large tie groups, which
        # corrupts ranking metrics and calibration. Inference is cheap; keep it exact.
        out = model(torch.from_numpy(bund["tokens"].astype(np.int64)).to(DEV),
                    torch.from_numpy(bund["mismatch"]).to(DEV),
                    torch.from_numpy(bund["tabular"]).to(DEV), ctx, asy, sty)
        ps.append(out["p_activity"].cpu().numpy())
        qs.append(out["q_report"].cpu().numpy())
        es.append(out["e_propensity"].cpu().numpy())
    out = df.copy()
    out["p_activity"] = np.concatenate(ps)
    out["q_report"] = np.concatenate(qs)
    out["e_propensity"] = np.concatenate(es)
    return out
