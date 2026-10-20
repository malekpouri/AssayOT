"""Assay-aware off-target model.

The generative story, per observation of a (guide, site) pair in assay `a`:

    z_c  ~ Bernoulli( p(x, c) )            latent cleavage event in context c
    s    ~ Bernoulli( e(u, c, a) )  if z_c = 1
    s    ~ Bernoulli( f(c, a) )     if z_c = 0
    => P(s = 1 | x, c, a) = p * e + (1 - p) * f

Three commitments distinguish this from a product of two free functions:

1.  **Contexts are not pooled.** `in_vitro` (cell-free genomic DNA) and `in_cell` get
    separate activity heads over a shared encoder. In-vitro activity is an auxiliary
    task, not the latent truth behind the cellular one (roadmap section 6).

2.  **The observation branch is capacity-constrained.** It sees only `u`, a short vector
    of pre-assay summaries (mismatch count and position, PAM class), plus assay and
    study embeddings -- never the full sequence representation `h`. Two flexible
    functions of the same input multiply to a likelihood that cannot separate them;
    restricting one of them is what makes the factorisation estimable at all. The
    `propensity_mode` switch exposes this as an ablation.

3.  **The confirmatory assay is a separate likelihood term**, not another detection
    channel: targeted tag sequencing with a stated limit of detection is treated as a
    direct, noisy observation of `z_cell`, with an inverse-selection weight because the
    sites sent for validation were chosen by a documented rule, not at random.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import N_PAIR_TOKEN, SITE_LEN


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------

class PairEncoder(nn.Module):
    """Local CNN over joint (guide, site) position tokens, then a small transformer.

    Sizes are a starting point, not a claim: the number of independent sgRNAs and
    studies here does not justify a large model (roadmap section 7).
    """

    def __init__(self, d_model: int = 128, n_cnn: int = 3, n_tf: int = 2,
                 n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(N_PAIR_TOKEN, d_model)
        self.pos = nn.Embedding(SITE_LEN, d_model)
        self.mm_proj = nn.Linear(2, d_model)
        cnn = []
        for _ in range(n_cnn):
            cnn += [nn.Conv1d(d_model, d_model, 5, padding=2),
                    nn.GELU(), nn.BatchNorm1d(d_model), nn.Dropout(dropout)]
        self.cnn = nn.Sequential(*cnn)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, n_tf, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, mismatch: torch.Tensor) -> torch.Tensor:
        """tokens (B, 23) int; mismatch (B, 20) bool -> (B, d_model)."""
        b, L = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device))[None]
        # mismatch channel, padded over the PAM positions, plus distance to the PAM
        mm = torch.zeros(b, L, device=tokens.device, dtype=x.dtype)
        mm[:, : mismatch.shape[1]] = mismatch.to(x.dtype)
        dist = (torch.arange(L, device=tokens.device, dtype=x.dtype) / L)[None].expand(b, L)
        x = x + self.mm_proj(torch.stack([mm, dist], -1))
        x = x + self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.tf(x)
        # attention-free pooling: mean over positions plus the PAM token
        return self.norm(x.mean(1) + x[:, -1])


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


class AssayAwareModel(nn.Module):
    """Activity branches per context + a constrained observation branch.

    propensity_mode:
      ``constrained``  propensity sees pre-assay summaries + assay/study embeddings
      ``constant``     propensity is one scalar per assay (no feature dependence)
      ``free``         propensity sees the full encoder representation (the ablation
                       that should degrade identifiability)
    """

    N_CONTEXT = 2  # 0 = in_vitro, 1 = in_cell

    def __init__(self, n_assay: int, n_study: int, n_tab: int,
                 d_model: int = 128, d_assay: int = 16, d_study: int = 8,
                 propensity_mode: str = "constrained",
                 model_false_positive: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder = PairEncoder(d_model=d_model, dropout=dropout)
        self.propensity_mode = propensity_mode
        self.model_false_positive = model_false_positive

        # --- activity: one head per context, shared trunk -------------------
        self.activity = nn.ModuleList([
            MLPHead(d_model + n_tab, 64, 1, dropout) for _ in range(self.N_CONTEXT)])

        # --- observation ----------------------------------------------------
        self.assay_emb = nn.Embedding(n_assay, d_assay)
        self.study_emb = nn.Embedding(n_study, d_study)
        nn.init.zeros_(self.study_emb.weight)  # shrink studies toward the grand mean
        if propensity_mode == "constrained":
            d_prop = n_tab + d_assay + d_study
        elif propensity_mode == "free":
            d_prop = d_model + n_tab + d_assay + d_study
        elif propensity_mode == "constant":
            d_prop = d_assay
        else:
            raise ValueError(propensity_mode)
        self.propensity = MLPHead(d_prop, 32, 1, dropout)
        # false-positive rate: per assay only, and pushed low by its prior
        self.fp_logit = nn.Parameter(torch.full((n_assay,), -6.0))

    # -- pieces ------------------------------------------------------------

    def represent(self, tokens, mismatch):
        return self.encoder(tokens, mismatch)

    def activity_logit(self, h, tab, context):
        """context: (B,) long in {0, 1}."""
        x = torch.cat([h, tab], -1)
        outs = torch.stack([head(x).squeeze(-1) for head in self.activity], -1)
        return outs.gather(-1, context[:, None]).squeeze(-1)

    def propensity_logit(self, h, tab, assay, study):
        a = self.assay_emb(assay)
        if self.propensity_mode == "constant":
            x = a
        elif self.propensity_mode == "constrained":
            x = torch.cat([tab, a, self.study_emb(study)], -1)
        else:
            x = torch.cat([h, tab, a, self.study_emb(study)], -1)
        return self.propensity(x).squeeze(-1)

    # -- forward -----------------------------------------------------------

    def forward(self, tokens, mismatch, tab, context, assay, study):
        h = self.represent(tokens, mismatch)
        log_p = F.logsigmoid(self.activity_logit(h, tab, context))
        log_e = F.logsigmoid(self.propensity_logit(h, tab, assay, study))
        p = log_p.exp()
        e = log_e.exp()
        if self.model_false_positive:
            f = torch.sigmoid(self.fp_logit[assay])
        else:
            f = torch.zeros_like(p)
        # P(report) = p*e + (1-p)*f
        q = p * e + (1 - p) * f
        return dict(p_activity=p, e_propensity=e, f_falsepos=f,
                    q_report=q.clamp(1e-7, 1 - 1e-7),
                    h=h, activity_logit=self.activity_logit(h, tab, context))

    def hierarchical_penalty(self) -> torch.Tensor:
        """Shrink study effects so small studies cannot fit unstable parameters
        (roadmap section 7D), and keep the false-positive rate near zero unless the
        data insist otherwise."""
        return (self.study_emb.weight.pow(2).sum()
                + 0.1 * (self.fp_logit + 6.0).pow(2).sum())


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def observation_nll(q: torch.Tensor, s: torch.Tensor,
                    weight: torch.Tensor | None = None) -> torch.Tensor:
    """Bernoulli NLL of the *report*, which is what is actually observed."""
    ll = s * torch.log(q) + (1 - s) * torch.log1p(-q)
    if weight is not None:
        ll = ll * weight
    return -ll.mean()


def validation_nll(activity_logit: torch.Tensor, y: torch.Tensor,
                   weight: torch.Tensor) -> torch.Tensor:
    """Confirmatory-assay term: a direct observation of the latent cellular event.

    Weighted by the inverse probability that the site was selected for validation.
    The weight matters: sites were sent for confirmation because they looked
    promising, so an unweighted fit would inherit that selection.
    """
    return (F.binary_cross_entropy_with_logits(
        activity_logit, y, reduction="none") * weight).mean()


def nnpu_loss(logit: torch.Tensor, is_positive: torch.Tensor, prior: float,
              beta: float = 0.0) -> torch.Tensor:
    """Non-negative PU risk estimator (Kiryo et al. 2017).

        R = pi * E_P[l(f, +1)] + max(0, E_U[l(f, -1)] - pi * E_P[l(f, -1)])

    with the logistic loss. This is the risk estimator itself, with beta = 0 and
    without the gradient-reversal refinement (gamma) of the original paper's
    algorithm: when the empirical negative risk goes below zero the term is clamped
    rather than stepped against. The clamp is what prevents the unbounded negative
    risk that plain unbiased PU suffers from, which is the property the baseline needs.

    Used for the level-2 baseline. It assumes positives are labelled completely at
    random, which is exactly the assumption this paper argues is false here, so it is
    reported as a constrained baseline rather than as the proposed method, and its
    sensitivity to the assumed class prior is swept rather than fixed.
    """
    pos = is_positive.bool()
    sp = F.softplus(-logit)   # -log sigmoid(logit)
    sn = F.softplus(logit)    # -log(1 - sigmoid(logit))
    r_p_plus = sp[pos].mean() if pos.any() else logit.sum() * 0
    r_p_minus = sn[pos].mean() if pos.any() else logit.sum() * 0
    r_u_minus = sn[~pos].mean() if (~pos).any() else logit.sum() * 0
    neg_risk = r_u_minus - prior * r_p_minus
    if neg_risk < -beta:
        return prior * r_p_plus - beta
    return prior * r_p_plus + neg_risk


# ---------------------------------------------------------------------------
# simple reference scorers
# ---------------------------------------------------------------------------

#: Cutting Frequency Determination (Doench et al. 2016) requires the published
#: mismatch-penalty table; where it is unavailable the mismatch-count and
#: position-weighted scores below serve as the transparent sequence-only baselines.

def position_weighted_score(mismatch: torch.Tensor) -> torch.Tensor:
    """Monotone baseline: mismatches weighted by proximity to the PAM."""
    L = mismatch.shape[1]
    w = torch.linspace(0.2, 1.0, L, device=mismatch.device)
    return -(mismatch.to(w.dtype) * w).sum(1)
