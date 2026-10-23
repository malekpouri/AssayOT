"""Published off-target architectures, reimplemented and trained on this corpus.

Why reimplement rather than download weights
--------------------------------------------
A referee is right to ask how the assay-aware model compares with the learned
predictors the field actually uses. Taking those models' *released weights* answers a
different and less useful question, because they were trained on other corpora under
the labelling convention this paper is arguing about: any gap would confound
architecture, training data and label policy, and would flatter whichever model had
seen more of the evaluation distribution.

So the architectures are reimplemented and fitted here, on the same folds, with the
same positives and the same sampled unreported negatives as `L1_supervised`. That is
the convention these models are published with -- a reported detection is a positive,
every other candidate is a negative -- and holding it fixed isolates the one thing the
comparison is about: does the architecture, or the observation model, carry the result?

The two families cover the published design space:

`cnn_bilstm`
    The convolution-plus-recurrent family (CRISPR-Net, CRISPR-IP and relatives). A
    guide-site pair is encoded per position as the two bases plus a mismatch flag,
    passed through a convolutional stack, read by a bidirectional LSTM, and pooled.

`gbt`
    The feature-engineering family (Elevation and the CFD successors). Position-by-
    substitution indicators plus the pre-assay tabular features, fitted with gradient
    boosting. No sequence model: the mismatch identity *is* the representation.

Neither is tuned here. They are given the same budget as the paper's own models, and
the point of the comparison is the ordering, not a leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import config as C, encode as E
except ImportError:  # pragma: no cover
    from src import config as C, encode as E

DEV = "cuda" if torch.cuda.is_available() else "cpu"

N_POS_CH = E.N_BASE          # guide base one-hot
N_SITE_CH = E.N_BASE         # site base one-hot
SITE_LEN = E.SITE_LEN


# ---------------------------------------------------------------------------
# shared training-set construction
# ---------------------------------------------------------------------------

def training_rows(frame, train_guides, ratio: int = 4, seed: int = 0) -> pd.DataFrame:
    """Positives plus a fixed random sample of unreported rows, from train guides only.

    This mirrors `Batcher` but materialises one fixed sample instead of resampling per
    epoch, because neither reference model has an epoch loop over the full candidate
    space and both must see an identical training set.
    """
    tr = frame.obs[frame.obs.guide_sample.isin(train_guides)]
    pos = tr[tr.s > 0]
    unl = tr[tr.s == 0]
    k = min(len(unl), ratio * max(1, len(pos)))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(unl), size=k, replace=False)
    return pd.concat([pos, unl.iloc[pick]], ignore_index=True)


# ---------------------------------------------------------------------------
# family 1: convolution + bidirectional LSTM
# ---------------------------------------------------------------------------

def _pair_channels(guide_seqs, site_seqs) -> np.ndarray:
    """(n, 13, 23) float32: guide one-hot, site one-hot, and a mismatch flag.

    The mismatch channel is redundant given the two one-hots, and is included because
    the published encodings include it: it hands the convolution the comparison it
    would otherwise have to learn.
    """
    g = E.encode_seqs(guide_seqs, E.SPACER_LEN)
    s = E.encode_seqs(site_seqs, SITE_LEN)
    n = len(s)
    # pad the 20-nt spacer to the 23-nt site frame so both align on the PAM
    gp = np.full((n, SITE_LEN), E.N_BASE - 2, dtype=np.uint8)
    gp[:, :E.SPACER_LEN] = g
    out = np.zeros((n, N_POS_CH + N_SITE_CH + 1, SITE_LEN), dtype=np.float32)
    rows = np.arange(n)[:, None]
    cols = np.arange(SITE_LEN)[None, :]
    out[rows, gp.astype(np.int64), cols] = 1.0
    out[rows, N_POS_CH + s.astype(np.int64), cols] = 1.0
    mm = np.zeros((n, SITE_LEN), dtype=np.float32)
    mm[:, :E.SPACER_LEN] = (g != s[:, :E.SPACER_LEN]).astype(np.float32)
    out[:, -1, :] = mm
    return out


class CnnBiLstm(nn.Module):
    """Convolutional stack feeding a bidirectional LSTM, as in the CRISPR-Net family."""

    def __init__(self, in_ch: int = N_POS_CH + N_SITE_CH + 1, width: int = 64,
                 lstm: int = 48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.lstm = nn.LSTM(width, lstm, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(4 * lstm, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)        # (n, L, width)
        o, _ = self.lstm(h)
        pooled = torch.cat([o.mean(1), o.max(1).values], dim=1)
        return self.head(pooled).squeeze(-1)


def fit_predict_cnn_bilstm(frame, train_guides, test_df: pd.DataFrame,
                           epochs: int = 6, ratio: int = 4, batch: int = 1024,
                           lr: float = 1e-3, seed: int = 0) -> np.ndarray:
    """Fit on the training sgRNAs and score `test_df`.

    The batch is deliberately smaller than the paper's own models use. Under a
    one-cycle schedule the step count is `epochs * rows / batch`, and at batch 8192
    a small training fold gets so few steps that the network returns an almost
    constant score -- which would be reported as a weak baseline when it is really an
    unoptimised one. Giving the comparator more steps costs little and removes the
    suspicion that it was handicapped.
    """
    torch.manual_seed(seed)
    tr = training_rows(frame, train_guides, ratio=ratio, seed=seed)
    X = _pair_channels(tr.guide_sequence.to_numpy(dtype=object),
                       tr.site_sequence.to_numpy(dtype=object))
    y = tr.s.to_numpy(dtype=np.float32)
    model = CnnBiLstm().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y)
    rng = np.random.default_rng(seed)
    steps = max(1, epochs * (n // batch + 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb = torch.from_numpy(X[idx]).to(DEV)
            yb = torch.from_numpy(y[idx]).to(DEV)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
    model.eval()
    Xt = _pair_channels(test_df.guide_sequence.to_numpy(dtype=object),
                        test_df.site_sequence.to_numpy(dtype=object))
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            xb = torch.from_numpy(Xt[i:i + batch]).to(DEV)
            out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    scores = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    _warn_if_degenerate("cnn_bilstm", scores)
    return scores


# ---------------------------------------------------------------------------
# family 2: gradient boosting on position-by-substitution indicators
# ---------------------------------------------------------------------------

def _elevation_features(guide_seqs, site_seqs) -> np.ndarray:
    """Position x substitution indicators plus the pre-assay tabular features.

    The indicator block is what the CFD lineage scores with a lookup table and what
    Elevation learns over: for each of the 20 spacer positions, which guide->site
    substitution occurs there. 20 positions x 16 ordered base pairs = 320 columns,
    plus the 11 tabular features.
    """
    g = E.encode_seqs(guide_seqs, E.SPACER_LEN)
    s = E.encode_seqs(site_seqs, SITE_LEN)
    n = len(s)
    sub = np.zeros((n, E.SPACER_LEN * 16), dtype=np.float32)
    gi = g.astype(np.int64)
    si = s[:, :E.SPACER_LEN].astype(np.int64)
    valid = (gi < 4) & (si < 4)
    pos = np.broadcast_to(np.arange(E.SPACER_LEN)[None, :], gi.shape)
    flat = pos * 16 + gi * 4 + si
    rows = np.broadcast_to(np.arange(n)[:, None], gi.shape)
    sub[rows[valid], flat[valid]] = 1.0
    tab, _ = E.tabular_pre_assay(g, s)
    return np.hstack([sub, tab])


def _warn_if_degenerate(name: str, scores: np.ndarray, tol: float = 1e-3) -> None:
    """Shout if a comparator collapsed to a near-constant score.

    A baseline that failed to optimise ranks almost at chance, and quoting that as the
    published architecture's performance would understate it. This makes the failure
    loud instead of letting it be read as a result.
    """
    if len(scores) and float(scores.std()) < tol:
        print(f"  !! {name}: score s.d. {scores.std():.2e} over {len(scores)} rows -- "
              "the model did not optimise; do not report this as a baseline",
              flush=True)


def fit_predict_gbt(frame, train_guides, test_df: pd.DataFrame,
                    ratio: int = 4, seed: int = 0) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = training_rows(frame, train_guides, ratio=ratio, seed=seed)
    X = _elevation_features(tr.guide_sequence.to_numpy(dtype=object),
                            tr.site_sequence.to_numpy(dtype=object))
    y = tr.s.to_numpy(dtype=np.float32)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
        early_stopping=False, random_state=seed)
    clf.fit(X, y)
    Xt = _elevation_features(test_df.guide_sequence.to_numpy(dtype=object),
                             test_df.site_sequence.to_numpy(dtype=object))
    scores = clf.predict_proba(Xt)[:, 1].astype(np.float32)
    _warn_if_degenerate("gbt_elevation", scores)
    return scores


REFERENCE_MODELS = {
    "R1_cnn_bilstm": fit_predict_cnn_bilstm,
    "R2_gbt_elevation": fit_predict_gbt,
}
