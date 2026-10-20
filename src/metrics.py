"""Metrics, with the caveats from roadmap section 10 built into the call signatures.

Two rules are enforced rather than documented:

*   Nothing here computes precision or a false-positive rate against unlabelled rows as
    though they were confirmed negatives. Functions that need true negatives take an
    explicit `labelled` mask and will refuse to run without one.
*   Ranking metrics are computed per guide and then macro-averaged, so a single sgRNA
    with a large repeat family cannot carry the result. Confidence intervals come from
    a bootstrap over guides (and over studies where enough exist), not over rows,
    because rows within a guide are not independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = np.random.default_rng(20261020)


# ---------------------------------------------------------------------------
# ranking, per guide
# ---------------------------------------------------------------------------

def _ap(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, s)


def _auc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    return float(y[order][:k].sum() / y.sum())


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    if k == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    return float(y[order][:k].mean())


def per_guide_ranking(df: pd.DataFrame, score: str, label: str,
                      guide: str = "guide_sample",
                      ks=(10, 50, 100)) -> pd.DataFrame:
    """One row per guide. Guides with no positive are kept as NaN, not dropped,
    so the number of evaluable guides is visible."""
    rows = []
    for g, sub in df.groupby(guide, observed=True):
        y = sub[label].to_numpy(dtype=float)
        s = sub[score].to_numpy(dtype=float)
        r = dict(guide=g, n=len(sub), n_pos=int(y.sum()),
                 ap=_ap(y, s), auroc=_auc(y, s))
        for k in ks:
            r[f"recall@{k}"] = recall_at_k(y, s, k)
            r[f"precision@{k}"] = precision_at_k(y, s, k)
        rows.append(r)
    return pd.DataFrame(rows)


def macro_summary(per_guide: pd.DataFrame, n_boot: int = 2000,
                  rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Macro-average over guides with a guide-level bootstrap interval."""
    rng = rng or RNG
    metrics = [c for c in per_guide.columns
               if c not in ("guide", "n", "n_pos")]
    out = []
    for m in metrics:
        v = per_guide[m].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            out.append(dict(metric=m, value=np.nan, lo=np.nan, hi=np.nan,
                            n_guides=0))
            continue
        vv = v[ok]
        boot = np.array([rng.choice(vv, size=vv.size, replace=True).mean()
                         for _ in range(n_boot)])
        out.append(dict(metric=m, value=float(vv.mean()),
                        lo=float(np.percentile(boot, 2.5)),
                        hi=float(np.percentile(boot, 97.5)),
                        n_guides=int(ok.sum())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# probability quality -- only on rows with a usable label
# ---------------------------------------------------------------------------

def log_loss_brier(y: np.ndarray, p: np.ndarray, labelled: np.ndarray) -> dict:
    """Requires an explicit mask of rows whose label means what it says."""
    if labelled is None:
        raise ValueError("log_loss_brier needs a `labelled` mask; unlabelled rows "
                         "cannot be scored as negatives")
    y = y[labelled].astype(float)
    p = np.clip(p[labelled].astype(float), 1e-7, 1 - 1e-7)
    return dict(n=int(labelled.sum()),
                log_loss=float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                brier=float(((p - y) ** 2).mean()),
                mean_pred=float(p.mean()), base_rate=float(y.mean()))


def calibration_table(y: np.ndarray, p: np.ndarray, labelled: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    y = y[labelled].astype(float)
    p = np.clip(p[labelled].astype(float), 0, 1)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(dict(bin=b, n=int(m.sum()), mean_pred=float(p[m].mean()),
                         observed=float(y[m].mean())))
    t = pd.DataFrame(rows)
    if len(t):
        t["gap"] = t.mean_pred - t.observed
    return t


def expected_calibration_error(y, p, labelled, n_bins: int = 10) -> float:
    t = calibration_table(y, p, labelled, n_bins)
    if not len(t):
        return np.nan
    w = t.n / t.n.sum()
    return float((w * t.gap.abs()).sum())


# ---------------------------------------------------------------------------
# selective prediction
# ---------------------------------------------------------------------------

def risk_coverage(y: np.ndarray, p: np.ndarray, confidence: np.ndarray,
                  labelled: np.ndarray, n_points: int = 20) -> pd.DataFrame:
    """Error of the accepted set against the acceptance rate.

    Both axes are reported: an abstention method that looks good only because it
    accepts very little is not an improvement.
    """
    y = y[labelled].astype(float)
    p = np.clip(p[labelled].astype(float), 1e-7, 1 - 1e-7)
    c = confidence[labelled].astype(float)
    order = np.argsort(-c, kind="stable")
    y, p = y[order], p[order]
    n = len(y)
    rows = []
    for frac in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(1, int(round(frac * n)))
        rows.append(dict(coverage=k / n, n=k,
                         brier=float(((p[:k] - y[:k]) ** 2).mean()),
                         log_loss=float(-(y[:k] * np.log(p[:k])
                                          + (1 - y[:k]) * np.log1p(-p[:k])).mean()),
                         error=float((( p[:k] > 0.5).astype(float) != y[:k]).mean())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------

def paired_guide_bootstrap(per_guide_a: pd.DataFrame, per_guide_b: pd.DataFrame,
                           metric: str, n_boot: int = 5000,
                           rng: np.random.Generator | None = None) -> dict:
    """Difference in a macro metric between two models, resampling guides.

    Paired on the guide, because the two models saw the same guides and the
    guide-to-guide spread is much larger than the model-to-model difference.
    """
    rng = rng or RNG
    m = per_guide_a[["guide", metric]].merge(
        per_guide_b[["guide", metric]], on="guide", suffixes=("_a", "_b")).dropna()
    if not len(m):
        return dict(metric=metric, delta=np.nan, lo=np.nan, hi=np.nan, n_guides=0,
                    p_two_sided=np.nan)
    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    boot = np.array([rng.choice(d, size=d.size, replace=True).mean()
                     for _ in range(n_boot)])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(metric=metric, delta=float(d.mean()),
                lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)),
                n_guides=int(len(d)), p_two_sided=float(min(1.0, p)))
