"""metrics.py v1.0.0 — predictive + economic metrics (spec v1.0.4)."""
import numpy as np

import train as T

METRICS_VERSION = "v1.0.0"


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0], qs[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, qs[1:-1], right=False), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if np.sum(m) == 0:
            continue
        tot += np.sum(m) * abs(float(np.mean(y[m])) - float(np.mean(p[m])))
    return tot / len(y)


def cal_slope_intercept(y, p):
    """Logistic fit y ~ logit(p): returns (intercept, slope) via Newton, no penalty.
    Constant predictions => singular system => (nan, nan); gates treat nan as FAIL.
    """
    import numpy.linalg
    y = np.asarray(y, dtype=np.float64)
    lp = T.logit(np.asarray(p, dtype=np.float64)).reshape(-1, 1)
    try:
        beta = T.irls_offset(lp, y, np.zeros(len(y)), lam=0.0, max_iter=500, tol=1e-10)
    except numpy.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def max_drawdown(equity):
    e = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(e)
    return float(np.min(e - peak))


def block_bootstrap_ci(values, days, n_reps=2000, seed=20260912):
    """Day-block bootstrap CI of the mean. values aligned with days array."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=np.float64)
    days = np.asarray(days)
    uniq = np.unique(days)
    per_day = [values[days == d] for d in uniq]
    means = np.empty(n_reps)
    for b in range(n_reps):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        pooled = np.concatenate([per_day[i] for i in pick])
        means[b] = pooled.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def positive_share(parts):
    """Max share of positive total from one part. parts: dict name->total."""
    pos = {k: max(0.0, float(v)) for k, v in parts.items()}
    tot = sum(pos.values())
    if tot <= 0:
        return 1.0, None
    k = max(pos, key=lambda k: pos[k])
    return pos[k] / tot, k
