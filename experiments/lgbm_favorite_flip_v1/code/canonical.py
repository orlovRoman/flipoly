"""canonical.py — frozen semantics for lgbm-favorite-flip-v1 (spec v1.0.0).

Target/side/execution definitions. No I/O, no model code.
"""
import numpy as np

FEE_RATE = 0.07
SL_RATE = 0.005
EPS = 1e-6


def favorite_side(yes_mid):
    """YES if yes_mid>0.5, NO if <0.5, AMBIGUOUS if exactly 0.5."""
    y = float(yes_mid)
    if not np.isfinite(y) or y <= 0.0 or y >= 1.0:
        raise ValueError("BAD_YES_MID: %r" % (yes_mid,))
    if y > 0.5:
        return "YES"
    if y < 0.5:
        return "NO"
    return "AMBIGUOUS"


def favorite_flip(fav, outcome):
    """Truth table: 1 iff current favorite lost. outcome must be YES/NO."""
    if outcome not in ("YES", "NO"):
        raise ValueError("BAD_OUTCOME: %r" % (outcome,))
    if fav not in ("YES", "NO"):
        raise ValueError("BAD_FAVORITE: %r" % (fav,))
    return int(fav != outcome)


def candidate_side(fav):
    """Single inversion: candidate is always the current outsider."""
    if fav == "YES":
        return "NO"
    if fav == "NO":
        return "YES"
    raise ValueError("BAD_FAVORITE: %r" % (fav,))


def market_flip(yes_mid, fav):
    """C0 market control p_market_flip + provenance.

    fav NO -> outsider YES mid observed; fav YES -> 1-yes_mid SYNTHETIC.
    """
    y = float(yes_mid)
    if fav == "NO":
        return y, "MID_OBSERVED"
    if fav == "YES":
        return 1.0 - y, "MID_SYNTHETIC"
    raise ValueError("BAD_FAVORITE: %r" % (fav,))


def exec_cost(ask):
    """Canonical all-in cost per share for $1 budget (symmetric YES/NO)."""
    a = float(ask)
    return a + FEE_RATE * a * (1.0 - a) + SL_RATE * a


def settle_net(candidate, won, ask):
    """Canonical net: win -> shares-1, loss -> -1. won is bool."""
    if won is None:
        return 0.0
    if bool(won):
        return 1.0 / exec_cost(ask) - 1.0
    return -1.0


def clip01(p):
    return float(np.clip(float(p), EPS, 1.0 - EPS))


def logit(p):
    p = clip01(p)
    return float(np.log(p / (1.0 - p)))


def sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-float(z))))


def apply_residual(p_market, residual):
    """L3 inference: sigmoid(logit(p_market) + residual). Zero -> identity."""
    return sigmoid(logit(p_market) + float(residual))


def trade_decision(p_flip, ask, min_edge=0.03):
    """DEVELOPMENT fixed-threshold rule: BUY iff edge > min_edge else SKIP."""
    edge = float(p_flip) - exec_cost(ask)
    return ("BUY" if edge > min_edge else "SKIP"), edge


def price_band(outsider_mid):
    """Frozen outsider-mid bands; None if out of [0.05, 0.50]."""
    x = float(outsider_mid)
    for lo, hi in ((0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50)):
        if lo <= x < hi or (hi == 0.50 and x == 0.50):
            return "%g-%g" % (lo, hi)
    return None
