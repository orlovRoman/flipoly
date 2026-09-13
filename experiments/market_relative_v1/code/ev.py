"""ev.py v1.0.0 — signal economics (spec v1.0.4, T19-T21 frozen).

EV rule (point):   gross_ev_yes = p_final - ask_yes
                   net_ev_yes   = gross_ev_yes - fee_per_share(ask_yes)
                   (NO mirrored with 1 - p_final, ask_no)
Uncertainty: one-sided 95% Wilson on OOF equal-frequency bins (10, min 200,
merge neighbours). YES lower = Wilson lower; NO lower = 1 - Wilson upper.
Signal: eligible sides = {net_edge_lower_bound > 0}; pick eligible side with
max POINT net EV; else ABSTAIN. Point and bound stored separately.
Historical execution: top-of-book ask + 0.5% adverse (slip = 0.005*exec*shares).
Forward execution: depth VWAP (no extra slippage).
"""
import numpy as np

import costs as C

EV_CODE_VERSION = "v1.0.0"
Z_95_ONE_SIDED = 1.6448536269514722
N_BINS = 10
MIN_BIN_OBS = 200
HIST_SLIP_PCT = 0.005


def wilson_bounds(k, n, z=Z_95_ONE_SIDED):
    if n <= 0:
        return 0.0, 1.0
    ph = k / n
    den = 1.0 + z * z / n
    center = (ph + z * z / (2.0 * n)) / den
    half = z * np.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def build_bins(oof_prob, oof_y, n_bins=N_BINS, min_obs=MIN_BIN_OBS):
    p = np.asarray(oof_prob, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p, qs)
    edges[0], edges[-1] = 0.0, 1.0
    edges = _merge_small(p, edges, min_obs)
    return edges


def _merge_small(p, edges, min_obs):
    edges = list(edges)
    while True:
        counts, _ = np.histogram(p, bins=np.array(edges))
        if bool((counts >= min_obs).all()) or len(edges) <= 3:
            break
        i = int(np.argmin(counts))
        if i == 0:
            del edges[1]
        elif i == len(counts) - 1:
            del edges[-2]
        else:
            if counts[i - 1] <= counts[i + 1]:
                del edges[i]
            else:
                del edges[i + 1]
    return np.array(edges)


def bin_table(probs, y, edges):
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, len(edges) - 2)
    table = []
    for b in range(len(edges) - 1):
        m = idx == b
        k = float(np.sum(y[m]))
        n = int(np.sum(m))
        lo, hi = wilson_bounds(k, n)
        table.append({"lo": lo, "hi": hi, "n": n, "k": k,
                      "lo_edge": float(edges[b]), "hi_edge": float(edges[b + 1])})
    return table


def side_lowers(p_final, edges, table):
    b = int(np.clip(np.digitize([p_final], edges[1:-1], right=False)[0], 0, len(table) - 1))
    t = table[b]
    return t["lo"], 1.0 - t["hi"], b


def decide_row(p_final, yes_ask, no_ask, edges, table):
    out = {"p_final": float(p_final), "side": None, "reason": "ABSTAIN",
           "point_yes": np.nan, "point_no": np.nan,
           "lower_yes": np.nan, "lower_no": np.nan, "bin": -1}
    if not (0.0 < yes_ask < 1.0 and 0.0 < no_ask < 1.0):
        out["reason"] = "ABSTAIN_BAD_PRICE"
        return out
    if not (0.0 < p_final < 1.0):
        out["reason"] = "ABSTAIN_BAD_PRED"
        return out
    ly, ln, b = side_lowers(p_final, edges, table)
    py = p_final - yes_ask - C.fee_per_share(yes_ask)
    pn = (1.0 - p_final) - no_ask - C.fee_per_share(no_ask)
    ey = ly - yes_ask - C.fee_per_share(yes_ask)
    en = ln - no_ask - C.fee_per_share(no_ask)
    out.update({"point_yes": py, "point_no": pn, "lower_yes": ey, "lower_no": en, "bin": b})
    cands = []
    if ey > 0:
        cands.append(("YES", py))
    if en > 0:
        cands.append(("NO", pn))
    if not cands:
        out["reason"] = "ABSTAIN_NEGATIVE_EDGE"
        return out
    cands.sort(key=lambda t: t[1], reverse=True)
    out["side"] = cands[0][0]
    out["reason"] = "TRADE_" + cands[0][0]
    return out


def realized_hist(side, p_market, yes_ask, no_ask, won_yes):
    """Historical realized economics: top-of-book + 0.5% adverse slippage."""
    if side == "YES":
        payout, ref, exe = (1.0 if won_yes else 0.0), p_market, yes_ask
    else:
        payout, ref, exe = (0.0 if won_yes else 1.0), 1.0 - p_market, no_ask
    t = C.trade_pnl(payout, ref, exe)
    slip_extra = t["shares"] * exe * HIST_SLIP_PCT
    canon = t["canon_net"] - slip_extra
    return {"raw": t["raw"], "executable": t["executable"], "slippage": t["slippage"],
            "fee07": t["fee07"], "slip_extra": slip_extra, "canon_net": canon,
            "shares": t["shares"]}


def realized_forward(side, p_market, vwap, won_yes):
    """Forward realized economics: depth VWAP, no extra slippage."""
    if side == "YES":
        payout, ref = (1.0 if won_yes else 0.0), p_market
    else:
        payout, ref = (0.0 if won_yes else 1.0), 1.0 - p_market
    t = C.trade_pnl(payout, ref, vwap)
    return {"raw": t["raw"], "executable": t["executable"], "slippage": t["slippage"],
            "fee07": t["fee07"], "canon_net": t["canon_net"], "shares": t["shares"]}
