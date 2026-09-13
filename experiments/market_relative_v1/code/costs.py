"""cost_calculator v1.0.0 — canonical taker economics (spec v1.0.4).

fee = shares * 0.07 * price * (1 - price)   (POLYMARKET_PRICE_DEPENDENT)
PnL levels per $1 gross stake (shares = 1 / exec_price):
  raw        = shares * (payout - reference_price)
  executable = shares * (payout - vwap_exec_price)
  canon_net  = executable - fee(shares, vwap_exec_price)
No PAPER 0.002 anywhere in this module (diagnostic appendix only, elsewhere).
"""
FEE_RATE = 0.07
COST_VERSION = "v1.0.0"


def fee_per_share(price):
    return FEE_RATE * price * (1.0 - price)


def fee(shares, price):
    return shares * fee_per_share(price)


def walk_asks(asks, budget=1.0):
    """Walk ask levels [(price, size), ...] spending up to budget gross.
    Levels sorted by price ascending (normalizes UNORDERED_LEVELS).
    Returns dict(shares, vwap, spent, shortfall, levels_used).
    shortfall True => ABSTAIN_INSUFFICIENT_LIQUIDITY (partial fills forbidden).
    """
    levels = sorted(((float(p), float(s)) for p, s in asks
                     if p == p and s == s and p > 0 and s > 0),
                    key=lambda t: t[0])
    spent = 0.0
    shares = 0.0
    used = 0
    for price, size in levels:
        if spent >= budget:
            break
        take_shares = min(size, (budget - spent) / price)
        spent += take_shares * price
        shares += take_shares
        used += 1
    vwap = spent / shares if shares > 0 else float("nan")
    return {"shares": shares, "vwap": vwap, "spent": spent,
            "shortfall": spent < budget - 1e-12, "levels_used": used}


def level_pnl(payout01, price, shares):
    return shares * (payout01 - price)


def trade_pnl(payout01, reference_price, exec_price):
    """Three canonical levels for a $1-stake trade. Returns dict."""
    shares = 1.0 / exec_price
    raw = level_pnl(payout01, reference_price, shares)
    exe = level_pnl(payout01, exec_price, shares)
    f = fee(shares, exec_price)
    return {"shares": shares, "raw": raw, "executable": exe,
            "slippage": exe - raw, "fee07": f, "canon_net": exe - f}
