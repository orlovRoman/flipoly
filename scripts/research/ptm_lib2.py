"""PTM v2 shared contracts (Stage 1 fixes).

Item 5: entry target explicit; item 8: fee/gross/net separation;
item 9: closed bin set incl. out-of-grid categories;
item 10: unified role with recorded basis; item 4: opportunity_id.
"""
import hashlib
import json

PRICE_BINS = [
    (0.01, 0.05, "[0.01,0.05)"),
    (0.05, 0.10, "[0.05,0.10)"),
    (0.10, 0.15, "[0.10,0.15)"),
    (0.15, 0.20, "[0.15,0.20)"),
    (0.20, 0.25, "[0.20,0.25)"),
    (0.25, 0.30, "[0.25,0.30)"),
    (0.30, 0.35, "[0.30,0.35)"),
    (0.35, 0.40, "[0.35,0.40)"),
    (0.40, 0.50, "[0.40,0.50)"),
    (0.50, 0.60, "[0.50,0.60)"),
    (0.60, 0.70, "[0.60,0.70)"),
    (0.70, 0.80, "[0.70,0.80)"),
    (0.80, 0.90, "[0.80,0.90)"),
    (0.90, 0.99, "[0.90,0.99]"),
]

ENTRY_TARGETS_MAIN = (12, 8, 5)
ENTRY_DELAY_MAX_SEC = 30.0
SCENARIO_FEE_RATE = 0.02  # SCENARIO only; fee_status=UNKNOWN


def bin_of(price):
    """Every included row gets exactly one bin.

    Returns (label, status): status OK, OUT_OF_RANGE_HIGH (>0.99),
    OUT_OF_RANGE_LOW (<0.01), INVALID (non-finite/None).
    """
    if price is None:
        return None, "MISSING_PRICE"
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None, "INVALID_PRICE"
    import math
    if not math.isfinite(p):
        return None, "INVALID_PRICE"
    if p < 0.01:
        return "BELOW_0.01", "OUT_OF_RANGE_LOW"
    if p > 0.99:
        return "ABOVE_0.99", "OUT_OF_RANGE_HIGH"
    for lo, hi, label in PRICE_BINS:
        if lo <= p < hi or (label == "[0.90,0.99]" and p == hi):
            return label, "OK"
    return None, "INVALID_PRICE"


def region_of(ask):
    if ask is None:
        return None
    if ask <= 0.30:
        return "deep_outsider"
    if ask <= 0.40:
        return "cheap_outsider"
    if 0.40 < ask < 0.60:
        return "near_parity"
    if ask >= 0.60:
        return "favorite"
    return None


def classify_role(mid, ask):
    """Unified role for GRID and FUNNEL (item 10).

    mid<=0.5 OUTSIDER else FAVORITE. If mid is missing but ask exists,
    classify by own ask and mark basis=ASK_ONLY.
    Returns (role, basis) with basis in MID / ASK_ONLY / UNKNOWN.
    """
    if mid is not None:
        try:
            m = float(mid)
            import math
            if math.isfinite(m):
                return ("OUTSIDER" if m <= 0.5 else "FAVORITE", "MID")
        except (TypeError, ValueError):
            pass
    if ask is not None:
        try:
            a = float(ask)
            import math
            if math.isfinite(a):
                return ("OUTSIDER" if a <= 0.5 else "FAVORITE", "ASK_ONLY")
        except (TypeError, ValueError):
            pass
    return None, "UNKNOWN"


def select_entry(sorted_snaps, target_at):
    """Item 5: first observation in [target_at, target_at+30s].

    Returns (row, status, reason, delay_sec). decision_at := row.recorded_at.
    Rows before target_at are rejected (too early); delay > 30s rejected.
    """
    for row in sorted_snaps:
        rec = row.get("recorded_at")
        if rec is None:
            continue
        delay = (rec - target_at).total_seconds()
        if delay < 0:
            continue  # before the boundary: reject
        if delay <= ENTRY_DELAY_MAX_SEC:
            return row, "OK", "", delay
        # first row at/after boundary is already too late -> nothing later qualifies
        return None, "MISSING_ENTRY_QUOTE", "LATE_ENTRY_GT_30S", None
    return None, "MISSING_ENTRY_QUOTE", "NO_OBSERVATION_IN_WINDOW", None


def pnl_row(entry_price, won, stake_usdc=1.0, fee_rate=None, fee_status="UNKNOWN"):
    """Item 8: unknown fee -> fee=None, net_pnl=None. Gross always computed.

    gross == target/ask - 1 per unit stake (won: shares*1 - stake).
    Scenario 2% on $1 stake reduces the result by exactly $0.02.
    """
    shares = stake_usdc / entry_price
    purchase_cash = shares * entry_price
    gross = shares * 1.0 - purchase_cash if won else -purchase_cash
    if fee_rate is None:
        return shares, purchase_cash, gross, None, None, fee_status
    fee = fee_rate * purchase_cash
    return shares, purchase_cash, gross, fee, gross - fee, "SCENARIO"


def opportunity_id(market_id, layer, entry_rule, side, recorded_at_iso):
    """Item 4: stable opportunity key."""
    raw = "|".join([str(market_id), layer, entry_rule, side, str(recorded_at_iso)])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def config_hash(config):
    """Item 2: run_id excluded by convention (callers must not include it)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
