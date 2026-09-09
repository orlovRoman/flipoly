"""PTM shared library: bins, entry selection, PnL math, config hash.

No I/O, no DB, no ML. Pure functions with unit-testable contracts.
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
ENTRY_TARGETS_DIAG = (10, 9, 7, 6, 4)
ENTRY_WINDOW = 0.5

SCENARIO_FEE_RATE = 0.02  # SCENARIO only; fee_status=UNKNOWN (no historical evidence)


def bin_of(price):
    """Half-open [lo,hi) bins; 0.40 -> [0.40,0.50), 0.50 -> [0.50,0.60)."""
    if price is None:
        return None
    for lo, hi, label in PRICE_BINS:
        if lo <= price < hi or (label == "[0.90,0.99]" and price == hi):
            return label
    return None


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


def select_entry(sorted_snaps, target_min, decision_at=None):
    """First snapshot with time_left<=target inside [target-0.5, target+0.5].

    sorted_snaps: ascending recorded_at, each with time_left_min + recorded_at.
    Returns (row, status, reason). The decision is executed AT the selected
    observation (decision_at := row.recorded_at), so causality holds by
    construction: earliest in-window row, never a later one.
    """
    lo = target_min - ENTRY_WINDOW
    for row in sorted_snaps:
        tl = row.get("time_left_min")
        if tl is None:
            continue
        if tl <= target_min and tl >= lo:
            return row, "OK", ""
    return None, "MISSING_ENTRY_QUOTE", "NO_OBSERVATION_IN_WINDOW"


def pnl_row(entry_price, won, stake_usdc=1.0, fee_rate=None):
    """Per-plan PnL math. fee_rate None -> gross only (fee UNKNOWN)."""
    shares = stake_usdc / entry_price
    purchase_cash = shares * entry_price  # == stake_usdc by construction
    gross = shares * 1.0 - purchase_cash if won else -purchase_cash
    if fee_rate is None:
        return shares, purchase_cash, gross, 0.0, gross, "UNKNOWN"
    fee = fee_rate * purchase_cash
    return shares, purchase_cash, gross, fee, gross - fee, "SCENARIO"


def config_hash(config):
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
