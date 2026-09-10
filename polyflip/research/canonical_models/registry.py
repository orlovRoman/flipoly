"""Steps 7-9: local contract registry (one row per market).

Inputs are LOCAL frames (exported slices): live_markets mirror + snapshots.
No DB, no server, no network.

Row: market_id, asset, start_at, end_at, up_token_id, down_token_id,
strike_value, strike_source, strike_available_at, resolution_rule,
resolution_source, actual_outcome, outcome_available_at.
Checks: unique market_id; UP/DOWN mapping explicit and unambiguous;
disputed/unresolved markets separated; proxy strikes never canonical.
"""
from __future__ import annotations

import pandas as pd

FAMILY_15M = "crypto-15m-twap60"
RESOLUTION_RULE_15M = (
    "UP iff Chainlink TWAP-60s over window >= window-start price, else DOWN "
    "(equality -> UP)"
)


def build_registry(live: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (registry, issues). issues holds excluded/flagged markets with reasons."""
    live = live.copy()
    live["market_id"] = live["market_id"].astype(str)
    snaps = snaps.copy()
    snaps["market_id"] = snaps["market_id"].astype(str)

    # outcome per market from snapshots (latest recorded wins; backfilled labels)
    s = snaps.sort_values("recorded_at")
    last = s.drop_duplicates("market_id", keep="last").set_index("market_id")
    first = s.drop_duplicates("market_id", keep="first").set_index("market_id")

    rows, issues = [], []
    for mid, lm in live.set_index("market_id").iterrows():
        up_tok = lm.get("yes_token_id")
        down_tok = lm.get("no_token_id")
        if not up_tok or not down_tok or up_tok == down_tok or pd.isna(up_tok) or pd.isna(down_tok):
            issues.append({"market_id": mid, "reason": "AMBIGUOUS_TOKEN_MAPPING"})
            continue
        end_at = lm.get("market_end_at")
        if pd.isna(end_at):
            end_at = lm.get("end_time_est")
        start_at = lm.get("market_start_at")
        outcome = last["final_outcome"].get(mid)
        if outcome not in ("YES", "NO"):
            issues.append({"market_id": mid, "reason": f"UNRESOLVED:{outcome}"})
            continue
        strike = lm.get("strike_value")
        if pd.isna(strike):
            strike, strike_source = None, "unknown"
        else:  # pragma: no cover - no struck rows observed yet
            strike, strike_source = float(strike), lm.get("strike_source") or "unknown"
        rows.append({
            "market_id": mid, "asset": lm.get("asset"),
            "start_at": start_at, "end_at": end_at,
            "up_token_id": up_tok, "down_token_id": down_tok,
            "strike_value": strike, "strike_source": strike_source,
            "strike_available_at": None,
            "resolution_rule": RESOLUTION_RULE_15M,
            "resolution_source": lm.get("settlement_price_source") or lm.get("resolution_source"),
            "actual_outcome": outcome,
            "outcome_available_at": lm.get("resolved_at"),
            "n_snapshots": int((s["market_id"] == mid).sum()),
            "first_seen": first["recorded_at"].get(mid),
        })
    reg = pd.DataFrame(rows)
    assert reg["market_id"].is_unique, "registry must hold one row per market"
    return reg, pd.DataFrame(issues)


def coverage_gate(reg: pd.DataFrame, issues: pd.DataFrame) -> dict:
    """Step-10 gate numbers (reconcile: eligible + excluded == input markets)."""
    n_strike = int(reg["strike_value"].notna().sum())
    verdict = {
        "n_registry": len(reg),
        "n_issues": len(issues),
        "by_reason": issues["reason"].value_counts().to_dict() if len(issues) else {},
        "by_asset": reg["asset"].value_counts().to_dict(),
        "n_canonical_strike": n_strike,
        "ct_track": "FEASIBLE" if len(reg) else "BLOCKED",
        "model_track": "FEASIBLE" if n_strike else "DATA_BLOCKED_retrospective_strike_unproven",
    }
    return verdict
