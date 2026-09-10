"""Steps 34-35: fixed trading policies on identical opportunities.

Policies: price_control, CT, M1, challenger, challenger_plus_CT.
Model policies share ONE pre-registered EV-after-costs threshold
(protocol.economics.ev_threshold_net). Same time/price-range/budget/fill.
Divergent decisions keep reasons; the opportunity universe never changes.

Side selection (main test): outsiders of both sides -> pick the side with
the lower observed mid, then check its ask + policy gate. Parity ->
SKIP_PARITY. The model never picks the ex-post more profitable side;
unclear book mapping is excluded explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Opportunity:
    market_id: str
    up_mid: float
    down_mid: float
    up_ask: float | None
    down_ask: float | None
    p_up_market: float
    p_up_model: float | None
    ct_regime: str = "NO_DATA"


@dataclass(frozen=True)
class PolicyDecision:
    trade: bool
    side: str | None  # UP | DOWN
    ask: float | None
    reason: str


def pick_outsider_side(opp: Opportunity) -> tuple[str | None, str]:
    if opp.up_mid < opp.down_mid:
        return "UP", "lower-mid"
    if opp.down_mid < opp.up_mid:
        return "DOWN", "lower-mid"
    return None, "SKIP_PARITY"


def ev_net(p: float, ask: float, fee_rate: float = 0.0) -> float:
    """Expected net per $1 of PURCHASE COST bought at ask.

    Gate-time default fee_rate=0.0 (gross gate: fee is UNKNOWN, never assumed).
    Net evaluation happens separately in signed fee scenarios (economics)."""
    if ask <= 0 or ask >= 1:
        return float("-inf")
    shares_per_dollar = 1.0 / (ask * (1.0 + fee_rate))
    return p * shares_per_dollar - 1.0


def decide(policy: str, opp: Opportunity, ev_threshold: float,
           ask_min: float = 0.01, ask_max: float = 0.40,
           fee_rate: float = 0.0) -> PolicyDecision:
    side, why = pick_outsider_side(opp)
    if side is None:
        return PolicyDecision(False, None, None, why)
    ask = opp.up_ask if side == "UP" else opp.down_ask
    if ask is None or not (ask_min <= ask <= ask_max):
        return PolicyDecision(False, side, ask, "ASK_OUT_OF_RANGE")
    if policy == "price_control":
        # price control trades every valid outsider (baseline universe)
        return PolicyDecision(True, side, ask, "price-control: valid outsider")
    if policy == "CT":
        # BTC_CT_T5_V1 entry: strictly REVERSION (see ct_feature.ct_allows_entry).
        if opp.ct_regime == "REVERSION":
            return PolicyDecision(True, side, ask, f"CT gate: {opp.ct_regime}")
        return PolicyDecision(False, side, ask, f"CT blocks: {opp.ct_regime}")
    # model policies: EV gate on the chosen outsider side
    p = opp.p_up_model if side == "UP" else (1.0 - opp.p_up_model if opp.p_up_model is not None else None)
    if p is None:
        return PolicyDecision(False, side, ask, "NO_MODEL_P")
    if policy == "challenger_plus_CT":
        if opp.ct_regime != "REVERSION":
            return PolicyDecision(False, side, ask, f"CT blocks: {opp.ct_regime}")
    if ev_net(p, ask, fee_rate) >= ev_threshold:
        return PolicyDecision(True, side, ask, f"EV {ev_net(p, ask, fee_rate):.4f}>=thr {ev_threshold}")
    return PolicyDecision(False, side, ask, "EV_BELOW_THRESHOLD")
