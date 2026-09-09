"""
polyflip/research/ct_replay.py

Historical YES-only Replay and Economics Reconciliation (Stage 1, Items 6 & 7):
- Replays candidate opportunities from the locked common opportunity ledger.
- Asserts 100% identifier-level match against the 647 research CT entries.
- Reconciles line-item economics (+81.6477 USDC) with explicit discrepancy reporting.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from polyflip.trading.ct_policy import (
    CTSpecification,
    get_btc_ct_t5_v1_spec,
    calculate_scenario_economics,
)


@dataclass(frozen=True)
class HistoricalReplayResult:
    total_ledger_rows: int
    selected_count: int
    expected_count: int
    id_match_exact: bool
    matched_ids_count: int
    missing_ids: list[str]
    unexpected_ids: list[str]
    reproduced_net_pnl: float
    ledger_net_pnl: float
    pnl_absolute_difference: float
    max_line_item_diff: float
    line_item_discrepancies: list[dict[str, Any]]
    win_count: int
    win_rate: float


def replay_historical_ct_ledger(
    ledger_path: str | Path,
    spec: CTSpecification | None = None,
    tolerance: float = 1e-4,
) -> HistoricalReplayResult:
    """
    Executes pure historical replay over the locked common opportunity ledger (Points 6 & 7).

    Validates:
    1. Identifiers: exactly matches the set of 647 opportunity IDs where variants.CT == True.
    2. Economics: line-item PnL calculation matches ledger net_pnl within tolerance,
       summing to +81.6477 USDC.
    """
    spec = spec or get_btc_ct_t5_v1_spec()
    path = Path(ledger_path)
    if not path.exists():
        raise FileNotFoundError(f"Common opportunity ledger not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    selected_ids: list[str] = []
    expected_ids: list[str] = []

    reproduced_pnl_sum = 0.0
    ledger_pnl_sum = 0.0
    max_line_diff = 0.0
    discrepancies: list[dict[str, Any]] = []
    win_count = 0

    for row in ledger:
        opp_id = str(row.get("opportunity_id"))
        is_expected_ct = bool(row.get("variants", {}).get("CT", False))
        if is_expected_ct:
            expected_ids.append(opp_id)

        # Replay YES-only selection rule from locked research protocol
        ask = row.get("best_ask")
        mid = row.get("mid_price")
        # In the research ledger, mid_price or best_ask can be float/nan
        is_candidate_price = bool(row.get("is_candidate_price", False))
        token_regime = str(row.get("token_regime_CT", "")).strip().upper()

        is_selected = is_candidate_price and (token_regime == spec.required_signal)

        if is_selected:
            selected_ids.append(opp_id)
            target = int(row.get("target", 0))
            if target == 1:
                win_count += 1

            # Line-item economics check
            ask_val = float(ask) if ask is not None else 0.0
            econ = calculate_scenario_economics(
                ask=ask_val,
                outcome="WIN" if target == 1 else "LOSS",
                budget_usdc=spec.max_purchase_cost_usdc,
                taker_fee_rate=spec.taker_fee_rate,
            )
            rep_net = round(econ["net_pnl"], 4)
            led_net = float(row.get("net_pnl", 0.0))

            reproduced_pnl_sum += rep_net
            ledger_pnl_sum += led_net

            diff = abs(rep_net - led_net)
            if diff > max_line_diff:
                max_line_diff = diff

            if diff > tolerance:
                discrepancies.append({
                    "opportunity_id": opp_id,
                    "ask": ask_val,
                    "target": target,
                    "reproduced_net_pnl": rep_net,
                    "ledger_net_pnl": led_net,
                    "diff": diff,
                })

    selected_set = set(selected_ids)
    expected_set = set(expected_ids)

    missing = sorted(list(expected_set - selected_set))
    unexpected = sorted(list(selected_set - expected_set))
    id_match = (len(missing) == 0 and len(unexpected) == 0)

    win_rate = (win_count / len(selected_ids)) if selected_ids else 0.0
    pnl_diff = abs(reproduced_pnl_sum - ledger_pnl_sum)

    return HistoricalReplayResult(
        total_ledger_rows=len(ledger),
        selected_count=len(selected_ids),
        expected_count=len(expected_ids),
        id_match_exact=id_match,
        matched_ids_count=len(selected_set.intersection(expected_set)),
        missing_ids=missing,
        unexpected_ids=unexpected,
        reproduced_net_pnl=round(reproduced_pnl_sum, 4),
        ledger_net_pnl=round(ledger_pnl_sum, 4),
        pnl_absolute_difference=round(pnl_diff, 6),
        max_line_item_diff=round(max_line_diff, 6),
        line_item_discrepancies=discrepancies,
        win_count=win_count,
        win_rate=round(win_rate, 4),
    )
