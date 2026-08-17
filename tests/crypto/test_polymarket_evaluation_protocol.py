from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from polyflip.crypto.polymarket_evaluation_protocol import (
    CANONICAL_EVALUATION_PROTOCOL,
    EVALUATION_PROTOCOL_VERSION,
    CanonicalEvaluationProtocol,
    evaluate_logreg_with_protocol,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "market_id": ["m1", "m2"],
            "recorded_at": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:01Z"]),
            "mid_price": [0.30, 0.70],
            "final_outcome": ["YES", "NO"],
        }
    )
    quotes = pd.DataFrame(
        {
            "market_id": ["m1", "m2"],
            "yes_price": [0.30, 0.70],
            "no_price": [0.70, 0.30],
            "mid_price": [0.30, 0.70],
            "final_outcome": ["YES", "NO"],
            "recorded_at": frame["recorded_at"],
        }
    )
    return frame, quotes


def test_protocol_is_versioned_and_all_parameters_are_explicit():
    assert CANONICAL_EVALUATION_PROTOCOL.protocol_version == EVALUATION_PROTOCOL_VERSION
    assert CANONICAL_EVALUATION_PROTOCOL.as_dict() == {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "strategy_branch": "COMBINED",
        "min_edge": 0.03,
        "fee_rate": 0.02,
        "slippage_pct": 0.0,
        "cost_buffer": 0.0,
        "stake_usdc": 1.0,
        "min_price": 0.05,
        "max_price": 0.95,
        "outsider_max_price": 0.45,
    }


def test_protocol_is_deterministic_and_documents_selection_contract():
    frame, quotes = _inputs()
    first = evaluate_logreg_with_protocol(frame, [0.8, 0.8], quotes)
    second = evaluate_logreg_with_protocol(frame, [0.8, 0.8], quotes)
    assert first == second
    metadata = first["evaluation_metadata"]
    assert metadata["protocol"]["strategy_branch"] == "COMBINED"
    assert "p_flip -> p_yes" in metadata["probability_conversion"]
    assert "recorded_at" in metadata["entry_snapshot_fields"]
    assert metadata["evaluator"] == "compute_logreg_polymarket_backtest"


def test_combined_selects_at_most_one_trade_per_market():
    frame, quotes = _inputs()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = evaluate_logreg_with_protocol(frame, [0.8, 0.8, 0.8], quotes)
    assert result["n_trades"] <= result["n_markets"]
    assert len({trade["market_id"] for trade in result["trades"]}) == result["n_trades"]


def test_fee_slippage_and_stake_are_reflected_in_pnl():
    frame, quotes = _inputs()
    base = evaluate_logreg_with_protocol(frame, [0.8, 0.8], quotes)
    more_cost = evaluate_logreg_with_protocol(
        frame,
        [0.8, 0.8],
        quotes,
        protocol=replace(CANONICAL_EVALUATION_PROTOCOL, fee_rate=0.20, slippage_pct=0.10),
    )
    double_stake = evaluate_logreg_with_protocol(
        frame,
        [0.8, 0.8],
        quotes,
        protocol=replace(CANONICAL_EVALUATION_PROTOCOL, stake_usdc=2.0),
    )
    assert more_cost["net_profit"] < base["net_profit"]
    assert double_stake["net_profit"] == pytest.approx(base["net_profit"] * 2.0)


def test_protocol_rejects_non_combined_or_invalid_values():
    with pytest.raises(ValueError):
        CanonicalEvaluationProtocol(strategy_branch="OUTSIDER_ONLY")
    with pytest.raises(ValueError):
        CanonicalEvaluationProtocol(slippage_pct=1.0)
    with pytest.raises(ValueError):
        CanonicalEvaluationProtocol(stake_usdc=0.0)
