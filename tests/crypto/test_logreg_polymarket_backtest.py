import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.logreg_polymarket_backtest import (
    _first_valid_per_market,
    compute_logreg_polymarket_backtest,
    flip_probability_to_yes_probability,
)


def test_flip_probability_conversion_matches_training_target():
    assert flip_probability_to_yes_probability(0.2, 0.8) == pytest.approx(0.8)
    assert flip_probability_to_yes_probability(0.2, 0.2) == pytest.approx(0.2)


def test_midpoint_has_no_canonical_favourite():
    with pytest.raises(ValueError, match="no canonical favourite"):
        flip_probability_to_yes_probability(0.2, 0.5)


def test_logreg_oot_uses_one_first_valid_trade_per_market():
    frame = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "recorded_at": "2026-01-01T00:00:01Z",
                "mid_price": 0.80,
                "best_bid": 0.74,
                "best_ask": 0.75,
                "final_outcome": "NO",
            },
            {
                "market_id": "m1",
                "recorded_at": "2026-01-01T00:01:01Z",
                "mid_price": 0.80,
                "best_bid": 0.74,
                "best_ask": 0.75,
                "final_outcome": "NO",
            },
            {
                "market_id": "m2",
                "recorded_at": "2026-01-01T00:00:01Z",
                "mid_price": 0.20,
                "best_bid": 0.14,
                "best_ask": 0.15,
                "final_outcome": "YES",
            },
        ]
    )
    quotes = frame[["market_id", "recorded_at", "mid_price", "best_bid", "best_ask", "final_outcome"]]
    result = compute_logreg_polymarket_backtest(
        frame,
        [np.nan, 0.2, 0.2],
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.0,
        cost_buffer=0.0,
    )
    assert result["n_markets"] == 2
    assert result["n_trades"] == 2
    assert len(result["trades"]) == 2

def test_midpoint_boundary_keeps_canonical_side():
    assert flip_probability_to_yes_probability(0.2, 0.4999) == pytest.approx(0.2)
    assert flip_probability_to_yes_probability(0.2, 0.5001) == pytest.approx(0.8)


def test_invalid_earliest_snapshot_does_not_hide_later_valid_snapshot():
    frame = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "recorded_at": "2026-01-01T00:00:01Z",
                "mid_price": 0.5,
                "final_outcome": "YES",
            },
            {
                "market_id": "m1",
                "recorded_at": "2026-01-01T00:01:01Z",
                "mid_price": 0.8,
                "final_outcome": "YES",
            },
        ]
    )
    selected, p_yes = _first_valid_per_market(frame, [0.2, 0.2])
    assert selected["recorded_at"].iloc[0] == pd.Timestamp(
        "2026-01-01T00:01:01Z"
    )
    assert p_yes.tolist() == pytest.approx([0.8])


def test_missing_quotes_are_not_inferred_from_oof_frame():
    frame = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "recorded_at": "2026-01-01T00:00:01Z",
                "mid_price": 0.2,
                "best_bid": 0.14,
                "best_ask": 0.15,
                "final_outcome": "YES",
            }
        ]
    )
    result = compute_logreg_polymarket_backtest(
        frame,
        [0.2],
        None,
        strategy_branch="COMBINED",
        min_edge=0.0,
        cost_buffer=0.0,
    )
    assert result["n_quotes"] == 0
    assert result["n_trades"] == 0
