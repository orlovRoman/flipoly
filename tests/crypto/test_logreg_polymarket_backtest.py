import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.logreg_polymarket_backtest import (
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
                "best_bid": 0.79,
                "best_ask": 0.81,
                "final_outcome": "NO",
            },
            {
                "market_id": "m1",
                "recorded_at": "2026-01-01T00:01:01Z",
                "mid_price": 0.80,
                "best_bid": 0.79,
                "best_ask": 0.81,
                "final_outcome": "NO",
            },
            {
                "market_id": "m2",
                "recorded_at": "2026-01-01T00:00:01Z",
                "mid_price": 0.20,
                "best_bid": 0.19,
                "best_ask": 0.21,
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
