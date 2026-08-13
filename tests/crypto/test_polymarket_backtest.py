from datetime import datetime, timezone

import pandas as pd
import pytest

from polyflip.crypto.polymarket_backtest import (
    aggregate_stored_polymarket_backtests,
    compute_oof_polymarket_backtest,
)


def _fixtures():
    starts = pd.to_datetime(
        ["2026-08-01T00:00:00Z", "2026-08-01T00:15:00Z"], utc=True
    )
    frame = pd.DataFrame(
        {
            "market_id": ["m1", "m2"],
            "asset": ["BTC", "BTC"],
            "market_start": starts,
            "recorded_at": starts,
            "target": [0, 1],
            "final_outcome": ["NO", "YES"],
            "time_left_min": [14.0, 8.0],
            "vol_regime": ["low_vol", "high_vol"],
        }
    )
    quotes = pd.DataFrame(
        {
            "market_id": ["m1", "m2"],
            "mid_price": [0.80, 0.20],
            "best_bid": [0.79, 0.19],
            "best_ask": [0.81, 0.21],
            "spread": [0.02, 0.02],
            "recorded_at": starts,
        }
    )
    return frame, quotes


def test_outside_branch_uses_real_yes_no_ask_and_outcome():
    frame, quotes = _fixtures()
    result = compute_oof_polymarket_backtest(
        frame,
        [0.20, 0.80],
        quotes,
        strategy_branch="OUTSIDER_ONLY",
        min_edge=0.04,
        cost_buffer=0.02,
        fee_rate=0.0,
    )

    assert result["n_trades"] == 2
    assert result["win_rate"] == pytest.approx(1.0)
    assert {item["side"] for item in result["trades"]} == {"BUY_NO", "BUY_YES"}
    no_trade = next(item for item in result["trades"] if item["side"] == "BUY_NO")
    # NO is bought at 1 - YES bid, never at min(YES, 1-YES).
    assert no_trade["price"] == pytest.approx(0.21)
    assert no_trade["p_win"] == pytest.approx(0.80)
    assert result["net_profit"] > 0


def test_strategy_branches_are_explicit_and_missing_quotes_are_not_losses():
    frame, quotes = _fixtures()
    favorite = compute_oof_polymarket_backtest(
        frame,
        [0.20, 0.80],
        quotes,
        strategy_branch="FAVORITE_ONLY",
        min_edge=0.04,
        cost_buffer=0.02,
    )
    assert favorite["n_trades"] == 0

    partial = compute_oof_polymarket_backtest(
        frame,
        [0.20, 0.80],
        quotes.iloc[:1],
        strategy_branch="OUTSIDER_ONLY",
        min_edge=0.04,
        cost_buffer=0.02,
    )
    assert partial["n_markets"] == 2
    assert partial["n_quotes"] == 1
    assert partial["coverage_pct"] == pytest.approx(50.0)
    assert partial["n_trades"] == 1
    assert partial["coverage_reasons"]["missing_quote"] == 1


def test_coverage_reasons_explain_price_and_edge_exclusions():
    frame, quotes = _fixtures()
    result = compute_oof_polymarket_backtest(
        frame,
        [0.51, 0.51],
        quotes,
        strategy_branch="OUTSIDER_ONLY",
        min_edge=0.5,
        cost_buffer=0.02,
        min_price=0.8,
        max_price=0.9,
        outsider_max_price=0.45,
    )

    assert result["n_trades"] == 0
    assert result["coverage_reasons"]["price_out_of_bounds"] == 2

    edge_result = compute_oof_polymarket_backtest(
        frame,
        [0.51, 0.51],
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.5,
        cost_buffer=0.02,
    )
    assert edge_result["coverage_reasons"]["insufficient_edge"] == 2


def test_aggregate_replays_trade_pnl_in_time_order():
    first = {
        "n_markets": 1,
        "n_quotes": 1,
        "n_oof": 1,
        "n_eligible": 1,
        "n_trades": 1,
        "win_rate": 1.0,
        "total_invested": 1.0,
        "net_profit": 0.5,
        "avg_edge": 0.1,
        "avg_net_edge": 0.08,
        "avg_entry_price": 0.4,
        "slices": [],
        "equity_curve": [{"entry_time": "2026-08-01T00:00:00+00:00", "trade_pnl": 0.5, "pnl": 0.5}],
    }
    second = {
        **first,
        "net_profit": -1.0,
        "equity_curve": [{"entry_time": "2026-08-01T00:15:00+00:00", "trade_pnl": -1.0, "pnl": -1.0}],
        "win_rate": 0.0,
        "avg_edge": 0.2,
    }
    result = aggregate_stored_polymarket_backtests(
        [first, second], strategy_branch="COMBINED"
    )
    assert result["strategy_branch"] == "COMBINED"
    assert result["n_trades"] == 2
    assert result["net_profit"] == pytest.approx(-0.5)
    assert [item["pnl"] for item in result["equity_curve"]] == [0.5, -0.5]
    assert result["max_drawdown_usdc"] == pytest.approx(1.0)
    # Two $1 trades deploy $2, so the $1 drawdown is 50%, not 100%.
    assert result["max_drawdown_pct"] == pytest.approx(50.0)


def test_aggregate_drawdown_uses_persisted_stake():
    frame, quotes = _fixtures()
    frame.loc[1, "final_outcome"] = "NO"
    quotes.loc[1, "best_ask"] = 0.85
    quotes.loc[1, "best_bid"] = 0.81
    computed = compute_oof_polymarket_backtest(
        frame,
        [0.90, 0.90],
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.01,
        cost_buffer=0.0,
        fee_rate=0.0,
        stake_usdc=2.0,
    )
    assert computed["n_trades"] == 2
    persisted = {key: value for key, value in computed.items() if key != "trades"}
    aggregated = aggregate_stored_polymarket_backtests(
        [persisted], strategy_branch="COMBINED"
    )
    assert aggregated["stake_usdc"] == pytest.approx(2.0)
    assert aggregated["max_drawdown_usdc"] == pytest.approx(computed["max_drawdown_usdc"])
    assert aggregated["max_drawdown_pct"] == pytest.approx(computed["max_drawdown_pct"])
    assert aggregated["max_drawdown_pct"] < 100.0
