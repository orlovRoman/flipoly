from datetime import datetime, timezone

import pandas as pd
import pytest

from polyflip.crypto.polymarket_backtest import (
    aggregate_stored_polymarket_backtests,
    compute_oof_polymarket_backtest,
    load_market_entry_quotes,
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
    frame.loc[0, "final_outcome"] = "YES"
    quotes.loc[0, "best_ask"] = 0.50
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


def test_aggregate_preserves_weighted_accounting_metadata():
    result = aggregate_stored_polymarket_backtests(
        [{
            "n_markets": 1,
            "n_quotes": 1,
            "n_oof": 1,
            "n_eligible": 1,
            "n_trades": 1,
            "win_rate": 1.0,
            "total_invested": 1.0,
            "stake_usdc": 1.0,
            "net_profit": 0.1,
            "avg_edge": 0.1,
            "avg_net_edge": 0.08,
            "avg_entry_price": 0.4,
            "policy_mode": "WEIGHTED_ACTIVE",
            "accounting_model": "POLYMARKET_PRICE_DEPENDENT_BUDGET",
            "fee_model": "POLYMARKET_PRICE_DEPENDENT",
            "fee_rate": 0.07,
            "execution_role": "TAKER",
            "slices": [],
            "equity_curve": [{
                "entry_time": "2026-08-01T00:00:00+00:00",
                "trade_pnl": 0.1,
                "pnl": 0.1,
            }],
        }],
        strategy_branch="COMBINED",
    )

    assert result["policy_mode"] == "WEIGHTED_ACTIVE"
    assert result["accounting_model"] == "POLYMARKET_PRICE_DEPENDENT_BUDGET"
    assert result["fee_model"] == "POLYMARKET_PRICE_DEPENDENT"
    assert result["fee_rate"] == pytest.approx(0.07)
    assert result["execution_role"] == "TAKER"


def test_weighted_backtest_uses_same_cost_aware_policy_and_source_alignment():
    """Weighted replay must use the matching OOF row after quote filtering."""
    frame, quotes = _fixtures()
    # Keep this test focused on source-index alignment. A separate spread
    # scenario verifies that historical spread reaches the weighted scorer.
    quotes["spread"] = 0.0
    # Keep only the second quote.  The second OOF score is deliberately high;
    # using the compressed merge index would incorrectly read the first score.
    result = compute_oof_polymarket_backtest(
        frame,
        [0.10, 0.90],
        quotes.iloc[[1]],
        strategy_branch="COMBINED",
        policy_mode="WEIGHTED",
        p_logreg_scores=[0.50, 0.50],
        min_edge=0.02,
    )

    assert result["policy_mode"] == "WEIGHTED"
    assert result["accounting_model"] == "POLYMARKET_PRICE_DEPENDENT_BUDGET"
    assert result["n_quotes"] == 1
    assert result["n_trades"] == 1
    trade = result["trades"][0]
    assert trade["market_id"] == "m2"
    assert trade["side"] == "BUY_YES"
    assert trade["p_win"] == pytest.approx(0.24879061)
    assert trade["fee_usdc"] > 0.0
    assert trade["shares"] > 0.0
    assert trade["total_cost_usdc"] == pytest.approx(1.0)


def test_weighted_backtest_rejects_edge_consumed_by_historical_spread():
    frame, quotes = _fixtures()
    result = compute_oof_polymarket_backtest(
        frame.iloc[[0]],
        [0.80],
        quotes.iloc[[0]],
        strategy_branch="COMBINED",
        policy_mode="WEIGHTED",
        p_logreg_scores=[0.80],
        weighted_fee_rate=0.0,
        weighted_slippage_rate=0.0,
        min_edge=0.0,
    )

    assert result["n_trades"] == 0
    assert result["coverage_reasons"]["no_positive_weighted_ev"] == 1


def test_weighted_backtest_missing_logreg_renormalizes_and_keeps_loss_budget_bounded():
    frame, quotes = _fixtures()
    frame.loc[0, "final_outcome"] = "NO"
    quotes.loc[0, "best_ask"] = 0.75
    result = compute_oof_polymarket_backtest(
        frame.iloc[[0]],
        [0.95],
        quotes.iloc[[0]],
        strategy_branch="COMBINED",
        policy_mode="WEIGHTED",
        min_edge=0.0,
    )

    assert result["n_trades"] == 1
    trade = result["trades"][0]
    assert trade["won"] is False
    # A fixed $1 budget must not turn a loss into a gross-price-dependent loss.
    assert trade["total_cost_usdc"] == pytest.approx(1.0)
    assert trade["pnl"] == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_load_market_entry_quotes_empty():
    assert (await load_market_entry_quotes(None, None)).empty
    assert (await load_market_entry_quotes(None, pd.DataFrame())).empty


@pytest.mark.asyncio
async def test_load_market_entry_quotes_portable_selection():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from polyflip.db.models import Base, MarketSnapshot

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    t_start_m1 = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    t_before_m1 = datetime(2026, 8, 1, 11, 55, tzinfo=timezone.utc)
    t_first_m1 = datetime(2026, 8, 1, 12, 1, tzinfo=timezone.utc)
    t_later_m1 = datetime(2026, 8, 1, 12, 5, tzinfo=timezone.utc)

    t_start_m2 = datetime(2026, 8, 1, 12, 15, tzinfo=timezone.utc)
    t_first_m2 = datetime(2026, 8, 1, 12, 15, tzinfo=timezone.utc)

    async with AsyncSession(engine) as session:
        session.add_all([
            MarketSnapshot(
                market_id="m1", asset="DOGE", recorded_at=t_before_m1,
                time_left_min=20.0, mid_price=0.20, spread=0.02, volume_5min=100.0,
                price_velocity=0.0, hour_of_day=11, final_outcome="YES",
            ),
            MarketSnapshot(
                market_id="m1", asset="DOGE", recorded_at=t_first_m1,
                time_left_min=14.0, mid_price=0.25, spread=0.03, volume_5min=100.0,
                price_velocity=0.0, hour_of_day=12, final_outcome="YES",
            ),
            MarketSnapshot(
                market_id="m1", asset="DOGE", recorded_at=t_later_m1,
                time_left_min=10.0, mid_price=0.30, spread=0.04, volume_5min=100.0,
                price_velocity=0.0, hour_of_day=12, final_outcome="YES",
            ),
            MarketSnapshot(
                market_id="m2", asset="DOGE", recorded_at=t_first_m2,
                time_left_min=15.0, mid_price=0.50, spread=0.01, volume_5min=50.0,
                price_velocity=0.0, hour_of_day=12, final_outcome="NO",
            ),
        ])
        await session.commit()

        market_starts = pd.DataFrame([
            {"market_id": "m1", "market_start": t_start_m1},
            {"market_id": "m2", "market_start": t_start_m2},
            {"market_id": "m3_missing", "market_start": t_start_m1},
        ])

        quotes = await load_market_entry_quotes(session, market_starts)
        assert len(quotes) == 2
        assert set(quotes["market_id"]) == {"m1", "m2"}

        m1_quote = quotes[quotes["market_id"] == "m1"].iloc[0]
        # Must pick t_first_m1, not t_before_m1 or t_later_m1
        assert m1_quote["mid_price"] == pytest.approx(0.25)
        assert m1_quote["time_left_min"] == pytest.approx(14.0)

        m2_quote = quotes[quotes["market_id"] == "m2"].iloc[0]
        assert m2_quote["mid_price"] == pytest.approx(0.50)

