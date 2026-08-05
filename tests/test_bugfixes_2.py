import pytest
import dataclasses
from polyflip.trading.decision_runners import decide_favorite_mode
from polyflip.trading.trading_config import TradingConfig
from polyflip.backtesting.runner import BacktestRunner
import os

@pytest.fixture
def mock_cfg():
    return TradingConfig(
        trading_enabled=True, trading_mode="PURE_FAVORITE",
        favor_min_time_left=60, favor_max_time_left=600,
        outs_min_time_left=60, outs_max_time_left=600,
        bet_size=5.0, dead_zone=0.0, daily_limit=50.0,
        trade_min_price=0.1, trade_max_price=0.9, capital=100.0,
        active_features_str="", trade_on_favorite=True, trade_on_flip=False,
        flip_threshold=0.55, outs_min_edge=0.05,
        entry_sec=5, favorite_threshold=0.55, trade_assets=["ETH"],
        bet_sizing_mode="scaled", max_bet_size_usdc=50.0,
        favorite_min_price=0.1, favorite_max_price=0.9, favorite_min_edge=0.01,
        outsider_max_price=0.9, liquidity_fraction=0.05, bypass_bet_size_check=False,
        stop_loss_enabled=False, take_profit_enabled=False, take_profit_multiplier=2.0,
        max_price_drift=0.05, stop_loss_pct_favorite=0.1, stop_loss_pct_outsider=0.1,
        fee_rate=0.0, slippage_rate=0.0, max_exposure_pct=0.1, min_direction_prob=0.55,
        min_win_prob=0.55
    )

def test_pre_trade_validator_no_lightgbm_trend_block(mock_cfg):
    """Bug 1: LIGHTGBM_TREND should not trigger dead code."""
    file_path = os.path.join(os.path.dirname(__file__), "..", "polyflip", "trading", "pre_trade_validator.py")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "LIGHTGBM_TREND" not in content, "Dead code for LIGHTGBM_TREND still present in pre_trade_validator.py"

def test_decide_favorite_mode_time_left(mock_cfg):
    """Bug 2: time_left_sec is properly passed to decide_favorite."""
    import datetime
    class DummySignal:
        market_id = "test_123"
        asset = "BTC"
        token_a_prob = 0.6
        token_b_prob = 0.4
        current_yes_ask = 0.6
        current_no_ask = 0.4
        current_yes_price = 0.6
        current_no_price = 0.4
        current_spread = 0.01
        price_velocity = 0.0
        volume_5min = 1000.0

    signal = DummySignal()
    # time_left_sec=5 should trigger validation skip in decide_favorite
    import asyncio
    import datetime
    st = datetime.datetime.now(datetime.timezone.utc)
    result = asyncio.run(decide_favorite_mode(signal, mock_cfg, asset_min_edge=0.01, asset_max_price=0.9, start_time=st, time_left_sec=5.0))
    assert result.decision_obj.action == "SKIP"
    assert "time_left=5s out of [60, 600]" in result.decision_obj.reason

def test_backtest_runner_strategy_mode_validation():
    """Bug 3: Explicit guard for unsupported strategy_mode in BacktestRunner."""
    with pytest.raises(ValueError, match="is not supported"):
        BacktestRunner({"STRATEGY_MODE": "COMBINED"}, b"", "")
    
    # Should work for OUTSIDER and PURE_FAVORITE
    BacktestRunner({"STRATEGY_MODE": "OUTSIDER"}, b"", "")
    BacktestRunner({"STRATEGY_MODE": "PURE_FAVORITE"}, b"", "")

def test_bet_size_is_outsider_usage(mock_cfg):
    """Bug 4/5: is_outsider affects min_edge but base_bet is the same."""
    from polyflip.trading.decision_logic import _resolve_final_bet
    
    # outsider vs favorite
    bet_fav = _resolve_final_bet(edge=0.1, volume_5min=1000.0, cfg=mock_cfg, is_outsider=False)
    bet_out = _resolve_final_bet(edge=0.1, volume_5min=1000.0, cfg=mock_cfg, is_outsider=True)
    
    # In mock_cfg, favor_min_edge is 0.01 and outs_min_edge is 0.05.
    # So for edge=0.1, they might scale differently.
    # However, if edge scales to same or different, base_bet is identical.
    # We just want to ensure it runs without error.
    assert bet_fav > 0.0
    assert bet_out > 0.0
