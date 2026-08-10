import pytest
from polyflip.trading.trading_config import TradingConfig
from polyflip.backtesting.runner import BacktestRunner
import pickle
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


class ConstantFlipModel:
    def predict_proba(self, frame):
        import numpy as np
        return np.tile([0.1, 0.9], (len(frame), 1))

@pytest.fixture
def mock_cfg():


    return TradingConfig(
        trading_enabled=True, trading_mode="combined",
        favor_min_time_left=60, favor_max_time_left=600,
        outs_min_time_left=60, outs_max_time_left=600,
        bet_size=5.0, dead_zone=0.0, daily_limit=50.0,
        trade_min_price=0.1, trade_max_price=0.9, capital=100.0,
        active_features_str="", trade_on_favorite=True, trade_on_flip=False,
        flip_threshold=0.55, outs_min_edge=0.05,
        favorite_threshold=0.55, trade_assets=["ETH"],
        bet_sizing_mode="scaled", max_bet_size_usdc=50.0,
        favorite_min_price=0.1, favorite_max_price=0.9, favorite_min_edge=0.01,
        outsider_max_price=0.9, liquidity_fraction=0.05, bypass_bet_size_check=False,
        stop_loss_enabled=False, take_profit_enabled=False, take_profit_multiplier=2.0,
        max_price_drift=0.05, stop_loss_pct_favorite=0.1, stop_loss_pct_outsider=0.1,
        fee_rate=0.0, slippage_rate=0.0, max_exposure_pct=0.1, min_direction_prob=0.55,
        min_win_prob=0.55
    )

def test_backtest_runner_strategy_mode_validation():
    """Bug 3: Explicit guard for unsupported strategy_mode in BacktestRunner."""
    with pytest.raises(ValueError, match="is not supported"):
        BacktestRunner({"STRATEGY_MODE": "COMBINED"}, b"", "")
    
    # Should work for OUTSIDER
    BacktestRunner({"STRATEGY_MODE": "OUTSIDER"}, b"", "")

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


def test_backtest_runner_uses_selected_logreg_model():
    from polyflip.backtesting.market_replay import MarketReplay

    now = datetime.now(timezone.utc)
    snapshots = [
        SimpleNamespace(
            market_id="m1", asset="BTC", time_left_min=14.0,
            mid_price=0.70, spread=0.02, volume_5min=100.0,
            price_velocity=0.01, hour_of_day=now.hour,
            final_outcome="NO", recorded_at=now,
        ),
        SimpleNamespace(
            market_id="m1", asset="BTC", time_left_min=10.0,
            mid_price=0.72, spread=0.02, volume_5min=100.0,
            price_velocity=0.01, hour_of_day=now.hour,
            final_outcome="NO", recorded_at=now + timedelta(minutes=4),
        ),
    ]
    replay = MarketReplay(snapshots)
    runner = BacktestRunner(
        {"STRATEGY_MODE": "OUTSIDER"},
        pickle.dumps(ConstantFlipModel()),
        "mid_price",
    )

    probability = runner._predict_flip(replay.ticks[-1], replay)

    assert probability == pytest.approx(0.9)


def test_logreg_feature_set_selector_is_unique_and_not_nested():
    from html.parser import HTMLParser
    from pathlib import Path

    class SelectParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.select_depth = 0
            self.nested_selects = 0
            self.feature_set_ids = 0

        def handle_starttag(self, tag, attrs):
            if tag != "select":
                return
            if self.select_depth:
                self.nested_selects += 1
            self.select_depth += 1
            if dict(attrs).get("id") == "logreg-feature-set":
                self.feature_set_ids += 1

        def handle_endtag(self, tag):
            if tag == "select":
                self.select_depth -= 1

    template = Path("polyflip/templates/index.html").read_text(encoding="utf-8")
    parser = SelectParser()
    parser.feed(template)

    assert parser.feature_set_ids == 1
    assert parser.nested_selects == 0
