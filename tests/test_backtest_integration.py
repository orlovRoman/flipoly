# tests/test_backtest_integration.py
import pytest
import pickle
from datetime import datetime, timezone, timedelta
from polyflip.api.backtest_schemas import BacktestConfig
from polyflip.api.backtest_api import _build_result
from polyflip.backtesting.market_replay import MarketReplay, group_snapshots_into_replays
from polyflip.backtesting.runner import BacktestRunner
from dataclasses import dataclass

@dataclass
class MockSnapshot:
    market_id: str
    asset: str
    time_left_min: float
    yes_price: float
    no_price: float
    best_bid_yes: float
    best_ask_yes: float
    final_outcome: str
    recorded_at: datetime
    spread: float = 0.02
    volume_5min: float = 0.0
    price_velocity: float = 0.0
    
    @property
    def hour_of_day(self):
        return self.recorded_at.hour
        
    @property
    def mid_price(self):
        return (self.best_bid_yes + self.best_ask_yes) / 2 if self.best_bid_yes > 0 and self.best_ask_yes > 0 else self.yes_price

def test_full_backtest_integration():
    # 1. Create mock snapshots
    now = datetime.now(timezone.utc)
    snapshots = [
        # Market 1: Trend YES (WIN)
        MockSnapshot("m1", "BTC", 50, 0.60, 0.40, 0.59, 0.61, "YES", now - timedelta(minutes=50)),
        MockSnapshot("m1", "BTC", 30, 0.70, 0.30, 0.69, 0.71, "YES", now - timedelta(minutes=30)),
        MockSnapshot("m1", "BTC", 10, 0.85, 0.15, 0.84, 0.86, "YES", now - timedelta(minutes=10)),
        
        # Market 2: Outsider flip (LOSS)
        MockSnapshot("m2", "ETH", 45, 0.80, 0.20, 0.79, 0.81, "NO", now - timedelta(minutes=45)),
        MockSnapshot("m2", "ETH", 25, 0.60, 0.40, 0.59, 0.61, "NO", now - timedelta(minutes=25)),
        MockSnapshot("m2", "ETH",  5, 0.40, 0.60, 0.39, 0.41, "NO", now - timedelta(minutes=5)),
    ]
    
    # 2. Group into replays
    replays = group_snapshots_into_replays(snapshots)
    assert len(replays) == 2
    
    # 3. Configure runner
    config = BacktestConfig(
        assets=["BTC", "ETH"],
        strategy_mode="PURE_FAVORITE",
        favorite_threshold=0.65,
        yes_min_price=0.5,
        yes_max_price=0.9,
        no_min_price=0.5,
        no_max_price=0.9,
        kelly_enabled=False,
        trade_bet_size_usdc=10.0,
        slippage_pct=0.01
    )
    runner_cfg = config.to_runner_config()
    
    # 4. Run Backtest
    runner = BacktestRunner(runner_cfg, model_blob=None, features="")
    trades = runner.run_all(replays)
    
    # Market 1 should have a trade because yes_price starts < 0.65 then goes > 0.65.
    # Actually wait, Market 1: 0.60 -> 0.70 (above 0.65, so PURE_FAVORITE buys YES).
    # Market 2: 0.80 -> 0.60 (was above 0.65, drops below. PURE_FAVORITE might not trigger or triggers on NO later).
    
    # 5. Build Result
    res = _build_result(
        run_id="test-run-1",
        config=config,
        started_at=now,
        finished_at=datetime.now(timezone.utc),
        total_loaded=len(snapshots),
        tradeable=len(replays),
        skipped=0,
        trades=trades,
        replays=replays
    )
    
    assert res.run_id == "test-run-1"
    assert res.total_markets_loaded == 6
    assert res.tradeable_markets == 2
    # At least check that we processed the logic without crashing
    assert isinstance(res.net_profit, float)
    assert isinstance(res.win_rate_pct, float)
    assert isinstance(res.strategies, list)
