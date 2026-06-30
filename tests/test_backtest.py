import pytest
from datetime import datetime, timezone
import pickle

from polyflip.backtesting.market_replay import MarketReplay, MarketTick
from polyflip.backtesting.simulated_trader import SimulatedTrader
from polyflip.backtesting.metrics import compute_trade_pnl, compute_metrics
from polyflip.backtesting.runner import BacktestRunner
from polyflip.trading.decision_logic import TradeDecision


# --- Mock Objects ---
class MockSnapshot:
    def __init__(self, market_id, asset, time_left_min, mid_price, final_outcome):
        self.market_id = market_id
        self.asset = asset
        self.time_left_min = time_left_min
        self.mid_price = mid_price
        self.final_outcome = final_outcome
        self.spread = 0.02
        self.volume_5min = 1000
        self.price_velocity = 0
        self.hour_of_day = 12
        self.recorded_at = datetime.now(timezone.utc)


class MockModel:
    def predict_proba(self, X):
        return [[0.8, 0.2]] * len(X)  # P(flip) = 0.2


# --- Tests ---
def test_market_replay_sorting():
    snaps = [
        MockSnapshot("m1", "BTC", 5.0, 0.7, "YES"),
        MockSnapshot("m1", "BTC", 15.0, 0.6, "YES"),
        MockSnapshot("m1", "BTC", 1.0, 0.8, "YES"),
    ]
    replay = MarketReplay(snaps)
    # По убыванию time_left
    assert [t.time_left_min for t in replay.ticks] == [15.0, 5.0, 1.0]

def test_market_replay_get_entry():
    snaps = [
        MockSnapshot("m1", "BTC", 15.0, 0.6, "YES"),
        MockSnapshot("m1", "BTC", 5.0, 0.7, "YES"),
    ]
    replay = MarketReplay(snaps)
    # Окно [1.0, 10.0]. Должен вернуть тик с 5.0 (так как он первый, кто попал в окно, если идём от начала к концу)
    tick = replay.get_entry_tick(1.0, 10.0)
    assert tick is not None
    assert tick.time_left_min == 5.0

def test_simulated_trader_slippage():
    trader = SimulatedTrader(slippage_pct=0.01) # 1% slippage
    decision = TradeDecision("BUY_YES", 0.50, 10.0, "test", "ML_TREND")
    trade = trader.execute_trade("m1", "BTC", decision, datetime.now(), 0.2)
    assert trade.executed_price == 0.505
    assert trade.slippage == pytest.approx(0.005)
    assert trade.shares == pytest.approx(10.0 / 0.505)

def test_metrics_pnl_win():
    snaps = [MockSnapshot("m1", "BTC", 5.0, 0.7, "YES")]
    replay = MarketReplay(snaps)
    trader = SimulatedTrader(slippage_pct=0.0)
    decision = TradeDecision("BUY_YES", 0.50, 10.0, "test", "ML_TREND")
    trade = trader.execute_trade("m1", "BTC", decision, datetime.now(), 0.2)
    
    pnl = compute_trade_pnl(trade, replay)
    # Выиграли. Revenue = 1.0 * (10 / 0.5) = 20.0
    # Profit = 20.0 - 10.0 = 10.0. Minus 2% fee = 9.8
    assert pnl == pytest.approx(9.8)

def test_metrics_pnl_loss():
    snaps = [MockSnapshot("m1", "BTC", 5.0, 0.7, "NO")]
    replay = MarketReplay(snaps)
    trader = SimulatedTrader(slippage_pct=0.0)
    decision = TradeDecision("BUY_YES", 0.50, 10.0, "test", "ML_TREND")
    trade = trader.execute_trade("m1", "BTC", decision, datetime.now(), 0.2)
    
    pnl = compute_trade_pnl(trade, replay)
    # Проиграли. Потеряли ставку.
    assert pnl == -10.0

def test_backtest_runner_integration():
    config = {
        "MIN_TIME_LEFT_MIN": 1.0,
        "MAX_TIME_LEFT_MIN": 30.0,
        "TRADE_ON_FLIP": "false",
        "NO_FLIP_THRESHOLD": 0.35,
        "FAVORITE_THRESHOLD": 0.65,
        "YES_MIN_PRICE": 0.55, "YES_MAX_PRICE": 0.95,
        "MIN_EDGE": -0.10, # Разрешаем любой эдж для теста
        "KELLY_ENABLED": "false",
        "TRADE_BET_SIZE_USDC": 10.0,
        "AUTO_DEAD_ZONE_WIDTH": 0.0,
    }
    model = MockModel()
    runner = BacktestRunner(config, pickle.dumps(model), "mid_price,spread,volume_5min,price_velocity,hour_of_day,time_left_min")
    
    snaps = [
        MockSnapshot("m1", "BTC", 35.0, 0.7, "YES"), # Out of window
        MockSnapshot("m1", "BTC", 15.0, 0.75, "YES"), # First in window -> BUY_YES
        MockSnapshot("m1", "BTC", 5.0, 0.8, "YES"),
    ]
    replay = MarketReplay(snaps)
    runner.run_all({"m1": replay})
    
    assert len(runner.trader.trades) == 1
    trade = runner.trader.trades[0]
    assert trade.market_id == "m1"
    assert trade.decision.action == "BUY_YES"
    assert trade.decision.strategy_type == "ML_TREND"

    metrics = compute_metrics(runner.trader.trades, {"m1": replay})
    assert metrics["total_trades"] == 1
    assert metrics["win_rate_pct"] == 100.0
