import pytest
from datetime import datetime
from polyflip.backtesting.metrics import compute_trade_pnl
from polyflip.backtesting.simulated_trader import SimulatedTrader, SimulatedTrade
from polyflip.backtesting.market_replay import MarketReplay
from polyflip.trading.decision_logic import TradeDecision

class MockSnapshot:
    def __init__(self, market_id, asset, time_left, mid_price, final_outcome, spread=0.0):
        self.market_id = market_id
        self.asset = asset
        self.time_left_min = time_left
        self.mid_price = mid_price
        self.final_outcome = final_outcome
        self.spread = spread
        self.volume_5min = 0.0
        self.price_velocity = 0.0
        self.hour_of_day = 12
        self.recorded_at = datetime.now()

def test_fee_only_on_profit():
    """Fee = 2% от прибыли, не от revenue."""
    trader = SimulatedTrader(slippage_pct=0.0)
    decision = TradeDecision("BUY_YES", 0.50, 10.0, "test", "ML_TREND")
    trade = trader.execute_trade("m1", "BTC", decision, datetime.now(), 0.2)
    
    snaps = [MockSnapshot("m1", "BTC", 5.0, 0.7, "YES")]
    replay = MarketReplay(snaps)
    pnl = compute_trade_pnl(trade, replay)
    assert pnl == pytest.approx(9.8)

def test_fee_is_zero_on_loss():
    """На убыточной сделке fee = 0."""
    trader = SimulatedTrader(slippage_pct=0.0)
    decision = TradeDecision("BUY_YES", 0.50, 10.0, "test", "ML_TREND")
    trade = trader.execute_trade("m1", "BTC", decision, datetime.now(), 0.2)
    snaps = [MockSnapshot("m1", "BTC", 5.0, 0.7, "NO")]
    replay = MarketReplay(snaps)
    pnl = compute_trade_pnl(trade, replay)
    assert pnl == pytest.approx(-10.0)
