# tests/test_backtest_metrics.py
import pytest
from datetime import datetime, timezone
from polyflip.api.backtest_api import _compute_max_drawdown, _build_result
from polyflip.api.backtest_schemas import EquityCurvePoint, BacktestConfig

def make_point(idx, cum_pnl):
    return EquityCurvePoint(
        trade_index=idx, cumulative_pnl=cum_pnl, trade_pnl=idx,
        market_id="m", asset="BTC", strategy="ML_TREND",
        outcome="WIN", p_flip=None, edge=None, bet_size=5, executed_price=0.5,
    )

def test_drawdown_all_losses():
    """Если equity всегда отрицательный — просадка должна быть < 10000%."""
    curve = [make_point(i, v) for i, v in enumerate([-5, -10, -15, -8])]
    dd = _compute_max_drawdown(curve)
    assert 0 <= dd <= 200, f"Неправдоподобная просадка: {dd}%"

def test_drawdown_peak_then_fall():
    """Классический пик-впадина: +20 → -5 = просадка 75%."""
    curve = [make_point(i, v) for i, v in enumerate([10, 20, 15, 5])]
    dd = _compute_max_drawdown(curve)
    assert abs(dd - 75.0) < 0.01, f"Ожидаем 75%, получили {dd}%"

def test_drawdown_empty():
    assert _compute_max_drawdown([]) == 0.0

def test_worst_trades_sorted_asc():
    """worst_trades[0] должен быть наихудшей сделкой."""
    # Создаем mock данные
    config = BacktestConfig(assets=["BTC"], strategy_mode="PURE_FAVORITE")
    from polyflip.backtesting.simulated_trader import SimulatedTrade
    from polyflip.trading.decision_logic import TradeDecision
    
    trades = [
        SimulatedTrade("m1", "BTC", TradeDecision("BUY_YES", 0.5, 10.0, "t", "ML_TREND"), 0.5, 0.0, 10.0, 20.0, datetime.now(timezone.utc), 0.2),
        SimulatedTrade("m2", "BTC", TradeDecision("BUY_YES", 0.5, 10.0, "t", "ML_TREND"), 0.5, 0.0, 10.0, 20.0, datetime.now(timezone.utc), 0.2),
        SimulatedTrade("m3", "BTC", TradeDecision("BUY_YES", 0.5, 10.0, "t", "ML_TREND"), 0.5, 0.0, 10.0, 20.0, datetime.now(timezone.utc), 0.2),
    ]
    
    # Сделка 1 выиграла (pnl=9.8), Сделка 2 проиграла (pnl=-10), Сделка 3 выиграла (pnl=9.8)
    from polyflip.backtesting.market_replay import MarketReplay
    
    class MockSnap:
        def __init__(self, market_id, outcome):
            self.market_id = market_id
            self.asset = "BTC"
            self.time_left_min = 10.0
            self.mid_price = 0.5
            self.spread = 0.02
            self.volume_5min = 0.0
            self.price_velocity = 0.0
            self.hour_of_day = 12
            self.final_outcome = outcome
            self.recorded_at = datetime.now(timezone.utc)
            
    replays = {
        "m1": MarketReplay([MockSnap("m1", "YES"), MockSnap("m1", "YES"), MockSnap("m1", "YES")]),
        "m2": MarketReplay([MockSnap("m2", "NO"), MockSnap("m2", "NO"), MockSnap("m2", "NO")]),
        "m3": MarketReplay([MockSnap("m3", "YES"), MockSnap("m3", "YES"), MockSnap("m3", "YES")]),
    }
    
    res = _build_result(
        run_id="run-1",
        config=config,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        total_loaded=9,
        tradeable=3,
        skipped=0,
        trades=trades,
        replays=replays
    )
    
    pnls = [t.trade_pnl for t in res.worst_trades]
    assert pnls == sorted(pnls), "worst_trades должен быть отсортирован по возрастанию PnL"
