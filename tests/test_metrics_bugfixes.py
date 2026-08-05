import pytest
import json
import pandas as pd
from datetime import datetime
from polyflip.backtesting.metrics import compute_metrics

class FakeReplay:
    pass

class DummyModelScaled:

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.3, 0.7]] * len(X))

class DummyModelInterpolation:

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.5, 0.5]] * len(X))

def make_trades(pnl_list: list[float], bet: float=10.0):
    """Хелпер: создаёт список SimulatedTrade-подобных объектов."""
    from polyflip.backtesting.simulated_trader import SimulatedTrade
    from polyflip.trading.decision_logic import TradeDecision
    trades = []
    for i, pnl in enumerate(pnl_list):
        decision = TradeDecision(action='BUY_NO', buy_price=0.5, bet_size_usdc=bet, reason='test', strategy_type='ML', p_flip=0.5, edge=0.1)
        t = SimulatedTrade(market_id=f'm{i}', asset='BTC', decision=decision, executed_price=0.5, slippage=0.0, bet_size=bet, shares=bet / 0.5, timestamp=datetime.now(), p_flip=0.5)
        trades.append(t)
    return trades

def test_max_drawdown_uses_actual_capital(monkeypatch):
    """max_drawdown_pct должен зависеть от переданного капитала, не хардкода 1000."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [-50.0, -30.0, 20.0]
    pnl_idx = [0]

    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, 'compute_trade_pnl', mock_pnl)
    trades = make_trades([-50.0, -30.0, 20.0])
    replays = {f'm{i}': FakeReplay() for i in range(len(trades))}
    result_1000 = m.compute_metrics(trades, replays, initial_capital=1000.0)
    pnl_idx[0] = 0
    result_500 = m.compute_metrics(trades, replays, initial_capital=500.0)
    assert result_500['max_drawdown_pct'] > result_1000['max_drawdown_pct'], 'Drawdown % должен быть больше при меньшем initial_capital'

def test_max_drawdown_zero_if_no_losses(monkeypatch):
    """Если все сделки прибыльные — drawdown должен быть 0."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [10.0, 20.0, 5.0]
    pnl_idx = [0]

    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, 'compute_trade_pnl', mock_pnl)
    trades = make_trades([10.0, 20.0, 5.0])
    replays = {f'm{i}': FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    assert result['max_drawdown_pct'] == 0.0

def test_profit_factor_serializable_when_no_losses(monkeypatch):
    """profit_factor должен быть JSON-сериализуемым (None, не inf)."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [10.0, 20.0, 5.0]
    pnl_idx = [0]

    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, 'compute_trade_pnl', mock_pnl)
    trades = make_trades([10.0, 20.0, 5.0])
    replays = {f'm{i}': FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    pf = result['profit_factor']
    assert pf is None or (isinstance(pf, float) and pf != float('inf')), f"profit_factor не должен быть float('inf'), получено: {pf}"
    try:
        json.dumps(result)
    except (ValueError, TypeError) as e:
        pytest.fail(f'Результат metrics не сериализуется в JSON: {e}')

def test_profit_factor_calculated_with_losses(monkeypatch):
    """При наличии убытков profit_factor = gross_profit / |gross_loss|."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [20.0, -10.0]
    pnl_idx = [0]

    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, 'compute_trade_pnl', mock_pnl)
    trades = make_trades([20.0, -10.0])
    replays = {f'm{i}': FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    assert abs(result['profit_factor'] - 2.0) < 1e-06

def test_sharpe_ratio_none_on_single_trade(monkeypatch):
    """При std=0 (одна сделка) sharpe должен быть None."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [10.0]
    pnl_idx = [0]

    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, 'compute_trade_pnl', mock_pnl)
    trades = make_trades([10.0])
    replays = {f'm{i}': FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    assert result['sharpe_ratio'] is None

def test_runner_bet_sizing_mode_scaled():
    """Runner должен применять scaled bet sizing при BET_SIZING_MODE=scaled."""
    from polyflip.backtesting.runner import BacktestRunner
    config = {'BET_SIZING_MODE': 'scaled', 'TRADE_BET_SIZE_USDC': '5.0', 'MAX_BET_SIZE_USDC': '50.0', 'OUTS_MIN_EDGE': '0.05', 'MAX_BET_EDGE': '0.50', 'SLIPPAGE_PCT': '0.005', 'STRATEGY_MODE': 'OUTSIDER', 'TRADE_ON_FLIP': False}
    import pickle
    runner = BacktestRunner(config, pickle.dumps(DummyModelScaled()), 'feature1')
    assert runner.bet_sizing_mode == 'scaled', f"bet_sizing_mode должен быть 'scaled', получено '{runner.bet_sizing_mode}'"