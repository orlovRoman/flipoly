# tests/test_metrics_bugfixes.py

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

def make_trades(pnl_list: list[float], bet: float = 10.0):
    """Хелпер: создаёт список SimulatedTrade-подобных объектов."""
    from polyflip.backtesting.simulated_trader import SimulatedTrade
    from polyflip.trading.decision_logic import TradeDecision
    trades = []
    for i, pnl in enumerate(pnl_list):
        decision = TradeDecision(
            action="BUY_NO",
            buy_price=0.5,
            bet_size_usdc=bet,
            reason="test",
            strategy_type="ML",
            p_flip=0.5,
            edge=0.1
        )
        t = SimulatedTrade(
            market_id=f"m{i}",
            asset="BTC",
            decision=decision,
            executed_price=0.5,
            slippage=0.0,
            bet_size=bet,
            shares=bet / 0.5,
            timestamp=datetime.now(),
            p_flip=0.5
        )
        # Hack to inject PnL directly since compute_metrics doesn't look at PnL from TradeHistory directly
        # Wait, compute_metrics uses compute_trade_pnl which looks at replay.final_outcome!
        # If we pass {}, replays is empty, compute_metrics will skip trades if no replay!
        # Let's check compute_metrics logic again.
        trades.append(t)
    return trades


# --- Ошибка 2: initial_capital должен влиять на max_drawdown_pct ---

def test_max_drawdown_uses_actual_capital(monkeypatch):
    """max_drawdown_pct должен зависеть от переданного капитала, не хардкода 1000."""
    import polyflip.backtesting.metrics as m
    # Monkeypatch compute_trade_pnl to return exactly the pnl we want
    pnl_returns = [-50.0, -30.0, 20.0]
    pnl_idx = [0]
    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, "compute_trade_pnl", mock_pnl)
    
    trades = make_trades([-50.0, -30.0, 20.0])
    replays = {f"m{i}": FakeReplay() for i in range(len(trades))}
    
    result_1000 = m.compute_metrics(trades, replays, initial_capital=1000.0)
    pnl_idx[0] = 0 # reset
    result_500 = m.compute_metrics(trades, replays, initial_capital=500.0)
    
    # При меньшем капитале drawdown в % должен быть больше
    assert result_500["max_drawdown_pct"] > result_1000["max_drawdown_pct"], (
        "Drawdown % должен быть больше при меньшем initial_capital"
    )


def test_max_drawdown_zero_if_no_losses(monkeypatch):
    """Если все сделки прибыльные — drawdown должен быть 0."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [10.0, 20.0, 5.0]
    pnl_idx = [0]
    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, "compute_trade_pnl", mock_pnl)
    
    trades = make_trades([10.0, 20.0, 5.0])
    replays = {f"m{i}": FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    assert result["max_drawdown_pct"] == 0.0


# --- Ошибка 3: profit_factor не должен быть float("inf") ---

def test_profit_factor_serializable_when_no_losses(monkeypatch):
    """profit_factor должен быть JSON-сериализуемым (None, не inf)."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [10.0, 20.0, 5.0]
    pnl_idx = [0]
    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, "compute_trade_pnl", mock_pnl)
    
    trades = make_trades([10.0, 20.0, 5.0])  # только прибыльные сделки
    replays = {f"m{i}": FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    
    # Проверяем что profit_factor не inf
    pf = result["profit_factor"]
    assert pf is None or (isinstance(pf, float) and pf != float("inf")), (
        f"profit_factor не должен быть float('inf'), получено: {pf}"
    )
    
    # Проверяем JSON-сериализуемость всего словаря
    try:
        json.dumps(result)
    except (ValueError, TypeError) as e:
        pytest.fail(f"Результат metrics не сериализуется в JSON: {e}")


def test_profit_factor_calculated_with_losses(monkeypatch):
    """При наличии убытков profit_factor = gross_profit / |gross_loss|."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [20.0, -10.0]
    pnl_idx = [0]
    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, "compute_trade_pnl", mock_pnl)
    
    trades = make_trades([20.0, -10.0])  # profit=20, loss=10 → pf=2.0
    replays = {f"m{i}": FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    assert abs(result["profit_factor"] - 2.0) < 1e-6


def test_sharpe_ratio_none_on_single_trade(monkeypatch):
    """При std=0 (одна сделка) sharpe должен быть None."""
    import polyflip.backtesting.metrics as m
    pnl_returns = [10.0]
    pnl_idx = [0]
    def mock_pnl(trade, replay):
        res = pnl_returns[pnl_idx[0]]
        pnl_idx[0] += 1
        return res
    monkeypatch.setattr(m, "compute_trade_pnl", mock_pnl)
    
    trades = make_trades([10.0])
    replays = {f"m{i}": FakeReplay() for i in range(len(trades))}
    result = m.compute_metrics(trades, replays, initial_capital=1000.0)
    assert result["sharpe_ratio"] is None


# --- Ошибка 1 (калибровка): ECE должен быть честным ---

def test_calibration_fit_on_val_not_train():
    """
    Проверяет что калибровка не даёт подозрительно низкий ECE
    (что было бы признаком data leakage train→calibration).
    """
    import numpy as np
    import pickle
    import pandas as pd
    from polyflip.models.trainer import _fit_and_serialize
    from sklearn.calibration import calibration_curve

    np.random.seed(0)
    n = 400
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
    })
    prob = 1 / (1 + np.exp(-X["feature1"] * 2))
    y = pd.Series((np.random.rand(n) < prob).astype(int))
    groups = pd.Series(np.repeat(np.arange(n // 2), 2))

    model_bytes, val_auc, baseline, threshold, ece = _fit_and_serialize(X, y, groups)

    # ECE должен быть float в [0, 1]
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0, f"ECE={ece:.4f} вне допустимого диапазона"

    # Подозрительно низкий ECE (< 0.001) — признак leakage
    assert ece > 0.001, (
        f"ECE={ece:.6f} подозрительно мал — возможен data leakage в калибровке"
    )

    # val_auc должен быть разумным
    assert val_auc > 0.5, f"val_auc={val_auc:.4f} хуже случайного — что-то не так"


# --- Ошибка 4: BET_SIZING_MODE должен корректно передаваться в runner ---

def test_runner_bet_sizing_mode_scaled():
    """Runner должен применять scaled bet sizing при BET_SIZING_MODE=scaled."""
    from polyflip.backtesting.runner import BacktestRunner

    config = {
        "BET_SIZING_MODE": "scaled",
        "TRADE_BET_SIZE_USDC": "5.0",
        "MAX_BET_SIZE_USDC": "50.0",
        "MIN_EDGE": "0.05",
        "MAX_EDGE": "0.50",
        "SLIPPAGE_PCT": "0.005",
        "STRATEGY_MODE": "ML",
        "TRADE_ON_FLIP": False,
    }
    import pickle

    runner = BacktestRunner(config, pickle.dumps(DummyModelScaled()), "feature1")
    assert runner.bet_sizing_mode == "scaled", (
        f"bet_sizing_mode должен быть 'scaled', получено '{runner.bet_sizing_mode}'"
    )


def test_runner_scaled_bet_size_interpolation():
    """Скейлинг ставки: при max edge → max bet, при min edge → base bet."""
    from polyflip.backtesting.runner import BacktestRunner
    import pickle

    config = {
        "BET_SIZING_MODE": "scaled",
        "TRADE_BET_SIZE_USDC": "5.0",
        "MAX_BET_SIZE_USDC": "50.0",
        "MIN_EDGE": "0.0",
        "MAX_EDGE": "1.0",
        "SLIPPAGE_PCT": "0.005",
        "STRATEGY_MODE": "ML",
        "TRADE_ON_FLIP": False,
    }
    runner = BacktestRunner(config, pickle.dumps(DummyModelInterpolation()), "feature1")

    class FakeDecision:
        edge = 0.0  # min_edge → base_bet

    assert runner._calc_bet_size(FakeDecision()) == pytest.approx(5.0)

    class FakeDecisionMax:
        edge = 1.0  # max_edge → max_bet

    assert runner._calc_bet_size(FakeDecisionMax()) == pytest.approx(50.0)

    class FakeDecisionMid:
        edge = 0.5  # середина → (5 + 50) / 2 = 27.5

    assert runner._calc_bet_size(FakeDecisionMid()) == pytest.approx(27.5)
