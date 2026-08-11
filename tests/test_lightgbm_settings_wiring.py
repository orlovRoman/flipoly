import pandas as pd
import pytest


class _SessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _BacktestResult:
    symbol = "BTCUSDT"
    n_candles_total = 600
    n_candles_test = 120
    n_trades = 3
    win_rate = 2 / 3
    total_return_net = 0.01
    sharpe_ratio = 1.0
    max_drawdown = 0.02
    edge_rate = 0.1
    epsilon = 0.03
    train_auc = 0.61
    pnl_curve = []

    def is_profitable(self):
        return True

    def summary(self):
        return {}


@pytest.mark.asyncio
async def test_dashboard_backtest_passes_saved_lgbm_settings(monkeypatch):
    """Dashboard backtest uses the same saved controls as the form."""
    import polyflip.api.crypto_dashboard as dashboard
    import polyflip.services.settings_service as settings_service

    floats = {
        "BACKTEST_MIN_EDGE": 0.04,
        "LGBM_EPSILON_QUANTILE": 0.9,
        "CRYPTO_LGBM_LEARNING_RATE": 0.02,
        "CRYPTO_LGBM_SUBSAMPLE": 0.8,
        "CRYPTO_LGBM_COLSAMPLE_BYTREE": 1.0,
        "CRYPTO_LGBM_REG_ALPHA": 0.1,
        "CRYPTO_LGBM_REG_LAMBDA": 1.0,
    }
    ints = {
        "CRYPTO_LGBM_NUM_LEAVES": 15,
        "CRYPTO_LGBM_MAX_DEPTH": 4,
        "CRYPTO_LGBM_MIN_CHILD_SAMPLES": 50,
        "CRYPTO_LGBM_N_ESTIMATORS": 300,
    }
    captured = {}

    async def fake_get_float(_session, key):
        return floats[key]

    async def fake_get_int(_session, key):
        return ints[key]

    async def fake_candles(*_args, **_kwargs):
        return [object()] * 600

    async def fake_to_thread(func, *args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _BacktestResult()

    monkeypatch.setattr(settings_service, "get_float", fake_get_float)
    monkeypatch.setattr(settings_service, "get_int", fake_get_int)
    monkeypatch.setattr(dashboard, "async_session", lambda: _SessionContext())
    monkeypatch.setattr(dashboard, "get_recent_candles", fake_candles)
    monkeypatch.setattr(
        dashboard, "build_features",
        lambda _candles: pd.DataFrame({"x": [1]}),
    )
    monkeypatch.setattr(dashboard.asyncio, "to_thread", fake_to_thread)

    result = await dashboard.crypto_backtest(symbol="BTCUSDT", feature_set="B")

    assert result["symbol"] == "BTCUSDT"
    assert captured["func"] is dashboard.run_backtest
    assert captured["kwargs"]["epsilon_quantile"] == 0.9
    assert captured["kwargs"]["lgbm_params"]["learning_rate"] == 0.02
    assert captured["kwargs"]["lgbm_params"]["n_estimators"] == 300
    assert captured["kwargs"]["lgbm_params"]["num_leaves"] == 15
    assert captured["kwargs"]["feature_set"] == "B"
    assert captured["kwargs"]["closed_candles"] is not None
