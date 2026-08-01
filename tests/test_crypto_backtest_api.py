import pytest
import pandas as pd
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch, MagicMock
from polyflip.api.main import app

@pytest.mark.asyncio
async def test_crypto_backtest_run_success():
    # Мокаем свечи и бэктестер
    mock_result = MagicMock()
    mock_result.symbol = "BTCUSDT"
    mock_result.n_candles_total = 1000
    mock_result.n_candles_test = 500
    mock_result.n_trades = 20
    mock_result.win_rate = 0.60
    mock_result.total_return = 0.15
    mock_result.total_return_net = 0.12
    mock_result.sharpe_ratio = 1.5
    mock_result.max_drawdown = 0.05
    mock_result.edge_rate = 0.10
    mock_result.epsilon = 0.005
    mock_result.train_auc = 0.65
    mock_result.is_profitable.return_value = True
    mock_result.summary.return_value = "BTCUSDT Sharpe=1.5"
    mock_result.pnl_curve = []

    with patch("polyflip.api.crypto_backtest_api.get_recent_candles") as mock_candles, \
         patch("polyflip.api.crypto_backtest_api.run_backtest") as mock_run, \
         patch("polyflip.api.crypto_backtest_api.build_features") as mock_features:
        
        # Возвращаем 600 фейковых свечей
        mock_candles.return_value = [object()] * 600
        mock_features.return_value = pd.DataFrame(columns=["ret_1", "vol_6"])
        mock_run.return_value = mock_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            headers = {"X-API-Key": "test-key"}
            payload = {
                "symbol": "BTCUSDT",
                "interval": "15m",
                "days": 60,
                "features": ["ret_1", "vol_6"]
            }
            response = await ac.post("/api/crypto/backtest/run", json=payload, headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["symbol"] == "BTCUSDT"
            assert data["win_rate"] == 0.60
            assert data["is_profitable"] is True

@pytest.mark.asyncio
async def test_crypto_backtest_run_insufficient_data():
    with patch("polyflip.api.crypto_backtest_api.get_recent_candles") as mock_candles:
        mock_candles.return_value = [object()] * 100  # мало свечей

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            headers = {"X-API-Key": "test-key"}
            payload = {"symbol": "BTCUSDT", "interval": "15m", "days": 60}
            response = await ac.post("/api/crypto/backtest/run", json=payload, headers=headers)
            
            assert response.status_code == 422
            assert "Not enough candles" in response.json()["detail"]


@pytest.mark.asyncio
async def test_crypto_backtest_passes_lgbm_params_to_run_backtest(
    db_session, monkeypatch
):
    """Проверяем что API передаёт lgbm_params из БД в run_backtest."""
    from polyflip.db.models import RuntimeSettings
    # Кладём кастомный параметр в БД
    from datetime import datetime, timezone
    db_session.add(RuntimeSettings(key="CRYPTO_LGBM_N_ESTIMATORS", value="7", updated_by="test", updated_at=datetime.now(timezone.utc)))
    await db_session.commit()

    captured = {}
    def fake_run_backtest(df, symbol, min_edge, commission, **kwargs):
        captured.update(kwargs)
        class FakeResult:
            symbol = "BTCUSDT"
            n_candles_total = 1000
            n_candles_test = 500
            n_trades = 10
            win_rate = 0.5
            total_return_net = 0.05
            sharpe_ratio = 1.0
            max_drawdown = 0.01
            edge_rate = 0.1
            epsilon = 0.001
            train_auc = 0.55
            pnl_curve = []
            def is_profitable(self): return True
            def summary(self): return "ok"
        return FakeResult()

    monkeypatch.setattr("polyflip.api.crypto_dashboard.run_backtest", fake_run_backtest)
    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # no-op to satisfy SonarQube rule
            pass
    monkeypatch.setattr("polyflip.api.crypto_dashboard.async_session", lambda: DummyAsyncContextManager(db_session))
    
    # Мокаем get_recent_candles чтобы не требовать реальных данных
    async def fake_get_recent_candles(*args, **kwargs):
        class FakeCandle:
            def __init__(self):
                self.open_time = 0
                self.open = 1.0
                self.high = 1.0
                self.low = 1.0
                self.close = 1.0
                self.volume = 1.0
                self.taker_buy_volume = 1.0
                self.is_closed = True
                self.symbol = "BTCUSDT"
                self.interval = "15m"
        return [FakeCandle() for _ in range(601)]
    monkeypatch.setattr("polyflip.api.crypto_dashboard.get_recent_candles", fake_get_recent_candles)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        headers = {"X-API-Key": "test-key"}
        resp = await client.get("/crypto/api/backtest?symbol=BTCUSDT&interval=15m", headers=headers)
        assert resp.status_code == 200
        assert captured.get("lgbm_params", {}).get("n_estimators") == 7
