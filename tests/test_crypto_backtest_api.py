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
