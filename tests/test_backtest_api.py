# tests/test_backtest_api.py
import pytest
import pickle
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from polyflip.api.main import app
from polyflip.api.backtest_api import _compute_max_drawdown, _build_result
from polyflip.api.backtest_schemas import EquityCurvePoint, BacktestConfig


# Тест утилиты max drawdown
def test_max_drawdown_flat():
    curve = [
        EquityCurvePoint(trade_index=i, cumulative_pnl=float(i),
            trade_pnl=1.0, market_id="m1", asset="BTC",
            strategy="ML_TREND", outcome="WIN",
            p_flip=0.2, edge=0.1, bet_size=10, executed_price=0.7)
        for i in range(5)
    ]
    dd = _compute_max_drawdown(curve)
    assert dd == pytest.approx(0.0, abs=1.0)


def test_max_drawdown_with_loss():
    # Peak=10, then drop to 5 → drawdown 50%
    pnls = [10.0, -5.0]
    cumulative = 0.0
    curve = []
    for i, pnl in enumerate(pnls):
        cumulative += pnl
        curve.append(EquityCurvePoint(
            trade_index=i, cumulative_pnl=cumulative,
            trade_pnl=pnl, market_id="m1", asset="BTC",
            strategy="ML_TREND", outcome="WIN" if pnl > 0 else "LOSS",
            p_flip=0.2, edge=0.1, bet_size=10, executed_price=0.7
        ))
    assert _compute_max_drawdown(curve) == pytest.approx(50.0, abs=1.0)


@pytest.mark.asyncio
async def test_run_backtest_no_data_returns_422():
    """Если нет данных в БД — должен вернуть статус failed"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-API-Key": "test-key"}
        payload = {
            "assets": ["NONEXISTENT_ASSET_XYZ"],
            "strategy_mode": "PURE_FAVORITE"
        }
        # Мокируем async_session чтобы вернуть пустой список
        with patch("polyflip.api.backtest_api.async_session") as mock_session_maker:
            mock_session = AsyncMock()
            mock_session.execute.return_value.all = MagicMock(return_value=[])
            mock_session_maker.return_value = MagicMock()
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            
            resp = await client.post("/api/backtest/submit", json=payload, headers=headers)
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]
            
            import asyncio
            status_data = {}
            for _ in range(20):
                await asyncio.sleep(0.1)
                status_resp = await client.get(f"/api/backtest/status/{run_id}", headers=headers)
                assert status_resp.status_code == 200
                status_data = status_resp.json()
                if status_data["status"] in ("completed", "failed"):
                    break
            
            assert status_data["status"] == "failed"
            assert "No resolved snapshots" in status_data["error"]


@pytest.mark.asyncio
async def test_get_result_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-API-Key": "test-key"}
        resp = await client.get("/api/backtest/result/nonexistent-run-id", headers=headers)
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_history_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-API-Key": "test-key"}
        resp = await client.get("/api/backtest/history", headers=headers)
        assert resp.status_code == 200
        assert "runs" in resp.json()
