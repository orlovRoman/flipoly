import pytest
from datetime import datetime, timezone
from unittest.mock import patch
from polyflip.crypto.historical_loader import load_history, load_history_all

@pytest.mark.asyncio
async def test_load_history_idempotent(db_session):
    """Повторный вызов за тот же период возвращает 0."""
    with patch("polyflip.crypto.historical_loader.asyncio.to_thread") as mock_thread:
        mock_thread.return_value = [
            {
                "open_time": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "open": 50000.0, "high": 51000.0, "low": 49000.0, "close": 50500.0,
                "volume": 100.0, "taker_buy_volume": 55.0
            }
        ]
        count1 = await load_history(db_session, "BTCUSDT", "15m", days_back=1)
        count2 = await load_history(db_session, "BTCUSDT", "15m", days_back=1)

    assert count1 == 1
    assert count2 == 0   # ON CONFLICT DO NOTHING

@pytest.mark.asyncio
async def test_load_history_returns_negative_on_error(db_session):
    """load_history_all пишет -1 при сетевой ошибке, не падает."""
    with patch("polyflip.crypto.historical_loader.load_history", side_effect=Exception("network")):
        results = await load_history_all(db_session, symbols=["BTCUSDT"], intervals=["15m"])
    assert results["BTCUSDT_15m"] == -1
