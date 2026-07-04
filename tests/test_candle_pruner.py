import pytest
from datetime import datetime, timedelta, timezone
from polyflip.db.models import CryptoCandle
from polyflip.crypto.candle_pruner import prune_old_candles

@pytest.mark.asyncio
async def test_pruner_removes_old_candles(db_session):
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    fresh_time = datetime.now(timezone.utc) - timedelta(days=10)

    db_session.add_all([
        CryptoCandle(
            symbol="BTCUSDT", interval="15m", open_time=old_time,
            open=50000.0, high=51000.0, low=49000.0, close=50500.0,
            volume=100.0, taker_buy_volume=55.0
        ),
        CryptoCandle(
            symbol="BTCUSDT", interval="15m", open_time=fresh_time,
            open=50000.0, high=51000.0, low=49000.0, close=50500.0,
            volume=100.0, taker_buy_volume=55.0
        ),
    ])
    await db_session.commit()

    deleted = await prune_old_candles(db_session, retention_days=90)
    assert deleted == 1   # только старая

@pytest.mark.asyncio
async def test_pruner_idempotent(db_session):
    """Второй вызов удаляет 0 строк — нечего удалять."""
    deleted_first  = await prune_old_candles(db_session, retention_days=90)
    deleted_second = await prune_old_candles(db_session, retention_days=90)
    assert deleted_second == 0
