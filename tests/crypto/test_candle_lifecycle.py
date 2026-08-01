import pytest
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.crypto.candle_repository import upsert_candles, get_recent_candles

@pytest.mark.asyncio
async def test_candle_lifecycle(db_session: AsyncSession):
    symbol = "BTCUSDT"
    interval = "15m"
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    
    # 1. Свеча открыта (is_closed=False)
    open_candle = [{
        "open_time": now_utc - datetime.timedelta(minutes=10),
        "close_time": now_utc + datetime.timedelta(minutes=5),
        "is_closed": False,
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
        "taker_buy_volume": 500.0,
    }]
    
    inserted = await upsert_candles(db_session, symbol, interval, open_candle)
    assert inserted == 1
    
    # get_recent_candles не должен видеть открытую свечу
    recent = await get_recent_candles(db_session, symbol, interval, limit=10)
    # Предполагаем, что БД пуста, либо мы фильтруем её по времени (упрощенно проверяем, что ее нет)
    assert all(c.open_time != open_candle[0]["open_time"] for c in recent)
    
    # 2. Повторный вызов возвращает свечу закрытой (is_closed=True, данные обновились)
    closed_candle = [{
        "open_time": open_candle[0]["open_time"],
        "close_time": open_candle[0]["close_time"],
        "is_closed": True,
        "open": 100.0,
        "high": 110.0, # High изменился
        "low": 95.0,
        "close": 108.0, # Close изменился
        "volume": 2000.0,
        "taker_buy_volume": 1200.0,
    }]
    
    # upsert_candles обновит строку
    inserted2 = await upsert_candles(db_session, symbol, interval, closed_candle)
    assert inserted2 > 0
    
    # get_recent_candles теперь видит её
    recent2 = await get_recent_candles(db_session, symbol, interval, limit=10)
    found_candle = recent2[0] if recent2 else None
    
    assert found_candle is not None
    assert found_candle.is_closed is True
    assert found_candle.high == 110.0
    assert found_candle.close == 108.0
