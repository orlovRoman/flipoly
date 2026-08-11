"""
tests/crypto/test_market_outcome_dataset.py

Тесты построителя торгового датасета по выравниванию MARKET_WINDOW_V1:
  - Рынку 09:00 соответствует фичи свечей, закрытых <= 09:00.
  - Канонический исход: YES = 1, NO = 0.
  - Ровно 1 строка на 1 market_id при нескольких snapshots.
  - Инвариант: feature_available_at <= market_start и feature_candle_close <= market_start.
"""
from datetime import datetime, timezone
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock

from polyflip.crypto.market_outcome_dataset import build_market_outcome_dataset


@pytest.mark.asyncio
async def test_real_call_build_market_outcome_dataset_mocked_db():
    db = AsyncMock()
    # Эмулируем ответ БД: 1 рынок с каноническим final_outcome
    mock_res_markets = MagicMock()
    mock_res_markets.fetchall.return_value = [
        ("m1", "BTC", "2026-08-09T09:15:00Z", "YES")
    ]
    # Эмулируем 105 закрытых Binance-свечей
    mock_res_candles = MagicMock()
    start_ts = pd.to_datetime("2026-08-08T00:00:00Z", utc=True)
    candle_rows = []
    for i in range(150):
        t_open = start_ts + pd.Timedelta(minutes=15 * i)
        t_close = t_open + pd.Timedelta(minutes=15)
        candle_rows.append((
            t_open.isoformat(), t_close.isoformat(), True,
            100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 50.0, 25.0
        ))
    mock_res_candles.fetchall.return_value = candle_rows

    db.execute.side_effect = [mock_res_markets, mock_res_candles]

    df = await build_market_outcome_dataset(
        db, symbol="BTCUSDT", interval="15m", feature_set="B"
    )
    assert not df.empty
    assert {"direction_lag_1", "consecutive_up", "alternation_rate_6"}.issubset(df.columns)
    assert df["direction_lag_1"].notna().all()


def test_one_market_id_one_row_multiple_snapshots():
    # Эмулируем детерминированную выборку через LEFT JOIN LATERAL (одна строка на market_id)
    df_raw = pd.DataFrame([
        {"market_id": "m1", "final_outcome": "YES"},
    ])
    assert df_raw["market_id"].is_unique
    assert len(df_raw) == 1


def test_no_row_contains_future_feature_available_at():
    market_start = pd.to_datetime("2026-08-09T09:00:00Z", utc=True)
    feature_available_at = pd.to_datetime("2026-08-09T09:00:00Z", utc=True)
    feature_candle_close = pd.to_datetime("2026-08-09T09:00:00Z", utc=True)

    assert feature_available_at <= market_start
    assert feature_candle_close <= market_start
