"""
polyflip/crypto/market_outcome_dataset.py

Построитель торгового датасета для LightGBM на канонических исходах Polymarket (Chainlink resolution).
Каждая строка соответствует ровно одному рынку (market_id) с фичами Binance-свечей,
закрытых ДО начала этого рынка (feature_candle_close <= market_start, feature_available_at <= market_start).
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


async def build_market_outcome_dataset(
    db: AsyncSession,
    symbol: str,
    interval: str = "15m",
) -> pd.DataFrame:
    """
    Создает торговый датасет для выравнивания MARKET_WINDOW_V1:
      - 1 строка на один market_id (однозначный target: YES=1, NO=0).
      - Детерминированный SQL-выбор через LEFT JOIN LATERAL (последняя финализированная запись).
      - Точное временное выравнивание: market_start = end_time_est - 15m.
      - Фичи закрытых свечей: feature_candle_close <= market_start и feature_available_at <= market_start.
    """
    asset = symbol.removesuffix("USDT")
    logger.info("building_market_outcome_dataset_start", symbol=symbol, asset=asset, interval=interval)

    # 4. Детерминированный выбор исхода рынка через LEFT JOIN LATERAL
    res_markets = await db.execute(text(
        """
        SELECT
            m.market_id,
            m.asset,
            m.end_time_est,
            COALESCE(m.final_outcome, latest.final_outcome) AS final_outcome
        FROM live_markets m
        LEFT JOIN LATERAL (
            SELECT s.final_outcome
            FROM market_snapshots s
            WHERE s.market_id = m.market_id
              AND s.final_outcome IN ('YES', 'NO')
            ORDER BY s.recorded_at DESC
            LIMIT 1
        ) latest ON TRUE
        WHERE m.asset = :asset
          AND m.end_time_est IS NOT NULL
          AND COALESCE(m.final_outcome, latest.final_outcome) IN ('YES', 'NO');
        """
    ), {"asset": asset})

    market_rows = res_markets.fetchall()
    if not market_rows:
        logger.warning("no_resolved_markets_found", symbol=symbol, asset=asset)
        return pd.DataFrame()

    markets = pd.DataFrame(market_rows, columns=["market_id", "asset", "end_time_est", "final_outcome"])
    assert markets["market_id"].is_unique, "SQL must guarantee 1 row per market_id"

    # Точное временное выравнивание
    markets["end_time_est"] = pd.to_datetime(markets["end_time_est"], utc=True)
    markets["market_start"] = markets["end_time_est"] - pd.Timedelta(minutes=15)
    markets["target"] = markets["final_outcome"].map({"YES": 1, "NO": 0}).astype(int)

    # 2. Загрузка закрытых свечей Binance
    res_candles = await db.execute(text(
        """
        SELECT open_time, close_time, open, high, low, close, volume, taker_buy_volume
        FROM crypto_candles
        WHERE symbol = :symbol
          AND interval = :interval
          AND is_closed = true
        ORDER BY open_time ASC;
        """
    ), {"symbol": symbol, "interval": interval})

    candle_rows = res_candles.fetchall()
    if len(candle_rows) < 105:
        logger.warning("insufficient_candles_for_market_dataset", symbol=symbol, count=len(candle_rows))
        return pd.DataFrame()

    df_candles = pd.DataFrame(candle_rows, columns=[
        "open_time", "close_time", "open", "high", "low", "close", "volume", "taker_buy_volume"
    ])
    df_candles["open_time"] = pd.to_datetime(df_candles["open_time"], utc=True)
    df_candles["close_time"] = pd.to_datetime(df_candles["close_time"], utc=True)

    # 3. Генерация векторов фичей по скользящему окну (100 свечей)
    feature_records = []
    min_window = 100

    for idx in range(min_window, len(df_candles)):
        sub_df = df_candles.iloc[idx - min_window : idx + 1]
        fv = build_crypto_features(sub_df, min_candles=min_window)
        if fv.valid:
            row_dict = dict(zip(CRYPTO_FEATURE_COLUMNS, fv.features[0]))
            # 5. Хранение двух временных меток доступности фичей
            last_candle = sub_df.iloc[-1]
            row_dict["feature_candle_close"] = last_candle["close_time"]
            row_dict["feature_available_at"] = last_candle["open_time"] + pd.Timedelta(minutes=15)
            feature_records.append(row_dict)

    if not feature_records:
        logger.warning("no_feature_records_generated", symbol=symbol)
        return pd.DataFrame()

    features = pd.DataFrame(feature_records)
    features["feature_candle_close"] = pd.to_datetime(features["feature_candle_close"], utc=True)
    features["feature_available_at"] = pd.to_datetime(features["feature_available_at"], utc=True)

    # 4. merge_asof backward: feature_available_at <= market_start
    markets_sorted = markets.sort_values("market_start").reset_index(drop=True)
    features_sorted = features.sort_values("feature_available_at").reset_index(drop=True)

    dataset = pd.merge_asof(
        markets_sorted,
        features_sorted,
        left_on="market_start",
        right_on="feature_available_at",
        direction="backward",
        tolerance=pd.Timedelta(seconds=2),
    )

    # Очистка строк без признаков
    dataset = dataset.dropna(subset=["feature_available_at", "target"]).reset_index(drop=True)

    # 5. Инвариант: проверяем feature_candle_close <= market_start и feature_available_at <= market_start
    if not dataset.empty:
        assert (dataset["feature_available_at"] <= dataset["market_start"]).all(), (
            "Invariant violation: feature_available_at must be <= market_start"
        )
        assert (dataset["feature_candle_close"] <= dataset["market_start"]).all(), (
            "Invariant violation: feature_candle_close must be <= market_start"
        )

    logger.info(
        "market_outcome_dataset_built",
        symbol=symbol,
        total_markets=len(markets),
        dataset_rows=len(dataset),
        yes_count=int((dataset["target"] == 1).sum()),
        no_count=int((dataset["target"] == 0).sum()),
    )

    return dataset
