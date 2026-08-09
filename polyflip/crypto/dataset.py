"""
polyflip/crypto/dataset.py

Модуль построения строгого обучающего датасета для LightGBM моделей с выравниванием pm_window_v1:
  - Исход: LiveMarket / MarketSnapshot final_outcome (YES -> 1, NO -> 0)
  - Фичи: Binance closed 15m свечи до market_start включительно (без lookahead)
  - Дедупликация: ровно 1 запись на market_id (без размножения снапшотами)
  - Без epsilon-фильтрации по будущему движению Binance
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


async def build_polymarket_training_dataset(
    db: AsyncSession,
    symbol: str,
    interval: str = "15m",
    limit_candles: int = 20_000,
) -> pd.DataFrame:
    """
    Строит датасет для обучения LightGBM на исходах Polymarket (Chainlink resolution).
    
    Схема выравнивания (pm_window_v1):
      - Для рынка 09:00 - 09:15 UTC:
        market_start = end_time_est - 15m = 09:00
        feature_available_at = close_time последней закрытой свечи (08:45-09:00) = 08:59:59.999
        target = final_outcome рынка 09:00-09:15 (YES=1, NO=0)
    """
    asset = symbol.removesuffix("USDT")
    logger.info("building_polymarket_dataset_start", symbol=symbol, asset=asset, interval=interval)

    # 1. Загрузка решённых рынков Polymarket (1 уникальная строка на market_id)
    res_markets = await db.execute(text(
        """
        SELECT DISTINCT ON (m.market_id)
            m.market_id,
            m.asset,
            m.end_time_est,
            s.final_outcome
        FROM live_markets m
        JOIN market_snapshots s ON m.market_id = s.market_id
        WHERE m.asset = :asset
          AND s.final_outcome IN ('YES', 'NO')
        ORDER BY m.market_id, s.recorded_at DESC;
        """
    ), {"asset": asset})
    
    market_rows = res_markets.fetchall()
    if not market_rows:
        logger.warning("no_resolved_polymarket_markets_found", symbol=symbol, asset=asset)
        return pd.DataFrame()

    df_markets = pd.DataFrame(market_rows, columns=["market_id", "asset", "end_time_est", "final_outcome"])
    
    # Приводим к datetime UTC
    df_markets["end_time_est"] = pd.to_datetime(df_markets["end_time_est"], utc=True)
    df_markets["market_start"] = df_markets["end_time_est"] - pd.Timedelta(minutes=15)
    df_markets["target"] = df_markets["final_outcome"].map({"YES": 1, "NO": 0}).astype(int)

    # 2. Загрузка исторических свечей Binance
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
    if len(candle_rows) < 200:
        logger.warning("insufficient_candles_for_dataset", symbol=symbol, count=len(candle_rows))
        return pd.DataFrame()

    df_candles = pd.DataFrame(candle_rows, columns=[
        "open_time", "close_time", "open", "high", "low", "close", "volume", "taker_buy_volume"
    ])
    df_candles["open_time"] = pd.to_datetime(df_candles["open_time"], utc=True)
    df_candles["close_time"] = pd.to_datetime(df_candles["close_time"], utc=True)
    
    # 3. Построение фичей по окну из 100 свечей для каждого 15m шага
    feature_records = []
    min_window = 100
    
    # Для эффективности используем векторные вычисления feature_builder по скользящему окну
    for idx in range(min_window, len(df_candles)):
        sub_df = df_candles.iloc[idx - min_window : idx + 1]
        fv = build_crypto_features(sub_df, min_candles=min_window)
        if fv.valid:
            row_dict = dict(zip(CRYPTO_FEATURE_COLUMNS, fv.features[0]))
            # Точный момент доступности признаков — время закрытия последней свечи окна
            row_dict["feature_available_at"] = sub_df.iloc[-1]["close_time"]
            feature_records.append(row_dict)

    if not feature_records:
        logger.warning("no_valid_feature_vectors_generated", symbol=symbol)
        return pd.DataFrame()

    df_features = pd.DataFrame(feature_records)
    df_features["feature_available_at"] = pd.to_datetime(df_features["feature_available_at"], utc=True)

    # 4. Слияние через merge_asof backward (без lookahead)
    # feature_available_at (08:59:59.999) <= market_start (09:00:00.000)
    df_markets_sorted = df_markets.sort_values("market_start").reset_index(drop=True)
    df_features_sorted = df_features.sort_values("feature_available_at").reset_index(drop=True)

    dataset = pd.merge_asof(
        df_markets_sorted,
        df_features_sorted,
        left_on="market_start",
        right_on="feature_available_at",
        direction="backward",
        tolerance=pd.Timedelta(seconds=5), # Допуск до 5 секунд для возможной задержки миллисекунд
    )

    # 5. Очистка и дедупликация
    dataset = dataset.dropna(subset=["feature_available_at", "target"]).copy()
    dataset = dataset.drop_duplicates("market_id").reset_index(drop=True)

    logger.info(
        "polymarket_dataset_built_success",
        symbol=symbol,
        total_markets=len(df_markets),
        matched_dataset_rows=len(dataset),
        target_yes_cnt=int((dataset["target"] == 1).sum()),
        target_no_cnt=int((dataset["target"] == 0).sum()),
    )

    return dataset
