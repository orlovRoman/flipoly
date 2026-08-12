"""
polyflip/crypto/market_outcome_dataset.py

Построитель торгового датасета для LightGBM на канонических исходах Polymarket (Chainlink resolution).
Каждая строка соответствует ровно одному рынку (market_id) с фичами Binance-свечей,
закрытых ДО начала этого рынка (feature_candle_close <= market_start, feature_available_at <= market_start).
"""
from __future__ import annotations
import hashlib
from collections import OrderedDict

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.feature_builder import build_features, CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.feature_sets import get_feature_set
from polyflip.models.sequence_features import attach_closed_candle_features, sequence_history_ready

_DATASET_CACHE_MAX = 12
_DATASET_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()

# Strike features depend on the market row, so they are materialized after the as-of join.
DATASET_FEATURE_COLUMNS = tuple(
    column for column in CRYPTO_FEATURE_COLUMNS
    if column not in {"strike_gap_pct", "log_moneyness"}
)


def clear_market_outcome_dataset_cache() -> None:
    """Clear prepared training datasets after candle/market ingestion."""
    _DATASET_CACHE.clear()


def _frame_fingerprint(*frames: pd.DataFrame) -> str:
    """Return a deterministic fingerprint for the raw inputs used by a dataset."""
    digest = hashlib.sha256()
    for frame in frames:
        normalized = frame.copy()
        normalized = normalized.sort_index(axis=1)
        digest.update("|".join(str(c) for c in normalized.columns).encode())
        digest.update(pd.util.hash_pandas_object(normalized, index=True).to_numpy().tobytes())
    return digest.hexdigest()[:24]


def _cache_get(key: str) -> pd.DataFrame | None:
    value = _DATASET_CACHE.get(key)
    if value is None:
        return None
    _DATASET_CACHE.move_to_end(key)
    return value.copy(deep=True)


def _cache_put(key: str, value: pd.DataFrame) -> None:
    _DATASET_CACHE[key] = value.copy(deep=True)
    _DATASET_CACHE.move_to_end(key)
    while len(_DATASET_CACHE) > _DATASET_CACHE_MAX:
        _DATASET_CACHE.popitem(last=False)

logger = structlog.get_logger(__name__)


async def build_market_outcome_dataset(
    db: AsyncSession,
    symbol: str,
    interval: str = "15m",
    feature_set: str = "A",
) -> pd.DataFrame:
    """
    Создает торговый датасет для выравнивания MARKET_WINDOW_V1:
      - 1 строка на один market_id (однозначный target: YES=1, NO=0).
      - Детерминированный SQL-выбор через LEFT JOIN LATERAL (последняя финализированная запись).
      - Точное временное выравнивание: market_start = end_time_est - 15m.
      - Фичи закрытых свечей: feature_candle_close <= market_start и feature_available_at <= market_start.
    """
    feature_spec = get_feature_set(feature_set)
    asset = symbol.removesuffix("USDT")
    logger.info(
        "building_market_outcome_dataset_start",
        symbol=symbol,
        asset=asset,
        interval=interval,
        feature_set=feature_spec.key,
        feature_set_version=feature_spec.version,
    )

    # 4. Детерминированный выбор исхода рынка через LEFT JOIN LATERAL
    res_markets = await db.execute(text(
        """
        SELECT
            m.market_id,
            m.asset,
            m.end_time_est,
            m.underlying_price,
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

    if market_rows and len(market_rows[0]) == 4:
        # Compatibility with lightweight test/fallback adapters that predate
        # the nullable canonical strike column.
        logger.warning(
            "market_outcome_dataset_legacy_schema",
            reason="underlying_price column missing; strike features default to zero",
        )
        markets = pd.DataFrame(
            market_rows, columns=["market_id", "asset", "end_time_est", "final_outcome"]
        )
        markets["underlying_price"] = np.nan
    else:
        markets = pd.DataFrame(
            market_rows,
            columns=["market_id", "asset", "end_time_est", "underlying_price", "final_outcome"],
        )
    assert markets["market_id"].is_unique, "SQL must guarantee 1 row per market_id"

    # Точное временное выравнивание
    markets["end_time_est"] = pd.to_datetime(markets["end_time_est"], utc=True)
    markets["market_start"] = markets["end_time_est"] - pd.Timedelta(minutes=15)
    markets["target"] = markets["final_outcome"].map({"YES": 1, "NO": 0}).astype(int)

    # 2. Загрузка закрытых свечей Binance
    res_candles = await db.execute(text(
        """
        SELECT open_time, close_time, is_closed, open, high, low, close, volume, taker_buy_volume
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
        "open_time", "close_time", "is_closed", "open", "high", "low", "close", "volume", "taker_buy_volume"
    ])
    df_candles["open_time"] = pd.to_datetime(df_candles["open_time"], utc=True)
    df_candles["close_time"] = pd.to_datetime(df_candles["close_time"], utc=True)
    dataset_fingerprint = _frame_fingerprint(
        markets[["market_id", "end_time_est", "underlying_price", "final_outcome"]],
        df_candles,
    )
    cache_key = f"{symbol}|{interval}|{dataset_fingerprint}|{feature_spec.key}"
    cached_dataset = _cache_get(cache_key)
    if cached_dataset is not None:
        logger.info(
            "market_outcome_dataset_cache_hit",
            symbol=symbol,
            feature_set=feature_spec.key,
            rows=len(cached_dataset),
        )
        return cached_dataset


    # 3. Генерация векторов фичей по скользящему окну (100 свечей)
    # Vectorized rolling features: calculate every column once instead of
    # rebuilding a 100-candle window for every market row.
    min_window = 100
    all_features = build_features(df_candles)
    feature_records = []
    for idx in range(min_window, len(df_candles)):
        candle = df_candles.iloc[idx]
        row_dict = {
            column: all_features.iloc[idx][column]
            for column in DATASET_FEATURE_COLUMNS
        }
        row_dict["feature_candle_close"] = candle["close_time"]
        row_dict["feature_available_at"] = candle["open_time"] + pd.Timedelta(minutes=15)
        row_dict["feature_close"] = candle["close"]
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

    if "feature_close" in dataset.columns:
        strike = pd.to_numeric(dataset["underlying_price"], errors="coerce")
        close = pd.to_numeric(dataset["feature_close"], errors="coerce")
        valid_strike = strike.gt(0) & close.gt(0)
        dataset["strike_gap_pct"] = np.where(
            valid_strike, (close - strike) / strike, 0.0
        )
        dataset["log_moneyness"] = np.where(
            valid_strike, np.log(close / strike), 0.0
        )
        dataset = dataset.drop(columns=["feature_close"])

    if not dataset.empty:
        assert {"strike_gap_pct", "log_moneyness"}.issubset(dataset.columns), (
            "Strike features must be materialized after the market join"
        )
        if not pd.to_numeric(dataset["underlying_price"], errors="coerce").gt(0).any():
            logger.warning(
                "market_outcome_dataset_missing_canonical_strike",
                symbol=symbol,
                rows=len(dataset),
            )

    # 5. Инвариант: проверяем feature_candle_close <= market_start и feature_available_at <= market_start
    if not dataset.empty:
        assert (dataset["feature_available_at"] <= dataset["market_start"]).all(), (
            "Invariant violation: feature_available_at must be <= market_start"
        )
        assert (dataset["feature_candle_close"] <= dataset["market_start"]).all(), (
            "Invariant violation: feature_candle_close must be <= market_start"
        )

    # B/C use only fully closed candles available at market_start.  The
    # backward as-of join keeps the timestamp contract explicit and prevents
    # future candle data from entering the training matrix.
    if feature_spec.key != "A":
        dataset = attach_closed_candle_features(
            dataset,
            df_candles.to_dict(orient="records"),
            decision_time_col="market_start",
        )
        sequence_ready = sequence_history_ready(dataset)
        dropped = int((~sequence_ready).sum())
        if dropped:
            logger.info(
                "sequence_rows_excluded_from_lgbm_dataset",
                symbol=symbol,
                feature_set=feature_spec.key,
                rows_dropped=dropped,
                rows_remaining=int(sequence_ready.sum()),
                min_history=6,
            )
        dataset = dataset.loc[sequence_ready].reset_index(drop=True)
        if dataset.empty:
            logger.warning(
                "sequence_feature_coverage_empty",
                symbol=symbol,
                feature_set=feature_spec.key,

                reason="SEQUENCE_COVERAGE_INSUFFICIENT",
            )
    _cache_put(cache_key, dataset)

    logger.info(

        "market_outcome_dataset_built",
        symbol=symbol,
        total_markets=len(markets),
        dataset_rows=len(dataset),
        yes_count=int((dataset["target"] == 1).sum()) if not dataset.empty else 0,
        no_count=int((dataset["target"] == 0).sum()) if not dataset.empty else 0,
        feature_set=feature_spec.key,
        feature_set_version=feature_spec.version,
    )

    return dataset
