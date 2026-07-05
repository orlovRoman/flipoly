"""
Join Binance candles с историческими ценами Polymarket (MarketSnapshot).

Алгоритм:
  1. Загрузить все MarketSnapshot для нужного актива и периода.
  2. Отсортировать оба DataFrame по времени.
  3. pd.merge_asof с tolerance=tolerance_sec → nearest match.
  4. Вернуть расширенный DataFrame с pm_yes_price, pm_outcome, pm_market_id.

Использование в backtester:
  df_merged = await join_polymarket_prices(session, df_candles, "BTC")
  run_backtest(..., polymarket_prices=df_merged, pnl_mode="polymarket")
"""
from __future__ import annotations

from datetime import timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.db.models import MarketSnapshot

logger = structlog.get_logger(__name__)


async def join_polymarket_prices(
    session: AsyncSession,
    df_candles: pd.DataFrame,      # колонка open_time (tz-aware, UTC)
    asset: str,                    # "BTC" или "ETH"
    tolerance_sec: int = 450,      # ±7.5 мин для 15m-свечей
) -> pd.DataFrame:
    """
    Для каждой свечи находит ближайший MarketSnapshot по времени.

    Добавляет колонки:
      pm_yes_price   — mid_price (цена YES-токена, 0..1)
      pm_outcome     — "YES" | "NO" | "INVALID"
      pm_market_id   — для отладки

    Строки без совпадения → pm_yes_price = NaN, pm_outcome = None.
    """
    if df_candles.empty:
        logger.warning("join_polymarket_prices: empty candles df", asset=asset)
        return df_candles.copy()

    # Диапазон времени из свечей
    t_min = df_candles["open_time"].min()
    t_max = df_candles["open_time"].max()

    # Загружаем снапшоты за нужный период одним запросом
    stmt = (
        select(
            MarketSnapshot.market_id,
            MarketSnapshot.mid_price,
            MarketSnapshot.final_outcome,
            MarketSnapshot.recorded_at,
        )
        .where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.recorded_at >= t_min,
            MarketSnapshot.recorded_at <= t_max,
            MarketSnapshot.final_outcome.in_(["YES", "NO"]),  # пропускаем INVALID и незавершённые
        )
        .order_by(MarketSnapshot.recorded_at)
    )
    rows = (await session.execute(stmt)).all()

    if not rows:
        logger.warning(
            "join_polymarket_prices: no snapshots found",
            asset=asset,
            t_min=str(t_min),
            t_max=str(t_max),
        )
        df_out = df_candles.copy()
        df_out["pm_yes_price"] = float("nan")
        df_out["pm_outcome"]   = None
        df_out["pm_market_id"] = None
        return df_out

    df_snaps = pd.DataFrame(
        [(r.market_id, r.mid_price, r.final_outcome, r.recorded_at) for r in rows],
        columns=["pm_market_id", "pm_yes_price", "pm_outcome", "recorded_at"],
    )

    # Нормализуем timezone: оба должны быть tz-aware UTC
    df_snaps["recorded_at"] = pd.to_datetime(df_snaps["recorded_at"], utc=True)
    df_snaps = df_snaps.sort_values("recorded_at").reset_index(drop=True)

    df_left = df_candles.copy()
    df_left["open_time"] = pd.to_datetime(df_left["open_time"], utc=True)
    df_left = df_left.sort_values("open_time").reset_index(drop=True)

    # merge_asof: для каждой свечи — ближайший снапшот по времени
    tolerance = pd.Timedelta(seconds=tolerance_sec)
    df_merged = pd.merge_asof(
        df_left,
        df_snaps,
        left_on="open_time",
        right_on="recorded_at",
        direction="nearest",
        tolerance=tolerance,
    )

    matched = df_merged["pm_yes_price"].notna().sum()
    total   = len(df_merged)
    coverage_pct = round(matched / total * 100, 1) if total > 0 else 0.0

    logger.info(
        "join_polymarket_prices: done",
        asset=asset,
        total_candles=total,
        matched=matched,
        coverage_pct=coverage_pct,
        snapshots_loaded=len(df_snaps),
    )

    # Убираем вспомогательную колонку recorded_at
    df_merged = df_merged.drop(columns=["recorded_at"], errors="ignore")

    return df_merged
