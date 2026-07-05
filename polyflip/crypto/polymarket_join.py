"""
Join Binance candles с историческими ценами Polymarket (MarketSnapshot).

Алгоритм:
  1. Загрузить все MarketSnapshot для нужного актива и периода
     (расширенного на ±tolerance_sec, чтобы не пропустить снапшоты на границах).
  2. Отсортировать оба DataFrame по времени.
  3. pd.merge_asof с tolerance=tolerance_sec → nearest match.
  4. Вернуть расширенный DataFrame с pm_yes_price, pm_outcome, pm_market_id.

Использование в backtester:
  df_merged = await join_polymarket_prices(session, df_candles, "BTC")
  run_backtest(..., polymarket_prices=df_merged, pnl_mode="polymarket")
"""
from __future__ import annotations

from datetime import timedelta

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
      pm_outcome     — "YES" | "NO"
      pm_market_id   — для отладки

    SQL-запрос расширяет диапазон на ±tolerance_sec, чтобы не пропустить
    снапшоты случайно попавшие незадолго до/после границ t_min / t_max.

    Строки без совпадения → pm_yes_price = NaN, pm_outcome = None.
    INVALID снапшоты отфильтрованы на уровне SQL.
    """
    if df_candles.empty:
        logger.warning("join_polymarket_prices: empty candles df", asset=asset)
        return df_candles.copy()

    t_min = df_candles["open_time"].min()
    t_max = df_candles["open_time"].max()

    # FIX: расширяем диапазон на ±tolerance_sec — не потеряем снапшоты на границах
    # Например: первая свеча 12:00, снапшот 11:58 (в пределах 7.5 мин) — попадёт
    _tolerance = timedelta(seconds=tolerance_sec)
    t_query_min = t_min - _tolerance
    t_query_max = t_max + _tolerance

    stmt = (
        select(
            MarketSnapshot.market_id,
            MarketSnapshot.mid_price,
            MarketSnapshot.final_outcome,
            MarketSnapshot.recorded_at,
        )
        .where(
            MarketSnapshot.asset == asset,
            MarketSnapshot.recorded_at >= t_query_min,
            MarketSnapshot.recorded_at <= t_query_max,
            MarketSnapshot.final_outcome.in_(["YES", "NO"]),  # INVALID отфильтрован в SQL
        )
        .order_by(MarketSnapshot.recorded_at)
    )
    rows = (await session.execute(stmt)).all()

    if not rows:
        logger.warning(
            "join_polymarket_prices: no snapshots found",
            asset=asset,
            t_query_min=str(t_query_min),
            t_query_max=str(t_query_max),
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

    # Нормализуем timezone: оба должны быть datetime64[ns, UTC]
    df_snaps["recorded_at"] = pd.to_datetime(df_snaps["recorded_at"], utc=True).dt.tz_convert("UTC")
    df_snaps = df_snaps.sort_values("recorded_at").reset_index(drop=True)

    df_left = df_candles.copy()
    # FIX: явное приведение к datetime64[ns, UTC] — защита от несовместимых tz-типов pandas
    df_left["open_time"] = pd.to_datetime(df_left["open_time"], utc=True).dt.tz_convert("UTC")
    df_left = df_left.sort_values("open_time").reset_index(drop=True)

    tolerance_td = pd.Timedelta(seconds=tolerance_sec)
    df_merged = pd.merge_asof(
        df_left,
        df_snaps,
        left_on="open_time",
        right_on="recorded_at",
        direction="nearest",
        tolerance=tolerance_td,
    )

    matched      = df_merged["pm_yes_price"].notna().sum()
    total        = len(df_merged)
    coverage_pct = round(matched / total * 100, 1) if total > 0 else 0.0

    logger.info(
        "join_polymarket_prices: done",
        asset=asset,
        total_candles=total,
        matched=matched,
        coverage_pct=coverage_pct,
        snapshots_loaded=len(df_snaps),
    )

    df_merged = df_merged.drop(columns=["recorded_at"], errors="ignore")
    return df_merged
