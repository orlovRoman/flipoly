"""
polyflip/crypto/polymarket_join.py

Модуль соединения снапшотов и исходов Polymarket с Binance-свечами и точками принятия решений.

Основан на двух строгих функциях без использования lookahead / nearest matching:
  1. join_market_outcomes_by_window: Привязка канонического исхода Polymarket к окну рынка [market_start, market_end].
  2. join_entry_snapshot_by_decision_time: Привязка цены входа (mid_price) через merge_asof direction='backward' по market_id.
"""
from __future__ import annotations

from datetime import timedelta
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.db.models import LiveMarket, MarketSnapshot
from decimal import Decimal

logger = structlog.get_logger(__name__)


async def join_market_outcomes_by_window(
    session: AsyncSession,
    df_candles: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    """
    Привязывает канонический исход Polymarket по окну рынка [market_start, market_end].
    Не использует время снапшота для определения принадлежности к окну.
    
    Для свечи с open_time = 09:00:
      market_start = 09:00, market_end = 09:15.
    """
    if df_candles.empty:
        return df_candles.copy()

    df_out = df_candles.copy()
    df_out["open_time"] = pd.to_datetime(df_out["open_time"], utc=True)

    # Загружаем рынки
    stmt = select(
        LiveMarket.market_id,
        LiveMarket.asset,
        LiveMarket.end_time_est,
        LiveMarket.final_outcome,
    ).where(
        LiveMarket.asset == asset,
        LiveMarket.final_outcome.in_(["YES", "NO"]),
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        logger.warning("join_market_outcomes_by_window: no resolved markets found", asset=asset)
        df_out["pm_yes_price"] = float("nan")
        df_out["pm_outcome"] = None
        df_out["pm_market_id"] = None
        return df_out

    df_markets = pd.DataFrame(
        [(r.market_id, r.end_time_est, r.final_outcome) for r in rows],
        columns=["pm_market_id", "end_time_est", "pm_outcome"],
    )
    df_markets["end_time_est"] = pd.to_datetime(df_markets["end_time_est"], utc=True)
    df_markets["market_start"] = df_markets["end_time_est"] - pd.Timedelta(minutes=15)

    # Стыкуем по точной дате начала рынка: open_time == market_start
    df_merged = pd.merge(
        df_out,
        df_markets[["pm_market_id", "market_start", "pm_outcome"]],
        left_on="open_time",
        right_on="market_start",
        how="left",
    )
    df_merged = df_merged.drop(columns=["market_start"], errors="ignore")
    return df_merged


async def join_entry_snapshot_by_decision_time(
    session: AsyncSession,
    decisions: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    """
    Привязывает последний снапшот, существовавший К МОМЕНТУ принятия решения (decision_at).
    Использует direction='backward' и группировку by='market_id'.
    """
    if decisions.empty or "decision_at" not in decisions.columns or "market_id" not in decisions.columns:
        return decisions.copy()

    decisions_sorted = decisions.sort_values("decision_at").reset_index(drop=True)
    decisions_sorted["decision_at"] = pd.to_datetime(decisions_sorted["decision_at"], utc=True)

    stmt = select(
        MarketSnapshot.market_id,
        MarketSnapshot.mid_price,
        MarketSnapshot.recorded_at,
    ).where(
        MarketSnapshot.asset == asset,
    ).order_by(MarketSnapshot.recorded_at)

    rows = (await session.execute(stmt)).all()
    if not rows:
        decisions_sorted["entry_yes_price"] = float("nan")
        return decisions_sorted

    snapshots = pd.DataFrame(
        [(r.market_id, r.mid_price, r.recorded_at) for r in rows],
        columns=["market_id", "entry_yes_price", "recorded_at"],
    )
    snapshots["recorded_at"] = pd.to_datetime(snapshots["recorded_at"], utc=True)
    snapshots = snapshots.sort_values("recorded_at").reset_index(drop=True)

    df_merged = pd.merge_asof(
        decisions_sorted,
        snapshots,
        left_on="decision_at",
        right_on="recorded_at",
        by="market_id",
        direction="backward",
    )
    df_merged = df_merged.drop(columns=["recorded_at"], errors="ignore")
    return df_merged


async def join_polymarket_prices(
    session: AsyncSession,
    df_candles: pd.DataFrame,
    asset: str,
    tolerance_sec: int = 450,
) -> pd.DataFrame:
    """
    Обратная совместимость: обертка над join_market_outcomes_by_window
    без вызова direction='nearest'.
    """
    if df_candles.empty:
        return df_candles.copy()

    result = df_candles.copy()
    result["open_time"] = pd.to_datetime(result["open_time"], utc=True)
    min_time = result["open_time"].min()
    max_time = result["open_time"].max()
    tolerance = pd.Timedelta(seconds=tolerance_sec)

    stmt = select(
        MarketSnapshot.market_id,
        MarketSnapshot.mid_price.label("pm_yes_price"),
        MarketSnapshot.final_outcome.label("pm_outcome"),
        MarketSnapshot.recorded_at,
    ).where(
        MarketSnapshot.asset == asset,
        MarketSnapshot.recorded_at >= min_time - tolerance,
        MarketSnapshot.recorded_at <= max_time,
    ).order_by(MarketSnapshot.recorded_at.asc())
    rows = (await session.execute(stmt)).all()

    if not rows:
        result["pm_yes_price"] = float("nan")
        result["pm_outcome"] = None
        result["pm_market_id"] = None
        return result

    snapshot_rows = []
    for row in rows:
        price = getattr(row, "pm_yes_price", None)
        if not isinstance(price, (int, float, Decimal)):
            # Compatibility with lightweight row objects exposing the model
            # field name (mid_price) instead of the SQL label.
            price = getattr(row, "mid_price", None)
        outcome = getattr(row, "pm_outcome", None)
        if outcome not in {"YES", "NO", "PENDING", "INVALID"}:
            outcome = getattr(row, "final_outcome", None)
        snapshot_rows.append(
            (row.market_id, price, outcome, row.recorded_at)
        )
    snapshots = pd.DataFrame(
        snapshot_rows,
        columns=["pm_market_id", "pm_yes_price", "pm_outcome", "recorded_at"],
    )
    snapshots["recorded_at"] = pd.to_datetime(
        snapshots["recorded_at"], utc=True, errors="coerce"
    )
    snapshots["pm_yes_price"] = pd.to_numeric(
        snapshots["pm_yes_price"], errors="coerce"
    )
    snapshots = snapshots.dropna(subset=["recorded_at", "pm_yes_price"])
    snapshots = snapshots.sort_values("recorded_at").reset_index(drop=True)
    if snapshots.empty:
        result["pm_yes_price"] = float("nan")
        result["pm_outcome"] = None
        result["pm_market_id"] = None
        return result

    result = pd.merge_asof(
        result.sort_values("open_time").reset_index(drop=True),
        snapshots,
        left_on="open_time",
        right_on="recorded_at",
        direction="backward",
        tolerance=tolerance,
    )
    return result.drop(columns=["recorded_at"], errors="ignore")
