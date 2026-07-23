import os
import time
import asyncio
from fastapi.templating import Jinja2Templates
from typing import Optional
from datetime import datetime, time as dt_time, timezone, timedelta
from fastapi import APIRouter, Request, Depends, Query

from sqlalchemy import select, func, cast, Date, case as sa_case
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.connection import get_db_session, async_session
from polyflip.db.models import TradeHistory, RuntimeSettings
from polyflip.config import settings
from polyflip.api.auth import verify_api_key
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["TradingDashboard"])
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

@router.get("/trading")
async def get_trading_dashboard(request: Request):
    return templates.TemplateResponse(
        "trading.html", 
        {
            "request": request,
            "timestamp": int(time.time()),
            "root_path": request.scope.get("root_path", ""), 
            "assets": settings.asset_list
        }
    )

_stats_cache = {}
_STATS_CACHE_TTL = 30  # 30 секунд кэша

def invalidate_stats_cache():
    _stats_cache.clear()

def _utc_cutoff(delta: timedelta) -> datetime:
    """Возвращает naive UTC datetime для сравнения с TIMESTAMP WITHOUT TIME ZONE."""
    return (datetime.now(timezone.utc) - delta).replace(tzinfo=None)

@router.get("/api/trading/stats", dependencies=[Depends(verify_api_key)])
async def get_trading_stats(
    timeframe: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db_session)
):
    current_time = time.time()
    cache_key = f"stats_{timeframe or 'all'}"
    if cache_key in _stats_cache and current_time - _stats_cache[cache_key]["time"] < _STATS_CACHE_TTL:
        return _stats_cache[cache_key]["data"]

    cutoff_dt = None
    if timeframe == "24h":
        cutoff_dt = _utc_cutoff(timedelta(hours=24))
    elif timeframe == "7d":
        cutoff_dt = _utc_cutoff(timedelta(days=7))
    elif timeframe == "30d":
        cutoff_dt = _utc_cutoff(timedelta(days=30))

    async def fetch_settings():
        async with async_session() as s:
            stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(["INITIAL_CAPITAL"]))
            res = await s.execute(stmt)
            return res.scalars().all()

    async def fetch_assets():
        async with async_session() as s:
            conds = [
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None)
            ]
            if cutoff_dt:
                conds.append(TradeHistory.created_at >= cutoff_dt)
            stmt = select(
                TradeHistory.asset,
                func.count(TradeHistory.id).label("total_trades"),
                func.sum(TradeHistory.pnl).label("total_pnl"),
                func.sum(sa_case((TradeHistory.pnl > 0, 1), else_=0)).label("wins")
            ).where(*conds).group_by(TradeHistory.asset)
            return (await s.execute(stmt)).all()

    async def fetch_daily():
        async with async_session() as s:
            conds = [
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None)
            ]
            if cutoff_dt:
                conds.append(TradeHistory.created_at >= cutoff_dt)
            stmt = select(
                cast(TradeHistory.created_at, Date).label("day"),
                func.sum(TradeHistory.pnl).label("daily_pnl"),
                func.sum(sa_case((TradeHistory.pnl > 0, 1), else_=0)).label("wins"),
                func.sum(sa_case((TradeHistory.pnl <= 0, 1), else_=0)).label("losses")
            ).where(*conds).group_by(cast(TradeHistory.created_at, Date))
            return (await s.execute(stmt)).all()

    async def fetch_params():
        async with async_session() as s:
            conds = [
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None)
            ]
            if cutoff_dt:
                conds.append(TradeHistory.created_at >= cutoff_dt)
            stmt = select(
                func.avg(sa_case((TradeHistory.pnl > 0, TradeHistory.executed_price), else_=None)).label("avg_win_price"),
                func.avg(sa_case((TradeHistory.pnl <= 0, TradeHistory.executed_price), else_=None)).label("avg_loss_price"),
                func.avg(sa_case((TradeHistory.pnl > 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_win_prob"),
                func.avg(sa_case((TradeHistory.pnl <= 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_loss_prob")
            ).where(*conds)
            return (await s.execute(stmt)).first()

    async def fetch_all_time_totals():
        async with async_session() as s:
            stmt = select(
                func.count(TradeHistory.id).label("total_trades"),
                func.sum(TradeHistory.pnl).label("total_pnl"),
                func.sum(sa_case((TradeHistory.pnl > 0, 1), else_=0)).label("wins")
            ).where(
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None)
            )
            return (await s.execute(stmt)).first()

    settings_rows, assets_rows, daily_rows, params_row, totals_row = await asyncio.gather(
        fetch_settings(),
        fetch_assets(),
        fetch_daily(),
        fetch_params(),
        fetch_all_time_totals()
    )

    initial_capital = settings.INITIAL_CAPITAL
    for row in settings_rows:
        if row.key == "INITIAL_CAPITAL":
            initial_capital = float(row.value)

    asset_stats = {asset: {"pnl": 0.0, "trades": 0, "wins": 0} for asset in settings.asset_list}
    for row in assets_rows:
        if row.asset in asset_stats:
            asset_stats[row.asset] = {
                "pnl": float(row.total_pnl or 0),
                "trades": int(row.total_trades or 0),
                "wins": int(row.wins or 0)
            }

    # Итоговые KPI карточки дашборда ВСЕГДА считаются за всё время
    all_total_pnl = float(totals_row.total_pnl or 0) if totals_row else 0.0
    all_wins = int(totals_row.wins or 0) if totals_row else 0
    all_trades_count = int(totals_row.total_trades or 0) if totals_row else 0
    all_losses = all_trades_count - all_wins
    all_capital = initial_capital + all_total_pnl
    all_winrate = (all_wins / all_trades_count) * 100 if all_trades_count > 0 else 0

    daily_pnl_map = {}
    for row in daily_rows:
        if row.day:
            day_str = str(row.day)
            daily_pnl_map[day_str] = {
                "pnl": float(row.daily_pnl or 0),
                "wins": int(row.wins or 0),
                "losses": int(row.losses or 0)
            }

    avg_win_price = float(params_row.avg_win_price or 0) if params_row else 0
    avg_loss_price = float(params_row.avg_loss_price or 0) if params_row else 0
    avg_win_prob = float(params_row.avg_win_prob or 0) if params_row else 0
    avg_loss_prob = float(params_row.avg_loss_prob or 0) if params_row else 0

    result = {
        "capital": round(all_capital, 2),
        "overall_pnl": round(all_total_pnl, 2),
        "daily_pnl": daily_pnl_map,
        "assets": asset_stats,
        "winrate": round(all_winrate, 1),
        "wins_vs_losses": {"wins": all_wins, "losses": all_losses},
        "parameters": {
            "avg_win_price": round(avg_win_price, 3),
            "avg_loss_price": round(avg_loss_price, 3),
            "avg_win_prob": round(avg_win_prob, 3),
            "avg_loss_prob": round(avg_loss_prob, 3)
        }
    }
    
    _stats_cache[cache_key] = {"time": current_time, "data": result}
    return result


from polyflip.db.models import DecisionFunnelLog


@router.get("/trading/funnel/stats")
@router.get("/funnel/stats")
async def get_funnel_stats(
    hours: int = 24,
    asset: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
):
    from sqlalchemy import and_

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    base_filter = [DecisionFunnelLog.created_at >= since]
    if asset:
        base_filter.append(DecisionFunnelLog.asset == asset.upper())

    gate_names = [
        "g1_model_loaded", "g2_price_fetched", "g3_dead_zone",
        "g4_no_flip", "g5_min_edge", "g6_price_range",
        "g7_crypto_confirm", "g8_combined_vote",
    ]

    # Одним SQL-запросом: total, traded, и COUNT blocked по каждому гейту
    gate_cols = [
        func.count(
            sa_case(
                (getattr(DecisionFunnelLog, g) == False, 1),  # noqa: E712
                else_=None
            )
        ).label(f"blocked_{g}")
        for g in gate_names
    ]
    q = select(
        func.count().label("total"),
        func.count(
            sa_case(
                (DecisionFunnelLog.final_action.in_(["BUY_YES", "BUY_NO"]), 1),
                else_=None
            )
        ).label("traded"),
        *gate_cols,
    ).where(and_(*base_filter))

    row = (await db.execute(q)).one()
    total = row.total
    if total == 0:
        return {"total": 0, "traded": 0, "hours": hours, "by_gate": {}, "by_asset": {}}

    by_gate = {
        g: {
            "blocked": getattr(row, f"blocked_{g}"),
            "pct": round(getattr(row, f"blocked_{g}") / total * 100, 1),
        }
        for g in gate_names
    }

    # by_asset — отдельный компактный GROUP BY запрос
    asset_q = select(
        DecisionFunnelLog.asset,
        func.count().label("total"),
        func.count(
            sa_case(
                (DecisionFunnelLog.final_action.in_(["BUY_YES", "BUY_NO"]), 1),
                else_=None
            )
        ).label("traded"),
    ).where(and_(*base_filter)).group_by(DecisionFunnelLog.asset)
    asset_rows = (await db.execute(asset_q)).all()
    by_asset = {r.asset: {"total": r.total, "traded": r.traded} for r in asset_rows}

    return {
        "total": total,
        "traded": row.traded,
        "hours": hours,
        "by_gate": by_gate,
        "by_asset": by_asset,
    }


@router.get("/trading/funnel/detail")
@router.get("/funnel/detail")
async def get_funnel_detail(
    hours: int = 6,
    asset: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session),
):
    """Детальный лог последних N записей для дебага конкретного рынка."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = (
        select(DecisionFunnelLog)
        .where(DecisionFunnelLog.created_at >= since)
        .order_by(DecisionFunnelLog.created_at.desc())
        .limit(limit)
    )
    if asset:
        q = q.where(DecisionFunnelLog.asset == asset.upper())
    rows = (await db.execute(q)).scalars().all()
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "asset": r.asset,
            "market_id": r.market_id,
            "trading_mode": r.trading_mode,
            "used_model": r.used_model,
            "p_flip": r.p_flip,
            "edge": r.edge,
            "fresh_price": r.fresh_price,
            "thresholds": {
                "lower": r.threshold_lower,
                "upper": r.threshold_upper,
                "min_edge": r.min_edge_used,
            },
            "gates": {
                "g1_model_loaded": r.g1_model_loaded,
                "g2_price_fetched": r.g2_price_fetched,
                "g3_dead_zone": r.g3_dead_zone,
                "g4_no_flip": r.g4_no_flip,
                "g5_min_edge": r.g5_min_edge,
                "g6_price_range": r.g6_price_range,
                "g7_crypto_confirm": r.g7_crypto_confirm,
                "g8_combined_vote": r.g8_combined_vote,
            },
            "final_action": r.final_action,
            "skip_reason": r.skip_reason,
        }
        for r in rows
    ]


@router.get("/pnl-markers")
async def get_pnl_markers(
    hours: int = Query(default=168, ge=1, le=720),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Возвращает маркеры событий (изменения параметров в strategy_config и рекорды ATH в config_presets)
    для наложения на график PnL.
    """
    from polyflip.db.models import StrategyConfig, ConfigPreset
    from collections import defaultdict
    from sqlalchemy import and_

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    cfg_rows = (await db.execute(
        select(StrategyConfig)
        .where(StrategyConfig.changed_at >= since)
        .order_by(StrategyConfig.changed_at.asc())
    )).scalars().all()

    grouped = defaultdict(list)
    for r in cfg_rows:
        ts_key = r.changed_at.strftime("%Y-%m-%dT%H:%M:00+00:00")
        grouped[ts_key].append({
            "key": r.key,
            "old_value": r.old_value,
            "new_value": r.new_value,
            "changed_by": r.changed_by,
        })

    markers = []
    for ts, changes in grouped.items():
        tooltip_lines = [f"{c['key']}: {c['old_value']} ➔ {c['new_value']}" for c in changes[:5]]
        if len(changes) > 5:
            tooltip_lines.append(f"... и ещё {len(changes)-5} параметров")

        markers.append({
            "timestamp": ts,
            "label": f"⚙️ {len(changes)} param(s)",
            "marker_type": "setting_change",
            "changes": changes,
            "tooltip": "\n".join(tooltip_lines),
        })

    ath_rows = (await db.execute(
        select(ConfigPreset)
        .where(
            and_(
                ConfigPreset.preset_type.in_(["ath_capital", "ath_pnl"]),
                ConfigPreset.created_at >= since,
                ConfigPreset.is_active == True,  # noqa: E712
            )
        )
        .order_by(ConfigPreset.created_at.asc())
    )).scalars().all()

    for a in ath_rows:
        markers.append({
            "timestamp": a.created_at.isoformat(),
            "label": f"🏆 {a.name}",
            "marker_type": "ath",
            "changes": [],
            "tooltip": f"Capital: ${a.capital_at_save:.2f} | PnL: ${a.pnl_at_save:.2f}" if a.capital_at_save else a.name,
        })

    markers.sort(key=lambda x: x["timestamp"])
    return {"count": len(markers), "markers": markers}

