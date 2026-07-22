import os
import time
import asyncio
from fastapi.templating import Jinja2Templates
from typing import Optional
from datetime import datetime, time as dt_time, timezone, timedelta
from fastapi import APIRouter, Request, Depends, Query

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
                func.sum(case((TradeHistory.pnl > 0, 1), else_=0)).label("wins")
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
                func.sum(case((TradeHistory.pnl > 0, 1), else_=0)).label("wins"),
                func.sum(case((TradeHistory.pnl <= 0, 1), else_=0)).label("losses")
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
                func.avg(case((TradeHistory.pnl > 0, TradeHistory.executed_price), else_=None)).label("avg_win_price"),
                func.avg(case((TradeHistory.pnl <= 0, TradeHistory.executed_price), else_=None)).label("avg_loss_price"),
                func.avg(case((TradeHistory.pnl > 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_win_prob"),
                func.avg(case((TradeHistory.pnl <= 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_loss_prob")
            ).where(*conds)
            return (await s.execute(stmt)).first()

    settings_rows, assets_rows, daily_rows, params_row = await asyncio.gather(
        fetch_settings(),
        fetch_assets(),
        fetch_daily(),
        fetch_params()
    )

    initial_capital = settings.INITIAL_CAPITAL
    for row in settings_rows:
        if row.key == "INITIAL_CAPITAL":
            initial_capital = float(row.value)

    asset_stats = {asset: {"pnl": 0.0, "trades": 0, "wins": 0} for asset in settings.asset_list}
    total_pnl = 0.0
    wins = 0
    trades_count = 0
    for row in assets_rows:
        if row.asset in asset_stats:
            asset_stats[row.asset] = {
                "pnl": float(row.total_pnl or 0),
                "trades": int(row.total_trades or 0),
                "wins": int(row.wins or 0)
            }
        total_pnl += float(row.total_pnl or 0)
        wins += int(row.wins or 0)
        trades_count += int(row.total_trades or 0)

    losses = trades_count - wins
    capital = initial_capital + total_pnl
    winrate = (wins / trades_count) * 100 if trades_count > 0 else 0

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
        "capital": round(capital, 2),
        "overall_pnl": round(total_pnl, 2),
        "daily_pnl": daily_pnl_map,
        "assets": asset_stats,
        "winrate": round(winrate, 1),
        "wins_vs_losses": {"wins": wins, "losses": losses},
        "parameters": {
            "avg_win_price": round(avg_win_price, 3),
            "avg_loss_price": round(avg_loss_price, 3),
            "avg_win_prob": round(avg_win_prob, 3),
            "avg_loss_prob": round(avg_loss_prob, 3)
        }
    }
    
    _stats_cache[cache_key] = {"time": current_time, "data": result}
    return result
