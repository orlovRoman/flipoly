import os
import time
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, Date
from datetime import datetime, time as dt_time, timezone

from polyflip.db.connection import get_db_session
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
            "assets": settings.asset_list
        }
    )

_stats_cache = {}
_STATS_CACHE_TTL = 30  # 30 секунд кэша

def invalidate_stats_cache():
    _stats_cache.clear()

@router.get("/api/trading/stats", dependencies=[Depends(verify_api_key)])
async def get_trading_stats(db: AsyncSession = Depends(get_db_session)):
    current_time = time.time()
    if "stats" in _stats_cache and current_time - _stats_cache["stats"]["time"] < _STATS_CACHE_TTL:
        return _stats_cache["stats"]["data"]

    # Load initial capital setting
    stmt_settings = select(RuntimeSettings).where(RuntimeSettings.key == "INITIAL_CAPITAL")
    res_settings = await db.execute(stmt_settings)
    initial_capital_row = res_settings.scalar_one_or_none()
    initial_capital = float(initial_capital_row.value) if initial_capital_row else settings.INITIAL_CAPITAL
 
    # Load KELLY_ENABLED setting
    stmt_kelly_enabled = select(RuntimeSettings).where(RuntimeSettings.key == "KELLY_ENABLED")
    res_kelly_enabled = await db.execute(stmt_kelly_enabled)
    kelly_enabled_row = res_kelly_enabled.scalar_one_or_none()
    kelly_enabled = (kelly_enabled_row.value.lower() == "true") if kelly_enabled_row else True

    # SQL Agreggations for Assets
    stmt_assets = select(
        TradeHistory.asset,
        func.count(TradeHistory.id).label("total_trades"),
        func.sum(TradeHistory.pnl).label("total_pnl"),
        func.sum(
            func.case(
                (TradeHistory.pnl > 0, 1),
                else_=0
            )
        ).label("wins")
    ).where(
        TradeHistory.status == "SUCCESS",
        TradeHistory.pnl.is_not(None)
    ).group_by(TradeHistory.asset)
    
    res_assets = await db.execute(stmt_assets)
    asset_stats = {asset: {"pnl": 0.0, "trades": 0, "wins": 0} for asset in settings.asset_list}
    
    total_pnl = 0.0
    wins = 0
    trades_count = 0
    for row in res_assets.all():
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

    # Daily PNL via SQL
    stmt_daily = select(
        cast(TradeHistory.created_at, Date).label("day"),
        func.sum(TradeHistory.pnl).label("daily_pnl"),
        func.sum(func.case((TradeHistory.pnl > 0, 1), else_=0)).label("wins"),
        func.sum(func.case((TradeHistory.pnl <= 0, 1), else_=0)).label("losses")
    ).where(
        TradeHistory.status == "SUCCESS",
        TradeHistory.pnl.is_not(None)
    ).group_by(cast(TradeHistory.created_at, Date))
    
    res_daily = await db.execute(stmt_daily)
    daily_pnl_map = {}
    for row in res_daily.all():
        if row.day:
            day_str = str(row.day)
            daily_pnl_map[day_str] = {
                "pnl": float(row.daily_pnl or 0),
                "wins": int(row.wins or 0),
                "losses": int(row.losses or 0)
            }

    # Parameters averages via SQL
    stmt_params = select(
        func.avg(func.case((TradeHistory.pnl > 0, TradeHistory.executed_price), else_=None)).label("avg_win_price"),
        func.avg(func.case((TradeHistory.pnl <= 0, TradeHistory.executed_price), else_=None)).label("avg_loss_price"),
        func.avg(func.case((TradeHistory.pnl > 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_win_prob"),
        func.avg(func.case((TradeHistory.pnl <= 0, TradeHistory.predicted_flip_prob), else_=None)).label("avg_loss_prob")
    ).where(
        TradeHistory.status == "SUCCESS",
        TradeHistory.pnl.is_not(None)
    )
    res_params = await db.execute(stmt_params)
    params_row = res_params.first()
    
    avg_win_price = float(params_row.avg_win_price or 0) if params_row else 0
    avg_loss_price = float(params_row.avg_loss_price or 0) if params_row else 0
    avg_win_prob = float(params_row.avg_win_prob or 0) if params_row else 0
    avg_loss_prob = float(params_row.avg_loss_prob or 0) if params_row else 0

    # Вычисляем Avg Kelly Today
    today_start = datetime.combine(datetime.now(timezone.utc).date(), dt_time.min).replace(tzinfo=timezone.utc)
    stmt_kelly = select(
        func.avg(TradeHistory.kelly_fraction).label("avg_f"),
        func.avg(TradeHistory.kelly_multiplier).label("avg_mult"),
        func.min(TradeHistory.kelly_fraction).label("min_f"),
        func.max(TradeHistory.kelly_fraction).label("max_f")
    ).where(
        TradeHistory.created_at >= today_start,
        TradeHistory.status.in_(["SUCCESS", "SKIPPED"]),
        TradeHistory.kelly_fraction.is_not(None)
    )
    res_kelly = await db.execute(stmt_kelly)
    kelly_row = res_kelly.first()
    
    kelly_stats = {
        "avg_f": round(float(kelly_row.avg_f), 4) if kelly_row and kelly_row.avg_f is not None else 0.0,
        "avg_mult": round(float(kelly_row.avg_mult), 2) if kelly_row and kelly_row.avg_mult is not None else 1.0,
        "min_f": round(float(kelly_row.min_f), 4) if kelly_row and kelly_row.min_f is not None else 0.0,
        "max_f": round(float(kelly_row.max_f), 4) if kelly_row and kelly_row.max_f is not None else 0.0,
    }

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
        },
        "kelly_stats": kelly_stats,
        "kelly_enabled": kelly_enabled
    }
    
    _stats_cache["stats"] = {"time": current_time, "data": result}
    return result
