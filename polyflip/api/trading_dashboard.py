import os
import time
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
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

@router.get("/api/trading/stats", dependencies=[Depends(verify_api_key)])
async def get_trading_stats(db: AsyncSession = Depends(get_db_session)):
    # Load initial capital setting
    stmt_settings = select(RuntimeSettings).where(RuntimeSettings.key == "INITIAL_CAPITAL")
    res_settings = await db.execute(stmt_settings)
    initial_capital_row = res_settings.scalar_one_or_none()
    initial_capital = float(initial_capital_row.value) if initial_capital_row else settings.INITIAL_CAPITAL

    # Load all successful trades
    stmt_trades = select(TradeHistory).where(TradeHistory.status == "SUCCESS").order_by(TradeHistory.created_at)
    res_trades = await db.execute(stmt_trades)
    trades = res_trades.scalars().all()

    if not trades:
        return {
            "capital": initial_capital,
            "overall_pnl": 0,
            "daily_pnl": {},
            "assets": {asset: {"pnl": 0.0, "trades": 0, "wins": 0} for asset in settings.asset_list},
            "winrate": 0,
            "wins_vs_losses": {"wins": 0, "losses": 0},
            "parameters": {
                "avg_win_price": 0,
                "avg_loss_price": 0,
                "avg_win_prob": 0,
                "avg_loss_prob": 0
            },
            "kelly_stats": {
                "avg_f": 0.0,
                "avg_mult": 1.0,
                "min_f": 0.0,
                "max_f": 0.0
            }
        }

    total_pnl = 0
    wins = 0
    losses = 0
    
    daily_pnl_map = {}
    asset_stats = {asset: {"pnl": 0.0, "trades": 0, "wins": 0} for asset in settings.asset_list}
    
    win_prices = []
    loss_prices = []
    win_probs = []
    loss_probs = []

    for t in trades:
        # R5 FIX: Используем pnl из TradeHistory напрямую. Сделки с pnl IS NULL считаем PENDING.
        if t.pnl is None:
            continue
            
        pnl = float(t.pnl)
        is_win = (pnl > 0)
        
        if is_win:
            wins += 1
            if t.executed_price:
                win_prices.append(float(t.executed_price))
            if t.predicted_flip_prob:
                win_probs.append(float(t.predicted_flip_prob))
        else:
            losses += 1
            if t.executed_price:
                loss_prices.append(float(t.executed_price))
            if t.predicted_flip_prob:
                loss_probs.append(float(t.predicted_flip_prob))

        total_pnl += pnl
        
        # Aggregate by day
        day_str = t.created_at.strftime("%Y-%m-%d")
        daily_pnl_map.setdefault(day_str, {"pnl": 0, "wins": 0, "losses": 0})
        daily_pnl_map[day_str]["pnl"] += pnl
        if is_win:
            daily_pnl_map[day_str]["wins"] += 1
        else:
            daily_pnl_map[day_str]["losses"] += 1
            
        # Aggregate by asset
        if t.asset not in asset_stats:
            asset_stats[t.asset] = {"pnl": 0, "trades": 0, "wins": 0}
            
        asset_stats[t.asset]["pnl"] += pnl
        asset_stats[t.asset]["trades"] += 1
        if is_win:
            asset_stats[t.asset]["wins"] += 1

    capital = initial_capital + total_pnl
    winrate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    
    avg_win_price = sum(win_prices) / len(win_prices) if win_prices else 0
    avg_loss_price = sum(loss_prices) / len(loss_prices) if loss_prices else 0
    avg_win_prob = sum(win_probs) / len(win_probs) if win_probs else 0
    avg_loss_prob = sum(loss_probs) / len(loss_probs) if loss_probs else 0

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

    return {
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
        "kelly_stats": kelly_stats
    }
