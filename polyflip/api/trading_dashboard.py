import os
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from polyflip.db.connection import get_db_session
from polyflip.db.models import TradeHistory, MarketSnapshot, RuntimeSettings
from polyflip.config import settings
from polyflip.api.auth import verify_api_key

router = APIRouter(tags=["TradingDashboard"])
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

@router.get("/trading")
async def get_trading_dashboard(request: Request):
    return templates.TemplateResponse("trading.html", {"request": request})

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
            "assets": {},
            "winrate": 0,
            "wins_vs_losses": {"wins": 0, "losses": 0},
            "parameters": {
                "avg_win_price": 0,
                "avg_loss_price": 0,
                "avg_win_prob": 0,
                "avg_loss_prob": 0
            }
        }

    # Load market outcomes for these trades (only one snapshot per market needed)
    market_ids = list(set([t.market_id for t in trades]))
    stmt_outcomes = select(MarketSnapshot.market_id, MarketSnapshot.final_outcome).where(
        and_(
            MarketSnapshot.market_id.in_(market_ids),
            MarketSnapshot.final_outcome != "PENDING"
        )
    )
    res_outcomes = await db.execute(stmt_outcomes)
    
    # Use a dictionary to store final outcomes for O(1) lookup
    market_outcomes = {}
    for r in res_outcomes.all():
        market_outcomes[r.market_id] = r.final_outcome

    total_pnl = 0
    wins = 0
    losses = 0
    
    daily_pnl_map = {}
    asset_stats = {}
    
    win_prices = []
    loss_prices = []
    win_probs = []
    loss_probs = []

    for t in trades:
        outcome = market_outcomes.get(t.market_id)
        if not outcome:
            continue # Market is still pending or not resolved
            
        is_win = (t.outcome_bought == outcome)
        pnl = 0
        if is_win:
            pnl = (t.amount_usdc / t.executed_price) - t.amount_usdc
            wins += 1
            win_prices.append(t.executed_price)
            win_probs.append(t.predicted_flip_prob)
        else:
            pnl = -t.amount_usdc
            losses += 1
            loss_prices.append(t.executed_price)
            loss_probs.append(t.predicted_flip_prob)

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
        }
    }
