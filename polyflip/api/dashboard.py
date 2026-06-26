import os
import time
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.connection import get_db_session
from polyflip.db.models import CollectorStatus, LiveMarket, MarketSnapshot, TradeHistory, ModelRegistry
from polyflip.api.auth import verify_api_key
from polyflip.config import settings

router = APIRouter(tags=["Dashboard"])

# Получаем абсолютный путь до папки templates, так как uvicorn может запускаться из разных мест
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

@router.get("/dashboard")
async def get_dashboard(request: Request):
    """Отдает главную страницу дашборда"""
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "timestamp": int(time.time()),
            "assets": settings.asset_list
        }
    )

@router.get("/api/dashboard/status", dependencies=[Depends(verify_api_key)])
async def get_dashboard_status(db: AsyncSession = Depends(get_db_session)):
    """Отдает данные для вкладки Статус Парсера"""
    # 1. Последний статус коллектора
    collector_stmt = select(CollectorStatus).order_by(CollectorStatus.run_at.desc()).limit(1)
    collector_res = await db.execute(collector_stmt)
    collector_last = collector_res.scalar_one_or_none()
    
    collector_data = None
    if collector_last:
        collector_data = {
            "run_at": collector_last.run_at,
            "status": collector_last.status,
            "duration_sec": round(collector_last.duration_sec, 2),
            "markets_found": collector_last.markets_found,
            "markets_saved": collector_last.markets_saved,
            "error_message": collector_last.error_message
        }
        
    # 2. Живые рынки
    live_stmt = select(LiveMarket).order_by(LiveMarket.asset, LiveMarket.end_time_est)
    live_res = await db.execute(live_stmt)
    live_markets = live_res.scalars().all()
    
    live_data = [
        {
            "asset": lm.asset,
            "question": lm.question,
            "end_time_est": lm.end_time_est,
            "current_yes_price": lm.current_yes_price,
            "current_spread": round(lm.current_spread, 4),
            "volume_5min": round(lm.volume_5min, 2)
        }
        for lm in live_markets
    ]
        
    # 3. Сводка по снепшотам
    # SELECT asset, final_outcome, count(*) FROM market_snapshots GROUP BY asset, final_outcome
    snap_stmt = select(
        MarketSnapshot.asset, 
        MarketSnapshot.final_outcome, 
        func.count(MarketSnapshot.id).label("cnt")
    ).group_by(MarketSnapshot.asset, MarketSnapshot.final_outcome)
    
    snap_res = await db.execute(snap_stmt)
    
    dataset_summary = {}
    for row in snap_res.all():
        ds = dataset_summary.setdefault(row.asset, {"PENDING": 0, "RESOLVED": 0})
        ds["PENDING" if row.final_outcome == "PENDING" else "RESOLVED"] += row.cnt
            
    # 4. Активные модели
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active == True)
    models_res = await db.execute(models_stmt)
    active_models = {}
    for m in models_res.scalars().all():
        active_models[m.asset] = m.version
            
    return {
        "collector": collector_data,
        "dataset_summary": dataset_summary,
        "live_markets": live_data,
        "active_models": active_models
    }

@router.get("/api/dashboard/trade_logs", dependencies=[Depends(verify_api_key)])
async def get_trade_logs(db: AsyncSession = Depends(get_db_session)):
    """Возвращает последние логи торговли (успешные, фейлы и пропущенные)"""
    stmt = select(TradeHistory).order_by(TradeHistory.created_at.desc()).limit(50)
    result = await db.execute(stmt)
    logs = result.scalars().all()
    
    return [
        {
            "id": log.id,
            "market_id": log.market_id,
            "asset": log.asset,
            "status": log.status,
            "outcome_bought": log.outcome_bought,
            "amount_usdc": log.amount_usdc,
            "executed_price": log.executed_price,
            "predicted_flip_prob": log.predicted_flip_prob,
            "model_version": getattr(log, 'model_version', None),
            "error_msg": log.error_msg,
            "mode": getattr(log, 'mode', 'LIVE'),
            "pnl": getattr(log, 'pnl', None),
            "created_at": log.created_at.isoformat()
        }
        for log in logs
    ]
