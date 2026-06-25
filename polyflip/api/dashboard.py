import os
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.connection import get_db_session
from polyflip.db.models import CollectorStatus, LiveMarket, MarketSnapshot

router = APIRouter(tags=["Dashboard"])

# Получаем абсолютный путь до папки templates, так как uvicorn может запускаться из разных мест
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

@router.get("/dashboard")
async def get_dashboard(request: Request):
    """Отдает главную страницу дашборда"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/api/dashboard/status")
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
    
    live_data = []
    for lm in live_markets:
        live_data.append({
            "asset": lm.asset,
            "question": lm.question,
            "end_time_est": lm.end_time_est,
            "current_yes_price": lm.current_yes_price,
            "current_spread": round(lm.current_spread, 4),
            "volume_5min": round(lm.volume_5min, 2)
        })
        
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
        asset = row.asset
        outcome = row.final_outcome
        cnt = row.cnt
        if asset not in dataset_summary:
            dataset_summary[asset] = {"PENDING": 0, "RESOLVED": 0}
        
        if outcome == "PENDING":
            dataset_summary[asset]["PENDING"] += cnt
        else:
            dataset_summary[asset]["RESOLVED"] += cnt
            
    return {
        "collector": collector_data,
        "dataset_summary": dataset_summary,
        "live_markets": live_data
    }
