import os
import time
import asyncio
import httpx
import json
from typing import Dict, Any, List
import structlog
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.connection import get_db_session
from polyflip.db.models import CollectorStatus, LiveMarket, MarketSnapshot, TradeHistory, ModelRegistry
from polyflip.api.auth import verify_api_key
from polyflip.config import settings

logger = structlog.get_logger(__name__)
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

@router.post("/api/dashboard/verify_resolves", dependencies=[Depends(verify_api_key)])
async def verify_resolves(db: AsyncSession = Depends(get_db_session)):
    """Сверяет последние 50 разрешенных рынков из БД с актуальными данными Polymarket Gamma API"""
    
    # 1. Загружаем последние снепшоты с разрешенными исходами
    stmt = select(MarketSnapshot).where(MarketSnapshot.final_outcome != "PENDING")\
        .order_by(MarketSnapshot.recorded_at.desc())\
        .limit(200)
    
    res = await db.execute(stmt)
    snapshots = res.scalars().all()
    
    # Отбираем уникальные market_id (до 50 штук)
    unique_markets = {}
    for s in snapshots:
        if s.market_id not in unique_markets and len(unique_markets) < 50:
            unique_markets[s.market_id] = {
                "asset": s.asset,
                "db_outcome": s.final_outcome
            }
            
    if not unique_markets:
        return {"status": "success", "results": [], "message": "Нет разрешенных рынков в БД для проверки"}
        
    results = []
    semaphore = asyncio.Semaphore(10)  # не более 10 одновременных запросов
    
    async def fetch_market(client, market_id, info):
        async with semaphore:
            try:
                response = await client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
                
                if response.status_code == 200:
                    market_data = response.json()
                    question = market_data.get("question", "N/A")
                    closed = market_data.get("closed", False)
                    
                    # Ищем ответ по той же логике, что и в resolver.py
                    answer = (
                        market_data.get("answer")
                        or market_data.get("winnerOutcome")
                        or market_data.get("resolvedBy")
                    )
                    
                    if not answer and closed:
                        prices = market_data.get("outcomePrices", [])
                        outcomes = market_data.get("outcomes", ["Yes", "No"])
                        if isinstance(outcomes, str):
                            outcomes = json.loads(outcomes)
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        if prices and len(prices) >= 2 and outcomes and len(outcomes) >= 2:
                            try:
                                max_price = max(float(p) for p in prices)
                                if max_price >= 0.95:
                                    idx = [float(p) for p in prices].index(max_price)
                                    answer = outcomes[idx]
                            except Exception:
                                pass
                                
                    if answer:
                        outcome_map = {"UP": "YES", "DOWN": "NO", "YES": "YES", "NO": "NO"}
                        api_outcome = outcome_map.get(answer.upper(), answer.upper())
                        
                        db_outcome = info["db_outcome"]
                        status = "OK" if db_outcome == api_outcome else "MISMATCH"
                        
                        return {
                            "market_id": market_id,
                            "asset": info["asset"],
                            "question": question,
                            "db_outcome": db_outcome,
                            "api_outcome": api_outcome,
                            "status": status
                        }
                    else:
                        return {
                            "market_id": market_id,
                            "asset": info["asset"],
                            "question": question,
                            "db_outcome": info["db_outcome"],
                            "api_outcome": "PENDING/UNRESOLVED",
                            "status": "UNRESOLVED_ON_API"
                        }
                else:
                    return {
                        "market_id": market_id,
                        "asset": info["asset"],
                        "question": f"Error HTTP {response.status_code}",
                        "db_outcome": info["db_outcome"],
                        "api_outcome": "ERROR",
                        "status": f"HTTP_ERROR_{response.status_code}"
                    }
            except Exception as e:
                return {
                    "market_id": market_id,
                    "asset": info["asset"],
                    "question": f"Request failed: {str(e)}",
                    "db_outcome": info["db_outcome"],
                    "api_outcome": "ERROR",
                    "status": "CONNECTION_FAILED"
                }

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_market(client, mid, info) for mid, info in unique_markets.items()]
        fetched_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in fetched_results:
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, Exception):
                logger.error("error_fetching_market_data", error=str(r))
                
    return {"status": "success", "results": results}
