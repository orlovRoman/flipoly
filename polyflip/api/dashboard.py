import os
import time
import asyncio
from datetime import datetime, timezone, timedelta
import httpx
import json
import structlog
import math
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from polyflip.db.connection import get_db_session
from polyflip.db.models import CollectorStatus, LiveMarket, MarketSnapshot, TradeHistory, ModelRegistry, RuntimeSettings
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

_dashboard_cache = {}
_DASHBOARD_CACHE_TTL = 30  # 30 секунд кэша

def invalidate_dashboard_cache():
    _dashboard_cache.clear()

@router.get("/api/dashboard/status", dependencies=[Depends(verify_api_key)])
async def get_dashboard_status(db: AsyncSession = Depends(get_db_session)):
    """Отдает данные для вкладки Статус Парсера"""
    current_time = time.time()
    if "status" in _dashboard_cache and current_time - _dashboard_cache["status"]["time"] < _DASHBOARD_CACHE_TTL:
        return _dashboard_cache["status"]["data"]

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
    
    dataset_summary = {asset: {"PENDING": 0, "RESOLVED": 0} for asset in settings.asset_list}
    for row in snap_res.all():
        if row.asset in dataset_summary:
            dataset_summary[row.asset]["PENDING" if row.final_outcome == "PENDING" else "RESOLVED"] += row.cnt
            
    # 4. Активные модели
    models_stmt = select(ModelRegistry).where(ModelRegistry.is_active)
    models_res = await db.execute(models_stmt)
    active_models = {}
    for m in models_res.scalars().all():
        active_models[m.asset] = m.version
        
    # 5. Скользящая точность за 7 дней (rolling accuracy) по активам
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    trades_stmt = select(TradeHistory.asset, TradeHistory.pnl).where(
        TradeHistory.created_at >= seven_days_ago,
        TradeHistory.status == "SUCCESS",
        TradeHistory.pnl.is_not(None)
    )
    trades_res = await db.execute(trades_stmt)
    trades_list = trades_res.all()
    
    asset_trades = {}
    for row in trades_list:
        asset_trades.setdefault(row.asset, []).append(row.pnl)
        
    rolling_accuracy = {}
    for asset, pnls in asset_trades.items():
        valid_pnls = [p for p in pnls if p != 0.0]
        if not valid_pnls:
            continue
        wins = sum(1 for p in valid_pnls if p > 0.0)
        acc = wins / len(valid_pnls)
        rolling_accuracy[asset] = {
            "accuracy": round(acc, 4),
            "total_trades": len(valid_pnls)
        }
            
    # 6. Список торгуемых активов
    trade_assets_stmt = select(RuntimeSettings.value).where(RuntimeSettings.key == "TRADE_ASSETS")
    trade_assets_res = await db.execute(trade_assets_stmt)
    trade_assets_val = trade_assets_res.scalar() or settings.TRADE_ASSETS
    trade_assets_list = [a.strip().upper() for a in trade_assets_val.split(",") if a.strip()]

    result = {
        "collector": collector_data,
        "dataset_summary": dataset_summary,
        "live_markets": live_data,
        "active_models": active_models,
        "rolling_accuracy": rolling_accuracy,
        "trade_assets": trade_assets_list
    }
    
    _dashboard_cache["status"] = {"time": current_time, "data": result}
    return result

@router.get("/api/dashboard/trade_logs", dependencies=[Depends(verify_api_key)])
async def get_trade_logs(
    db: AsyncSession = Depends(get_db_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100)
):
    """Возвращает последние логи торговли (успешные, фейлы и пропущенные) с пагинацией"""
    offset = (page - 1) * page_size

    # Общее кол-во записей (для кол-ва страниц)
    count_stmt = select(func.count(TradeHistory.id))
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = (
        select(TradeHistory, LiveMarket.question)
        .outerjoin(LiveMarket, TradeHistory.market_id == LiveMarket.market_id)
        .order_by(TradeHistory.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    logs_with_questions = result.all()
    
    items = [
        {
            "id": log.id,
            "market_id": log.market_id,
            "question": question or log.market_id,
            "asset": log.asset,
            "status": log.status,
            "outcome_bought": log.outcome_bought,
            "amount_usdc": log.amount_usdc,
            "executed_price": log.executed_price,
            "predicted_flip_prob": log.predicted_flip_prob,
            "model_version": getattr(log, 'model_version', None),
            "active_features": getattr(log, 'active_features', None),
            "error_msg": log.error_msg,
            "mode": getattr(log, 'mode', 'LIVE'),
            "pnl": getattr(log, 'pnl', None),
            "kelly_fraction": getattr(log, 'kelly_fraction', None),
            "kelly_multiplier": getattr(log, 'kelly_multiplier', None),
            "edge": getattr(log, 'edge', None),
            "created_at": log.created_at.isoformat(),
            "updated_at": log.updated_at.isoformat() if getattr(log, 'updated_at', None) else None
        }
        for log, question in logs_with_questions
    ]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": math.ceil(total / page_size) if page_size > 0 else 0,
        "items": items
    }

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
        return None

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_market(client, mid, info) for mid, info in unique_markets.items()]
        fetched_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in fetched_results:
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, Exception):
                logger.error("error_fetching_market_data", error=str(r))
                
    return {"status": "success", "results": results}
