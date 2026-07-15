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
from sqlalchemy import select, func, case, or_
from polyflip.db.connection import get_db_session, async_session
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

_logs_cache = {}
_LOGS_CACHE_TTL = 10  # 10 секунд кэша для логов торговли

@router.get("/api/dashboard/status", dependencies=[Depends(verify_api_key)])
async def get_dashboard_status(db: AsyncSession = Depends(get_db_session)):
    """Отдает данные для вкладки Статус Парсера"""
    current_time = time.time()
    if "status" in _dashboard_cache and current_time - _dashboard_cache["status"]["time"] < _DASHBOARD_CACHE_TTL:
        return _dashboard_cache["status"]["data"]

    async def fetch_collector():
        async with async_session() as s:
            stmt = select(CollectorStatus).order_by(CollectorStatus.run_at.desc()).limit(1)
            return (await s.execute(stmt)).scalar_one_or_none()

    async def fetch_live():
        async with async_session() as s:
            now = datetime.now(timezone.utc)
            stmt = select(LiveMarket).where(
                or_(
                    LiveMarket.end_time_est >= now,
                    LiveMarket.end_time_est.is_(None)
                )
            ).order_by(LiveMarket.asset, LiveMarket.end_time_est)
            return (await s.execute(stmt)).scalars().all()

    async def fetch_snaps():
        async with async_session() as s:
            stmt = select(
                MarketSnapshot.asset, 
                MarketSnapshot.final_outcome, 
                func.count(MarketSnapshot.id).label("cnt")
            ).group_by(MarketSnapshot.asset, MarketSnapshot.final_outcome)
            return (await s.execute(stmt)).all()

    async def fetch_models():
        async with async_session() as s:
            stmt = select(ModelRegistry).where(ModelRegistry.is_active)
            return (await s.execute(stmt)).scalars().all()

    async def fetch_rolling():
        async with async_session() as s:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            stmt = select(
                TradeHistory.asset,
                func.count(TradeHistory.id).label("total"),
                func.sum(case((TradeHistory.pnl > 0, 1), else_=0)).label("wins")
            ).where(
                TradeHistory.created_at >= seven_days_ago,
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None),
                TradeHistory.pnl != 0.0
            ).group_by(TradeHistory.asset)
            return (await s.execute(stmt)).all()

    async def fetch_settings_dict():
        async with async_session() as s:
            stmt = select(RuntimeSettings.key, RuntimeSettings.value)
            return {k: v for k, v in (await s.execute(stmt)).all()}

    collector_last, live_markets, snap_rows, models_rows, rolling_rows, trade_assets_val = await asyncio.gather(
        fetch_collector(),
        fetch_live(),
        fetch_snaps(),
        fetch_models(),
        fetch_rolling(),
        fetch_settings_dict()
    )

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
        
    dataset_summary = {asset: {"PENDING": 0, "RESOLVED": 0} for asset in settings.asset_list}
    for row in snap_rows:
        if row.asset in dataset_summary:
            dataset_summary[row.asset]["PENDING" if row.final_outcome == "PENDING" else "RESOLVED"] += row.cnt
            
    settings_dict = trade_assets_val
    trade_assets_str = settings_dict.get("TRADE_ASSETS", settings.TRADE_ASSETS)
    trade_assets_list = [a.strip().upper() for a in trade_assets_str.split(",") if a.strip()]

    global_mode = settings_dict.get("TRADING_MODE", "ml")
    
    active_models = {}
    for m in models_rows:
        # Определяем базовый ассет (BTC, ETH и т.д.)
        base_asset = m.asset.split("USDT")[0] if "USDT" in m.asset else m.asset.split("_")[0]
        if base_asset not in trade_assets_list:
            continue
            
        mode_str = settings_dict.get(f"TRADING_MODE_{base_asset}", global_mode)
        if not mode_str:
            mode_str = global_mode
        mode = mode_str.lower()
        
        if mode == "ml" and "USDT" not in m.asset:
            active_models[m.asset] = m.version
        elif mode == "crypto" and "USDT" in m.asset:
            active_models[m.asset] = m.version
        
    rolling_accuracy = {}
    for row in rolling_rows:
        total = int(row.total or 0)
        wins = int(row.wins or 0)
        if total > 0:
            rolling_accuracy[row.asset] = {
                "accuracy": round(wins / total, 4),
                "total_trades": total
            }
            


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
    current_time = time.time()
    cache_key = (page, page_size)
    if cache_key in _logs_cache and current_time - _logs_cache[cache_key]["time"] < _LOGS_CACHE_TTL:
        return _logs_cache[cache_key]["data"]

    offset = (page - 1) * page_size

    from sqlalchemy import text
    try:
        est_stmt = text("SELECT reltuples::bigint FROM pg_class WHERE relname = 'trade_history'")
        total = (await db.execute(est_stmt)).scalar()
        if total is None or total <= 0:
            count_stmt = select(func.count(TradeHistory.id))
            total = (await db.execute(count_stmt)).scalar_one()
    except Exception:
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
    
    from polyflip.api.settings import get_all_settings
    settings_dict = await get_all_settings(db=db)
    
    items = []
    for log, question in logs_with_questions:
        active_feat = getattr(log, 'active_features', None)
        if not active_feat and log.status == "SKIPPED":
            base_asset = log.asset.split("USDT")[0] if "USDT" in log.asset else log.asset.split("_")[0]
            mode = settings_dict.get(f"TRADING_MODE_{base_asset}", settings_dict.get("TRADING_MODE", "ml")).lower()
            if mode == "crypto":
                active_feat = "CRYPTO_TREND"
            elif mode == "favorite":
                active_feat = "PURE_FAVORITE"
            else:
                active_feat = "ml_strategy"
                
        items.append({
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
            "active_features": active_feat,
            "strategy_type": getattr(log, 'strategy_type', None),
            "error_msg": log.error_msg,
            "mode": getattr(log, 'mode', 'LIVE'),
            "pnl": getattr(log, 'pnl', None),
            "kelly_fraction": getattr(log, 'kelly_fraction', None),
            "kelly_multiplier": getattr(log, 'kelly_multiplier', None),
            "edge": getattr(log, 'edge', None),
            "stop_loss_status": getattr(log, 'stop_loss_status', None),
            "take_profit_status": getattr(log, 'take_profit_status', None),
            "take_profit_sell_price": getattr(log, 'take_profit_sell_price', None),
            "created_at": log.created_at.isoformat(),
            "updated_at": log.updated_at.isoformat() if getattr(log, 'updated_at', None) else None
        })

    out_data = {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": math.ceil(total / page_size) if page_size > 0 else 0,
        "items": items
    }
    _logs_cache[cache_key] = {"time": current_time, "data": out_data}
    return out_data

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

@router.get("/api/dashboard/daily_pnl", dependencies=[Depends(verify_api_key)])
async def get_daily_pnl(db: AsyncSession = Depends(get_db_session)):
    """Возвращает отчет PnL за текущие сутки (с полуночи UTC)"""
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    stmt = select(
        TradeHistory.asset,
        TradeHistory.active_features,
        TradeHistory.pnl,
        TradeHistory.amount_usdc
    ).where(
        TradeHistory.created_at >= midnight,
        TradeHistory.status.in_(["SUCCESS", "FAILED"]),
        TradeHistory.pnl.is_not(None)
    )
    
    result = await db.execute(stmt)
    trades = result.all()
    
    aggregated = {}
    for row in trades:
        asset = row.asset.split('_')[0].split('USDT')[0].upper()
        features = (row.active_features or "").lower()
        
        if 'аутсайдер' in features or 'outsider' in features:
            strategy = 'Аутсайдер'
        elif 'фаворит' in features or 'favorite' in features:
            strategy = 'Фаворит'
        elif 'crypto' in features or 'крипто' in features:
            strategy = 'Крипто'
        elif trade.executed_price is not None:
            if float(trade.executed_price) >= 0.5:
                strategy = 'Фаворит'
            else:
                strategy = 'Аутсайдер'
        else:
            strategy = 'Другое'
            
        key = f"{asset}_{strategy}"
        if key not in aggregated:
            aggregated[key] = {
                "asset": asset,
                "strategy": strategy,
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "volume": 0.0
            }
            
        aggregated[key]["trades"] += 1
        if row.pnl and row.pnl > 0:
            aggregated[key]["wins"] += 1
        aggregated[key]["pnl"] += (row.pnl or 0.0)
        aggregated[key]["volume"] += (row.amount_usdc or 0.0)
        
    response_data = []
    for data in aggregated.values():
        wr = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
        response_data.append({
            "asset": data["asset"],
            "strategy": data["strategy"],
            "trades": data["trades"],
            "win_rate": round(wr, 1),
            "pnl": round(data["pnl"], 2),
            "volume": round(data["volume"], 2)
        })
        
    def sort_key(x):
        s_order = 0 if x["strategy"] == "Аутсайдер" else (1 if x["strategy"] == "Фаворит" else 2)
        return (s_order, x["asset"])
        
    response_data.sort(key=sort_key)
    
    return {"status": "success", "data": response_data}
