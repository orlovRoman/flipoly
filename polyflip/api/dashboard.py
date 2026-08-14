import os

STATIC_VERSION = os.getenv("POLYFLIP_BUILD_SHA", "dev")
import time
import asyncio
from datetime import datetime, timezone, timedelta
import httpx
import json
import structlog
import math
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, or_, and_, cast, Numeric
from polyflip.db.connection import get_db_session, async_session
from polyflip.db.models import (
    CollectorStatus,
    LiveMarket,
    MarketSnapshot,
    TradeHistory,
    ModelRegistry,
    RuntimeSettings,
    DecisionFunnelLog,
)
from polyflip.ui_helpers import direction_display_value
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
        request=request,
        name="index.html",
        context={
            "request": request,
            "timestamp": int(time.time()),
            "static_version": STATIC_VERSION,
            "assets": settings.asset_list,
            "root_path": request.scope.get("root_path", ""),
        },
    )


@router.get("/optimizer")
async def get_optimizer_dashboard(request: Request):
    """Отдает страницу дашборда автономного оптимизатора AI Lab"""
    return templates.TemplateResponse(
        request=request,
        name="optimizer.html",
        context={
            "request": request,
            "timestamp": int(time.time()),
            "static_version": STATIC_VERSION,
            "assets": settings.asset_list,
            "root_path": request.scope.get("root_path", ""),
        },
    )


@router.get("/execution")
async def get_execution_dashboard(request: Request):
    """Отдает страницу дашборда Исполнения"""
    return templates.TemplateResponse(
        request=request,
        name="execution.html",
        context={
            "request": request,
            "timestamp": int(time.time()),
            "static_version": STATIC_VERSION,
            "root_path": request.scope.get("root_path", ""),
        },
    )


_dashboard_cache = {}


def invalidate_dashboard_cache():
    _dashboard_cache.clear()
    _model_pnl_cache.clear()

_DASHBOARD_CACHE_TTL = 30  # 30 секунд кэша


@router.get("/api/dashboard_data", dependencies=[Depends(verify_api_key)])
async def get_dashboard_data(
    hours: int = 24,
    cache_bust: int = 0,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Возвращает агрегированные данные для графиков и таблиц
    Кэширует результат в памяти на _DASHBOARD_CACHE_TTL секунд,
    если передан cache_bust (например, текущий timestamp при ручном рефреше) - кэш сбрасывается.
    """
    now = time.time()
    if (
        cache_bust == 0
        and hours in _dashboard_cache
        and now - _dashboard_cache[hours]["timestamp"] < _DASHBOARD_CACHE_TTL
    ):
        return _dashboard_cache[hours]["data"]

    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    try:
        # 1. Последние статусы парсеров (берем последнюю запись для каждого сервиса)
        subq = (
            select(
                CollectorStatus.service_name,
                func.max(CollectorStatus.timestamp).label("max_ts"),
            )
            .group_by(CollectorStatus.service_name)
            .subquery()
        )

        statuses_query = (
            select(CollectorStatus)
            .join(
                subq,
                and_(
                    CollectorStatus.service_name == subq.c.service_name,
                    CollectorStatus.timestamp == subq.c.max_ts,
                ),
            )
            .order_by(CollectorStatus.service_name)
        )
        statuses_res = await db.execute(statuses_query)
        statuses = statuses_res.scalars().all()

        # 2. Активные рынки (live_markets)
        markets_query = select(LiveMarket).order_by(LiveMarket.asset)
        markets_res = await db.execute(markets_query)
        markets = markets_res.scalars().all()

        # 3. Активность сбора данных по часам за указанный период (hours)
        snapshot_counts_query = (
            select(
                MarketSnapshot.asset,
                func.strftime(
                    "%Y-%m-%d %H:00:00", MarketSnapshot.market_timestamp
                ).label("hour"),
                func.count(MarketSnapshot.id).label("count"),
            )
            .where(MarketSnapshot.market_timestamp >= start_time)
            .group_by(
                MarketSnapshot.asset,
                func.strftime(
                    "%Y-%m-%d %H:00:00", MarketSnapshot.market_timestamp
                ),
            )
            .order_by("hour")
        )
        counts_res = await db.execute(snapshot_counts_query)
        counts_data = counts_res.all()

        # Форматируем данные для отдачи
        services_data = [
            {
                "service": s.service_name,
                "status": s.status,
                "latency_ms": s.latency_ms,
                "last_seen": s.timestamp.isoformat() if s.timestamp else None,
                "details": s.details,
            }
            for s in statuses
        ]

        markets_data = [
            {
                "asset": m.asset,
                "slug": m.slug,
                "condition_id": m.condition_id,
                "clob_token_id": m.clob_token_id,
                "end_date": m.end_date.isoformat() if m.end_date else None,
                "price_up": m.current_price_up,
                "price_down": m.current_price_down,
            }
            for m in markets
        ]

        activity_data = {}
        for row in counts_data:
            asset, hour, count = row
            if asset not in activity_data:
                activity_data[asset] = []
            activity_data[asset].append({"hour": hour, "count": count})

        # 4. Получаем данные об активных моделях для вкладки "Модели"
        models_query = select(ModelRegistry).where(
            ModelRegistry.is_active == True
        )
        models_res = await db.execute(models_query)
        models = models_res.scalars().all()

        models_data = [
            {
                "id": m.id,
                "asset": m.asset,
                "version": m.version,
                "model_type": m.model_type,
                "features": m.features,
                "decision_threshold": m.decision_threshold,
                "decision_threshold_down": m.decision_threshold_down,
                "accuracy": m.accuracy,
                "backtest_pnl": m.backtest_pnl,
                "backtest_trades": m.backtest_trades,
                "created_at": m.created_at.isoformat()
                if m.created_at
                else None,
            }
            for m in models
        ]

        result = {
            "status": "success",
            "data": {
                "services": services_data,
                "markets": markets_data,
                "activity": activity_data,
                "models": models_data,
            },
        }

        # Сохраняем в кэш
        _dashboard_cache[hours] = {"timestamp": now, "data": result}

        return result
    except Exception as e:
        logger.exception("Error generating dashboard data", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate dashboard data",
        ) from e


_model_pnl_cache = {}
_MODEL_PNL_CACHE_TTL = 30  # 30 секунд кэша


@router.get("/api/dashboard/models_pnl", dependencies=[Depends(verify_api_key)])
async def get_models_pnl(
    days: int = Query(30, ge=1, le=90),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Возвращает PnL, винрейт и количество сделок по каждой версии модели для каждого актива.
    Покрывает и закрытые по экспирации сделки, и закрытые по стоп-лоссу/тейк-профиту.
    """
    current_time = time.time()
    cache_key = f"models_pnl_{days}"
    if (
        cache_key in _model_pnl_cache
        and current_time - _model_pnl_cache[cache_key]["time"]
        < _MODEL_PNL_CACHE_TTL
    ):
        return _model_pnl_cache[cache_key]["data"]

    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    # 1. Получаем все модели
    models_res = await db.execute(
        select(ModelRegistry).order_by(
            ModelRegistry.asset, ModelRegistry.version
        )
    )
    all_models = models_res.scalars().all()

    # 2. Получаем закрытые сделки за период (как по экспирации, так и по SL/TP)
    trades_query = select(TradeHistory).where(
        TradeHistory.position_status == "CLOSED",
        TradeHistory.pnl.is_not(None),
        TradeHistory.created_at >= start_date,
    )
    trades_res = await db.execute(trades_query)
    trades_rows = trades_res.scalars().all()

    # 3. Инициализируем структуру статистики для каждой модели
    # Ключ: (asset, version)
    stats = {}
    for m in all_models:
        key = (m.asset, m.version)
        stats[key] = {
            "model_id": m.id,
            "asset": m.asset,
            "version": m.version,
            "model_type": m.model_type,
            "features": m.features,
            "decision_threshold": m.decision_threshold,
            "decision_threshold_down": m.decision_threshold_down,
            "is_active": m.is_active,
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "accuracy": m.accuracy,
            "backtest_pnl": m.backtest_pnl,
            "backtest_trades": m.backtest_trades,
        }

    # 4. Агрегируем сделки по моделям
    for trade in trades_rows:
        try:
            pnl_val = float(trade.pnl)
        except (ValueError, TypeError):
            continue

        assigned = False
        # Сначала проверяем точное совпадение по direction_model_version
        if trade.direction_model_version is not None:
            key = (trade.asset, trade.direction_model_version)
            if key in stats:
                stats[key]["total_trades"] += 1
                stats[key]["total_pnl"] += pnl_val
                if pnl_val > 0:
                    stats[key]["winning_trades"] += 1
                assigned = True

        # Если не назначено, но есть trade.model_type — пробуем сопоставить по типу
        if not assigned and trade.model_type:
            for key, s in stats.items():
                if (
                    s["asset"] == trade.asset
                    and s["model_type"] == trade.model_type
                ):
                    s["total_trades"] += 1
                    s["total_pnl"] += pnl_val
                    if pnl_val > 0:
                        s["winning_trades"] += 1
                    break

    # 5. Рассчитываем винрейт и форматируем результат
    result_data = []
    for (asset, version), s in stats.items():
        total = s["total_trades"]
        wins = s["winning_trades"]
        win_rate = (wins / total * 100.0) if total > 0 else 0.0
        result_data.append(
            {
                "model_id": s["model_id"],
                "asset": s["asset"],
                "version": s["version"],
                "model_type": s["model_type"],
                "features": s["features"],
                "decision_threshold": s["decision_threshold"],
                "decision_threshold_down": s["decision_threshold_down"],
                "is_active": s["is_active"],
                "total_trades": total,
                "winning_trades": wins,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(s["total_pnl"], 4),
                "accuracy": s["accuracy"],
                "backtest_pnl": s["backtest_pnl"],
                "backtest_trades": s["backtest_trades"],
            }
        )

    # Сортируем: сначала активные, затем по активу и версии
    result_data.sort(
        key=lambda x: (not x["is_active"], x["asset"], x["version"])
    )

    response = {"status": "success", "data": result_data}
    _model_pnl_cache[cache_key] = {"time": current_time, "data": response}
    return response
