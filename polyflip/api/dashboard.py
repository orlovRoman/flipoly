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

        status_query = select(CollectorStatus).join(
            subq,
            and_(
                CollectorStatus.service_name == subq.c.service_name,
                CollectorStatus.timestamp == subq.c.max_ts,
            ),
        )
        status_res = await db.execute(status_query)
        statuses = status_res.scalars().all()

        # 2. Активные рынки
        active_markets_query = select(LiveMarket).where(
            LiveMarket.status == "ACTIVE"
        )
        markets_res = await db.execute(active_markets_query)
        markets = markets_res.scalars().all()

        # 3. Количество снапшотов за период (для графика активности)
        # Группируем по часам
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

    models_stmt = select(ModelRegistry).order_by(
        ModelRegistry.asset, ModelRegistry.version
    )
    models_res = await db.execute(models_stmt)
    models_rows = models_res.scalars().all()

    trades_stmt = (
        select(TradeHistory)
        .where(
            TradeHistory.timestamp >= start_date,
            TradeHistory.pnl.isnot(None),
        )
        .order_by(TradeHistory.timestamp.asc())
    )
    trades_res = await db.execute(trades_stmt)
    trades_rows = trades_res.scalars().all()

    trades_by_model = {}
    exact_trades_count = {}
    reconstructed_trades_count = {}
    unattributed_trades = 0
    unattributed_pnl = 0.0

    for trade in trades_rows:
        try:
            pnl_val = float(trade.pnl)
        except (ValueError, TypeError):
            continue

        assigned = False

        if trade.model_registry_id is not None:
            model = next(
                (m for m in models_rows if m.id == trade.model_registry_id),
                None,
            )
            if model:
                key = (model.asset, model.version)
                trades_by_model.setdefault(key, []).append(pnl_val)
                exact_trades_count[key] = exact_trades_count.get(key, 0) + 1
                assigned = True

        if not assigned and trade.model_version is not None:
            key = (trade.asset, trade.model_version)
            trades_by_model.setdefault(key, []).append(pnl_val)
            exact_trades_count[key] = exact_trades_count.get(key, 0) + 1
            assigned = True

        if not assigned:
            unattributed_trades += 1
            unattributed_pnl += pnl_val

    result_map = {}
    for row in models_rows:
        asset = row.asset
        version = row.version
        key = f"{asset}_v{version}"

        valid_trades = trades_by_model.get((asset, version), [])
        total = len(valid_trades)
        total_pnl = sum(valid_trades) if total > 0 else 0.0
        wins = sum(1 for pnl in valid_trades if pnl > 0)

        result_map[key] = {
            "asset": asset,
            "version": version,
            "total_trades": total,
            "pnl": round(float(total_pnl), 2),
            "win_rate": round(wins / total * 100, 1) if total > 0 else None,
            "exact_trades": exact_trades_count.get((asset, version), 0),
            "reconstructed_trades": reconstructed_trades_count.get(
                (asset, version), 0
            ),
        }

    result_map["_unattributed"] = {
        "total_trades": unattributed_trades,
        "pnl": round(float(unattributed_pnl), 2),
    }

    response_data = {"status": "success", "data": result_map}
    _model_pnl_cache[cache_key] = {"time": current_time, "data": response_data}

    return response_data
