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
from polyflip.db.execution_models import ExecutionRequest
from polyflip.ui_helpers import direction_display_value
from polyflip.api.auth import verify_api_key
from polyflip.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Dashboard"])

# Получаем абсолютный путь до папки templates, так как uvicorn может запускаться из разных мест
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))


def _mrf_audit_payload(funnel) -> dict | None:
    """Return a JSON-safe MRF payload for a trade-log row.

    The detailed audit is stored in ``mrf_audit_json``. Older rows and rows
    where the classifier was not ready only have scalar telemetry columns;
    expose those as a compatibility payload instead of dropping MRF from the
    dashboard response.
    """
    if funnel is None:
        return None

    raw = getattr(funnel, "mrf_audit_json", None)
    payload: dict = {}
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                payload = parsed
        except (TypeError, ValueError):
            logger.warning("invalid_mrf_audit_json", funnel_id=getattr(funnel, "id", None))

    telemetry_names = (
        "mrf_mode",
        "mrf_evaluated",
        "mrf_phase",
        "mrf_asset_phase",
        "mrf_strength",
        "mrf_confidence",
        "mrf_multiplier",
        "mrf_applied",
        "mrf_failure_reason",
        "mrf_final_action",
    )
    has_telemetry = any(
        getattr(funnel, name, None) is not None for name in telemetry_names
    )
    if not payload and not has_telemetry:
        return None

    payload.setdefault("mode", getattr(funnel, "mrf_mode", None))
    payload.setdefault("evaluated", getattr(funnel, "mrf_evaluated", None))
    payload.setdefault("global_phase", getattr(funnel, "mrf_phase", None))
    payload.setdefault("asset_phase", getattr(funnel, "mrf_asset_phase", None))
    payload.setdefault("global_strength", getattr(funnel, "mrf_strength", None))
    payload.setdefault("global_confidence", getattr(funnel, "mrf_confidence", None))
    payload.setdefault("applied", getattr(funnel, "mrf_applied", None))
    payload.setdefault("failure_reason", getattr(funnel, "mrf_failure_reason", None))

    policy = payload.get("policy")
    if not isinstance(policy, dict):
        policy = {}
        payload["policy"] = policy
    policy.setdefault("multiplier", getattr(funnel, "mrf_multiplier", None))
    if "allow" not in policy:
        final_action = getattr(funnel, "mrf_final_action", None)
        if final_action is not None:
            policy["allow"] = final_action != "SKIP"

    return payload


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

_logs_cache = {}
_LOGS_CACHE_TTL = 10  # 10 секунд кэша для логов торговли
_logs_cache_lock = asyncio.Lock()


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
        # 1. Последние статусы парсеров (по каждому сервису отдельно)
        #    Используем ROW_NUMBER() over (PARTITION BY service_name ORDER BY run_at/timestamp DESC)
        service_expr = func.coalesce(CollectorStatus.service_name, "polymarket_collector")
        sort_time_expr = func.coalesce(CollectorStatus.run_at, CollectorStatus.timestamp)
        rn_expr = (
            func.row_number()
            .over(
                partition_by=service_expr,
                order_by=sort_time_expr.desc(),
            )
            .label("rn")
        )

        ranked_subq = (
            select(
                CollectorStatus.id,
                CollectorStatus.status,
                CollectorStatus.latency_ms,
                CollectorStatus.last_event_timestamp,
                CollectorStatus.error_message,
                CollectorStatus.details,
                CollectorStatus.timestamp,
                CollectorStatus.run_at,
                CollectorStatus.markets_found,
                CollectorStatus.markets_saved,
                CollectorStatus.duration_sec,
                service_expr.label("effective_service"),
                rn_expr,
            )
            .subquery()
        )

        statuses_query = (
            select(ranked_subq)
            .where(ranked_subq.c.rn == 1)
            .order_by(ranked_subq.c.effective_service)
        )
        statuses_res = await db.execute(statuses_query)
        statuses_rows = statuses_res.all()

        # 2. Активные рынки (live_markets) — используем реальные заполненные поля
        markets_query = (
            select(LiveMarket)
            .where(LiveMarket.status == "ACTIVE")
            .order_by(LiveMarket.id.desc())
            .limit(20)
        )
        markets_res = await db.execute(markets_query)
        markets = markets_res.scalars().all()

        # 3. Активность сбора данных по часам — используем recorded_at (market_timestamp не заполняется)
        hour_trunc = func.date_trunc("hour", MarketSnapshot.recorded_at)
        snapshot_counts_query = (
            select(
                MarketSnapshot.asset,
                func.to_char(hour_trunc, "YYYY-MM-DD HH24:00:00").label("hour"),
                func.count(MarketSnapshot.id).label("count"),
            )
            .where(MarketSnapshot.recorded_at >= start_time)
            .group_by(
                MarketSnapshot.asset,
                hour_trunc,
            )
            .order_by(hour_trunc)
        )
        counts_res = await db.execute(snapshot_counts_query)
        counts_data = counts_res.all()

        # Форматируем данные для отдачи
        services_data = [
            {
                "service": row.effective_service,
                "status": (row.status or "UNKNOWN").upper(),
                "latency_ms": row.latency_ms,
                "last_seen": row.run_at.isoformat() if row.run_at else (row.timestamp.isoformat() if row.timestamp else None),
                "details": {
                    "markets_found": row.markets_found,
                    "markets_saved": row.markets_saved,
                    "duration_sec": round(row.duration_sec, 2) if row.duration_sec is not None else None,
                    "error": row.error_message,
                },
            }
            for row in statuses_rows
        ]

        markets_data = [
            {
                "asset": m.asset,
                "slug": m.question or m.market_id,   # question заполнен, slug пуст
                "condition_id": m.condition_id or m.market_id,
                "clob_token_id": m.yes_token_id,
                "end_date": m.end_time_est.isoformat() if m.end_time_est else None,
                "price_up": m.current_yes_price,
                "price_down": m.current_no_price,
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
        models_query = select(
            ModelRegistry.id,
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.model_type,
            ModelRegistry.features,
            ModelRegistry.decision_threshold,
            ModelRegistry.decision_threshold_down,
            ModelRegistry.accuracy,
            ModelRegistry.backtest_pnl,
            ModelRegistry.backtest_trades,
            ModelRegistry.trained_at,
        ).where(ModelRegistry.is_active == True)
        models_res = await db.execute(models_query)
        models_rows = models_res.all()

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
                "created_at": m.trained_at.isoformat()
                if m.trained_at
                else None,
            }
            for m in models_rows
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



@router.get("/api/dashboard/status", dependencies=[Depends(verify_api_key)])
async def get_dashboard_status(db: AsyncSession = Depends(get_db_session)):
    """Отдает данные для вкладки Статус Парсера"""
    current_time = time.time()
    if (
        "status" in _dashboard_cache
        and current_time - _dashboard_cache["status"]["time"] < _DASHBOARD_CACHE_TTL
    ):
        return _dashboard_cache["status"]["data"]

    async def fetch_collector():
        async with async_session() as s:
            stmt = (
                select(CollectorStatus).order_by(CollectorStatus.run_at.desc()).limit(1)
            )
            return (await s.execute(stmt)).scalar_one_or_none()

    async def fetch_live():
        async with async_session() as s:
            now = datetime.now(timezone.utc)
            stmt = (
                select(LiveMarket)
                .where(
                    or_(
                        LiveMarket.end_time_est >= now,
                        LiveMarket.end_time_est.is_(None),
                    )
                )
                .order_by(LiveMarket.asset, LiveMarket.end_time_est)
            )
            return (await s.execute(stmt)).scalars().all()

    async def fetch_snaps():
        async with async_session() as s:
            stmt = select(
                MarketSnapshot.asset,
                MarketSnapshot.final_outcome,
                func.count(MarketSnapshot.id).label("cnt"),
            ).group_by(MarketSnapshot.asset, MarketSnapshot.final_outcome)
            return (await s.execute(stmt)).all()

    async def fetch_models():
        async with async_session() as s:
            stmt = select(ModelRegistry).where(ModelRegistry.is_active)
            return (await s.execute(stmt)).scalars().all()

    async def fetch_rolling():
        async with async_session() as s:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            stmt = (
                select(
                    TradeHistory.asset,
                    TradeHistory.mode,
                    func.count(TradeHistory.id).label("total"),
                    func.sum(case((TradeHistory.pnl > 0, 1), else_=0)).label("wins"),
                )
                .where(
                    TradeHistory.created_at >= seven_days_ago,
                    TradeHistory.position_status == "CLOSED",
                    TradeHistory.pnl.is_not(None),
                )
                .group_by(TradeHistory.asset, TradeHistory.mode)
            )
            return (await s.execute(stmt)).all()

    async def fetch_settings_dict():
        async with async_session() as s:
            stmt = select(RuntimeSettings.key, RuntimeSettings.value)
            return {k: v for k, v in (await s.execute(stmt)).all()}

    (
        collector_last,
        live_markets,
        snap_rows,
        models_rows,
        rolling_rows,
        trade_assets_val,
    ) = await asyncio.gather(
        fetch_collector(),
        fetch_live(),
        fetch_snaps(),
        fetch_models(),
        fetch_rolling(),
        fetch_settings_dict(),
    )

    collector_data = None
    if collector_last:
        collector_data = {
            "run_at": collector_last.run_at,
            "status": collector_last.status,
            "duration_sec": round(collector_last.duration_sec, 2),
            "markets_found": collector_last.markets_found,
            "markets_saved": collector_last.markets_saved,
            "error_message": collector_last.error_message,
        }

    live_data = [
        {
            "asset": lm.asset,
            "question": lm.question,
            "end_time_est": lm.end_time_est,
            "current_yes_price": lm.current_yes_price,
            "current_spread": round(lm.current_spread, 4),
            "volume_5min": round(lm.volume_5min, 2),
        }
        for lm in live_markets
    ]

    dataset_summary = {
        asset: {"PENDING": 0, "RESOLVED": 0} for asset in settings.asset_list
    }
    for row in snap_rows:
        if row.asset in dataset_summary:
            dataset_summary[row.asset][
                "PENDING" if row.final_outcome == "PENDING" else "RESOLVED"
            ] += row.cnt

    settings_dict = trade_assets_val
    trade_assets_str = settings_dict.get("TRADE_ASSETS", settings.TRADE_ASSETS)
    trade_assets_list = [
        a.strip().upper() for a in trade_assets_str.split(",") if a.strip()
    ]

    global_mode = settings_dict.get("TRADING_MODE", "ml")

    active_models = {}
    for m in models_rows:
        # Определяем базовый ассет (BTC, ETH и т.д.)
        base_asset = (
            m.asset.split("USDT")[0] if "USDT" in m.asset else m.asset.split("_")[0]
        )
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
            key = f"{row.asset}_{row.mode}"
            rolling_accuracy[key] = {
                "accuracy": round(wins / total, 4),
                "total_trades": total,
                "mode": row.mode,
            }

    result = {
        "collector": collector_data,
        "dataset_summary": dataset_summary,
        "live_markets": live_data,
        "active_models": active_models,
        "rolling_accuracy": rolling_accuracy,
        "trade_assets": trade_assets_list,
    }

    _dashboard_cache["status"] = {"time": current_time, "data": result}
    return result


@router.get("/api/dashboard/trade_logs", dependencies=[Depends(verify_api_key)])
async def get_trade_logs(
    db: AsyncSession = Depends(get_db_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
):
    """Возвращает последние логи торговли (успешные, фейлы и пропущенные) с пагинацией"""
    current_time = time.time()
    cache_key = (page, page_size)
    if (
        cache_key in _logs_cache
        and current_time - _logs_cache[cache_key]["time"] < _LOGS_CACHE_TTL
    ):
        return _logs_cache[cache_key]["data"]

    offset = (page - 1) * page_size

    from sqlalchemy import text

    try:
        est_stmt = text(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = 'trade_history'"
        )
        total = (await db.execute(est_stmt)).scalar()
        if total is None or total <= 0:
            count_stmt = select(func.count(TradeHistory.id))
            total = (await db.execute(count_stmt)).scalar_one()
    except Exception:
        count_stmt = select(func.count(TradeHistory.id))
        total = (await db.execute(count_stmt)).scalar_one()

    stmt = (
        select(TradeHistory, LiveMarket.question, LiveMarket.end_time_est)
        .outerjoin(LiveMarket, TradeHistory.market_id == LiveMarket.market_id)
        .order_by(TradeHistory.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    logs_with_questions = result.all()

    decision_run_ids = [
        log.decision_run_id for log, _, _ in logs_with_questions if log.decision_run_id
    ]
    funnel_map = {}
    if decision_run_ids:
        funnel_res = await db.execute(
            select(DecisionFunnelLog)
            .where(DecisionFunnelLog.decision_run_id.in_(decision_run_ids))
            .order_by(DecisionFunnelLog.created_at.asc())
        )
        # Latest decision_run_id log takes precedence
        funnel_map = {f.decision_run_id: f for f in funnel_res.scalars().all()}

    execution_failure_map = {}
    trade_history_ids = [log.id for log, _, _ in logs_with_questions]
    if trade_history_ids:
        execution_res = await db.execute(
            select(ExecutionRequest)
            .where(ExecutionRequest.trade_history_id.in_(trade_history_ids))
            .where(
                ExecutionRequest.state.in_(
                    {"REJECTED", "EXPIRED", "CANCELED", "MANUAL_REVIEW_FAILED"}
                )
            )
            .order_by(ExecutionRequest.updated_at.desc())
        )
        # The latest terminal request is the canonical execution explanation.
        for request in execution_res.scalars().all():
            execution_failure_map.setdefault(request.trade_history_id, request)

    settings_res = await db.execute(
        select(RuntimeSettings.key, RuntimeSettings.value)
    )
    settings_dict = {k: v for k, v in settings_res.all()}

    items = []
    for log, question, end_time_est in logs_with_questions:
        funnel = funnel_map.get(log.decision_run_id)
        # The funnel row is the canonical record for the decision that was
        # just evaluated. A TradeHistory row can predate direction telemetry
        # (or be a legacy SKIP row with NULL direction_value).
        funnel_direction_value = getattr(funnel, "direction_value", None) if funnel else None
        direction_value = (
            funnel_direction_value
            if funnel_direction_value is not None
            else getattr(log, "direction_value", None)
        )
        active_feat = getattr(log, "active_features", None)
        if not active_feat and log.status == "SKIPPED":
            base_asset = (
                log.asset.split("USDT")[0]
                if "USDT" in log.asset
                else log.asset.split("_")[0]
            )
            mode = settings_dict.get(
                f"TRADING_MODE_{base_asset}", settings_dict.get("TRADING_MODE", "ml")
            ).lower()
            if mode == "lightgbm":
                active_feat = "LIGHTGBM_TREND"
            elif mode == "combined":
                active_feat = "COMBINED"
            else:
                active_feat = "ml_strategy"

        execution_failure = execution_failure_map.get(log.id)
        error_msg = getattr(log, "error_msg", None)
        if not error_msg and log.status == "FAILED" and execution_failure:
            error_msg = (
                execution_failure.error_reason
                or execution_failure.terminal_code
                or f"Execution request {execution_failure.state}"
            )

        item = {
                "id": log.id,
                "market_id": log.market_id,
                "question": question or log.market_id,
                "end_time_est": end_time_est.isoformat() if end_time_est else None,
                "asset": log.asset,
                "status": log.status,
                "outcome_bought": log.outcome_bought,
                "amount_usdc": log.amount_usdc,
                "executed_price": log.executed_price,
                "predicted_flip_prob": log.predicted_flip_prob,
                "model_version": getattr(log, "model_version", None),
                "active_features": active_feat,
                "strategy_type": getattr(log, "strategy_type", None),
                "market_role": getattr(log, "market_role", None),
                "p_flip_effective": getattr(log, "p_flip_effective", None),
                "p_win_effective": getattr(log, "p_win_effective", None),
                "error_msg": error_msg,
                "mode": getattr(log, "mode", "LIVE"),
                "pnl": getattr(log, "pnl", None),
                "kelly_fraction": getattr(log, "kelly_fraction", None),
                "kelly_multiplier": getattr(log, "kelly_multiplier", None),
                "edge": getattr(log, "edge", None),
                "stop_loss_status": getattr(log, "stop_loss_status", None),
                "take_profit_status": getattr(log, "take_profit_status", None),
                "exit_reason": getattr(log, "exit_reason", None),
                "close_price": getattr(log, "close_price", None),
                "take_profit_hit_at": log.take_profit_hit_at.isoformat() if getattr(log, "take_profit_hit_at", None) else None,
                "take_profit_sell_price": getattr(log, "take_profit_sell_price", None),
                "take_profit_sell_size": getattr(log, "take_profit_sell_size", None),
                "created_at": log.created_at.isoformat(),
                "updated_at": (
                    log.updated_at.isoformat()
                    if getattr(log, "updated_at", None)
                    else None
                ),
                "direction_value": direction_value,
            }
        
        if funnel:
            mrf_audit = _mrf_audit_payload(funnel)
            item["funnel_log"] = {
                "direction_model_key": funnel.direction_model_key,
                "direction_model_version": funnel.direction_model_version,
                "direction_status": funnel.direction_status,
                "direction_p_up": funnel.direction_p_up,
                "direction_p_down": funnel.direction_p_down,
                "direction_probability": funnel.direction_probability,
                "direction_threshold_up": funnel.direction_threshold_up,
                "direction_threshold_down": funnel.direction_threshold_down,
                "direction_value": funnel.direction_value,
                "raw_opinion": getattr(funnel, "direction_raw_opinion", None),
                "actionable_signal": funnel.direction_value,
                "direction_p_up_raw": getattr(funnel, "direction_p_up_raw", None),
                "direction_p_down_raw": getattr(funnel, "direction_p_down_raw", None),
                "entry_model_key": funnel.entry_model_key,
                "entry_model_version": funnel.entry_model_version,
                "entry_status": funnel.entry_status,
                "p_flip": funnel.p_flip,
                "edge": funnel.edge,
                "min_edge_used": funnel.min_edge_used,
                "threshold_lower": funnel.threshold_lower,
                "threshold_upper": funnel.threshold_upper,
                "p_candidate_win": funnel.p_candidate_win,
                "candidate_ask": funnel.candidate_ask,
                "net_edge": funnel.net_edge,
                "fresh_price": funnel.fresh_price,
                "gates": {
                    "g1_model_loaded": funnel.g1_model_loaded,
                    "g2_price_fetched": funnel.g2_price_fetched,
                    "g3_dead_zone": funnel.g3_dead_zone,
                    "g4_no_flip": funnel.g4_no_flip,
                    "g5_min_edge": funnel.g5_min_edge,
                    "g6_price_range": funnel.g6_price_range,
                    "g7_crypto_confirm": funnel.g7_crypto_confirm,
                    "g8_combined_vote": funnel.g8_combined_vote,
                },
                "fallback_reason": funnel.fallback_reason,
                "reason": funnel.skip_reason,
                "mrf_audit": mrf_audit,
                "mrf": mrf_audit,
            }
            
        item["direction_display"] = direction_display_value(
            direction_value,
            getattr(funnel, "direction_status", None) if funnel else None,
            getattr(funnel, "entry_status", None) if funnel else None,
        )

        items.append(item)
        
    out_data = {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": math.ceil(total / page_size) if page_size > 0 else 0,
        "items": items,
    }
    _logs_cache[cache_key] = {"time": current_time, "data": out_data}
    return out_data


@router.post("/api/dashboard/verify_resolves", dependencies=[Depends(verify_api_key)])
async def verify_resolves(db: AsyncSession = Depends(get_db_session)):
    """Сверяет последние 50 разрешенных рынков из БД с актуальными данными Polymarket Gamma API"""

    # 1. Загружаем последние снепшоты с разрешенными исходами
    stmt = (
        select(MarketSnapshot)
        .where(MarketSnapshot.final_outcome != "PENDING")
        .order_by(MarketSnapshot.recorded_at.desc())
        .limit(200)
    )

    res = await db.execute(stmt)
    snapshots = res.scalars().all()

    # Отбираем уникальные market_id (до 50 штук)
    unique_markets = {}
    for s in snapshots:
        if s.market_id not in unique_markets and len(unique_markets) < 50:
            unique_markets[s.market_id] = {
                "asset": s.asset,
                "db_outcome": s.final_outcome,
            }

    if not unique_markets:
        return {
            "status": "success",
            "results": [],
            "message": "Нет разрешенных рынков в БД для проверки",
        }

    results = []
    semaphore = asyncio.Semaphore(10)  # не более 10 одновременных запросов

    async def fetch_market(client, market_id, info):
        async with semaphore:
            try:
                response = await client.get(
                    f"https://gamma-api.polymarket.com/markets/{market_id}"
                )

                if response.status_code == 200:
                    market_data = response.json()
                    question = market_data.get("question", "N/A")
                    closed = market_data.get("closed", False)

                    # Ищем ответ по той же логике, что и в resolver.py
                    answer = market_data.get("answer") or market_data.get(
                        "winnerOutcome"
                    )

                    if not answer and closed:
                        prices = market_data.get("outcomePrices", [])
                        outcomes = market_data.get("outcomes", ["Yes", "No"])
                        if isinstance(outcomes, str):
                            outcomes = json.loads(outcomes)
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        if (
                            prices
                            and len(prices) >= 2
                            and outcomes
                            and len(outcomes) >= 2
                        ):
                            try:
                                max_price = max(float(p) for p in prices)
                                if max_price >= 0.95:
                                    idx = [float(p) for p in prices].index(max_price)
                                    answer = outcomes[idx]
                            except Exception:
                                pass

                    if answer:
                        outcome_map = {
                            "UP": "YES",
                            "DOWN": "NO",
                            "YES": "YES",
                            "NO": "NO",
                        }
                        api_outcome = outcome_map.get(answer.upper(), answer.upper())

                        db_outcome = info["db_outcome"]
                        status = "OK" if db_outcome == api_outcome else "MISMATCH"

                        return {
                            "market_id": market_id,
                            "asset": info["asset"],
                            "question": question,
                            "db_outcome": db_outcome,
                            "api_outcome": api_outcome,
                            "status": status,
                        }
                    else:
                        return {
                            "market_id": market_id,
                            "asset": info["asset"],
                            "question": question,
                            "db_outcome": info["db_outcome"],
                            "api_outcome": "PENDING/UNRESOLVED",
                            "status": "UNRESOLVED_ON_API",
                        }
                else:
                    return {
                        "market_id": market_id,
                        "asset": info["asset"],
                        "question": f"Error HTTP {response.status_code}",
                        "db_outcome": info["db_outcome"],
                        "api_outcome": "ERROR",
                        "status": f"HTTP_ERROR_{response.status_code}",
                    }
            except Exception as e:
                return {
                    "market_id": market_id,
                    "asset": info["asset"],
                    "question": f"Request failed: {str(e)}",
                    "db_outcome": info["db_outcome"],
                    "api_outcome": "ERROR",
                    "status": "CONNECTION_FAILED",
                }
        return None

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [
            fetch_market(client, mid, info) for mid, info in unique_markets.items()
        ]
        fetched_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in fetched_results:
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, Exception):
                logger.error("error_fetching_market_data", error=str(r))

    return {"status": "success", "results": results}


@router.get("/api/dashboard/daily_pnl", dependencies=[Depends(verify_api_key)])
async def get_daily_pnl(
    timeframe: str = Query("24h", description="Период аналитики: 24h, 7d, 30d, all"),
    db: AsyncSession = Depends(get_db_session),
):
    """Возвращает отчет PnL по стратегиям за выбранный период (24h, 7d, 30d, all)."""
    now = datetime.now(timezone.utc)

    # Единый якорь времени: дата закрытия (если есть), иначе дата открытия
    _time_col = func.coalesce(TradeHistory.closed_at, TradeHistory.created_at)

    where_clause = [
        TradeHistory.position_status == "CLOSED",
        TradeHistory.pnl.is_not(None),
    ]

    if timeframe == "24h":
        start_time = now - timedelta(hours=24)
        where_clause.append(_time_col >= start_time)
    elif timeframe == "7d":
        start_time = now - timedelta(days=7)
        where_clause.append(_time_col >= start_time)
    elif timeframe == "30d":
        start_time = now - timedelta(days=30)
        where_clause.append(_time_col >= start_time)
    elif timeframe == "all":
        pass  # без фильтра по времени
    else:
        start_time = now - timedelta(hours=24)
        where_clause.append(_time_col >= start_time)

    stmt = select(
        TradeHistory.asset,
        TradeHistory.active_features,
        TradeHistory.pnl,
        TradeHistory.realized_pnl_usdc,
        TradeHistory.amount_usdc,
        TradeHistory.executed_price,
        TradeHistory.mode,
    ).where(*where_clause)

    result = await db.execute(stmt)
    trades = result.all()

    aggregated = {}
    for row in trades:
        asset = row.asset.split("_")[0].split("USDT")[0].upper()
        features = (row.active_features or "").lower()

        if "аутсайдер" in features or "outsider" in features:
            strategy = "Аутсайдер"
        elif "фаворит" in features or "favorite" in features:
            strategy = "Фаворит"
        elif row.executed_price is not None:
            if float(row.executed_price) >= 0.5:
                strategy = "Фаворит"
            else:
                strategy = "Аутсайдер"
        else:
            strategy = "Другое"

        mode = getattr(row, "mode", "PAPER")
        key = f"{asset}_{strategy}_{mode}"
        if key not in aggregated:
            aggregated[key] = {
                "asset": asset,
                "strategy": strategy,
                "mode": mode,
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "volume": 0.0,
            }

        # Приоритет: realized_pnl_usdc (новые сделки), fallback: pnl (старые)
        # Явное приведение к float — realized_pnl_usdc приходит из БД как Decimal
        effective_pnl = (
            float(row.realized_pnl_usdc)
            if row.realized_pnl_usdc is not None
            else float(row.pnl or 0)
        )
        aggregated[key]["trades"] += 1
        if effective_pnl > 0:
            aggregated[key]["wins"] += 1
        aggregated[key]["pnl"] += effective_pnl
        aggregated[key]["volume"] += float(row.amount_usdc or 0)

    response_data = []
    for data in aggregated.values():
        wr = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
        response_data.append(
            {
                "asset": data["asset"],
                "strategy": data["strategy"],
                "mode": data["mode"],
                "trades": data["trades"],
                "win_rate": round(wr, 1),
                "pnl": round(data["pnl"], 2),
                "volume": round(data["volume"], 2),
            }
        )

    def sort_key(x):
        s_order = (
            0
            if x["strategy"] == "Аутсайдер"
            else (1 if x["strategy"] == "Фаворит" else 2)
        )
        return (s_order, x["asset"])

    response_data.sort(key=sort_key)

    return {"status": "success", "data": response_data}


def _normalize_model_asset(asset: str) -> str:
    if not asset:
        return ""
    return asset.split("_")[0].split("USDT")[0].upper()


@router.get("/api/dashboard/model_pnl", dependencies=[Depends(verify_api_key)])
async def get_model_pnl(
    requested_mode: str = Query("PAPER"), db: AsyncSession = Depends(get_db_session)
):
    """
    Возвращает PnL, число сделок и win-rate для каждой версии модели
    за период её активности (с trained_at до trained_at следующей версии).
    """
    current_time = time.time()
    cache_key = f"model_pnl_{requested_mode}"
    if (
        cache_key in _model_pnl_cache
        and current_time - _model_pnl_cache[cache_key].get("time", 0)
        < _MODEL_PNL_CACHE_TTL
    ):
        return _model_pnl_cache[cache_key]["data"]

    # 1. Загружаем все версии моделей из ModelRegistry
    models_stmt = select(
        ModelRegistry.asset,
        ModelRegistry.version,
        ModelRegistry.trained_at,
        ModelRegistry.is_active,
    ).order_by(ModelRegistry.asset, ModelRegistry.version)
    models_rows = (await db.execute(models_stmt)).all()

    from collections import defaultdict

    by_asset = defaultdict(list)
    for row in models_rows:
        by_asset[row.asset].append(row)

    periods = []  # (asset, version, since, until)
    now = datetime.now(timezone.utc)
    for asset, versions in by_asset.items():
        for i, m in enumerate(versions):
            since = m.trained_at
            if since and since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)

            # Для активных моделей период не должен обрезаться неактивными драфтами переобучения.
            # Интервал активной модели продолжается до следующей АКТИВНОЙ модели или до текущего времени (now).
            if m.is_active:
                next_active = next(
                    (other for other in versions[i + 1 :] if other.is_active), None
                )
                if next_active and next_active.trained_at:
                    until = next_active.trained_at
                    if until.tzinfo is None:
                        until = until.replace(tzinfo=timezone.utc)
                else:
                    until = now
            else:
                if i + 1 < len(versions) and versions[i + 1].trained_at:
                    until = versions[i + 1].trained_at
                    if until and until.tzinfo is None:
                        until = until.replace(tzinfo=timezone.utc)
                else:
                    until = now

            periods.append((asset, m.version, since, until))

    if not periods:
        return {"status": "success", "data": {}}

    valid_sinces = [s for _, _, s, _ in periods if s is not None]
    earliest_since = min(valid_sinces) if valid_sinces else None

    # 2. Выполняем точный запрос к TradeHistory.
    # Тянем и LogReg-поля (model_key/model_version), и LGBM-поля
    # (direction_model_key/direction_model_version), чтобы агрегировать PnL
    # отдельно для каждого типа слотов.
    pnl_expr = func.coalesce(
        TradeHistory.realized_pnl_usdc, cast(TradeHistory.pnl, Numeric)
    )
    trades_stmt = select(
        TradeHistory.model_key,
        TradeHistory.model_version,
        TradeHistory.direction_model_key,
        TradeHistory.direction_model_version,
        pnl_expr.label("pnl"),
        TradeHistory.model_attribution_source,
    ).where(
        TradeHistory.position_status == "CLOSED",
        TradeHistory.mode == requested_mode,
        pnl_expr.is_not(None),
    )
    trades_rows = (await db.execute(trades_stmt)).all()

    from collections import defaultdict

    # Бакет 1: LogReg атрибуция — ключ (model_key, model_version)
    trades_by_model = defaultdict(list)
    exact_trades_count = defaultdict(int)
    reconstructed_trades_count = defaultdict(int)

    # Бакет 2: LGBM атрибуция — ключ (direction_model_key, direction_model_version).
    # Одна сделка может попасть в оба бакета одновременно — это корректно,
    # т.к. они показываются в разных строках таблицы моделей.
    lgbm_trades_by_slot: defaultdict[tuple, list] = defaultdict(list)

    unattributed_trades = 0
    unattributed_pnl = 0.0

    for row in trades_rows:
        attr_src = row.model_attribution_source
        m_key = row.model_key
        pnl_val = float(row.pnl) if row.pnl is not None else 0.0

        # --- LogReg бакет ---
        if m_key is not None and attr_src in ("EXACT", "RECONSTRUCTED"):
            trades_by_model[(m_key, row.model_version)].append(pnl_val)
            if attr_src == "EXACT":
                exact_trades_count[(m_key, row.model_version)] += 1
            else:
                reconstructed_trades_count[(m_key, row.model_version)] += 1
        else:
            unattributed_trades += 1
            unattributed_pnl += pnl_val

        # --- LGBM бакет (параллельно, независимо от LogReg) ---
        dir_key = row.direction_model_key
        dir_ver = row.direction_model_version
        if dir_key is not None and dir_ver is not None:
            lgbm_trades_by_slot[(dir_key, dir_ver)].append(pnl_val)

    # Определяем, является ли слот LGBM-слотом (asset формата "SYMBOL_regime")
    _LGBM_REGIME_SUFFIXES = ("_low_vol", "_mid_vol", "_high_vol")

    def _is_lgbm_slot(asset: str) -> bool:
        return any(asset.endswith(suffix) for suffix in _LGBM_REGIME_SUFFIXES)

    # 3. Агрегируем результаты для каждой модели из реестра.
    # Для LGBM-слотов используем lgbm_trades_by_slot, для LogReg — trades_by_model.
    result_map = {}
    for row in models_rows:
        asset = row.asset
        version = row.version
        key = f"{asset}_v{version}"

        if _is_lgbm_slot(asset):
            # LGBM-слот: берём сделки, где direction_model_key == asset И direction_model_version == version
            valid_trades = lgbm_trades_by_slot.get((asset, version), [])
            total = len(valid_trades)
            total_pnl = sum(valid_trades) if total > 0 else 0.0
            wins = sum(1 for pnl in valid_trades if pnl > 0)
            result_map[key] = {
                "asset": asset,
                "version": version,
                "total_trades": total,
                "pnl": round(float(total_pnl), 2),
                "win_rate": round(wins / total * 100, 1) if total > 0 else None,
                "exact_trades": total,        # все LGBM записи — точные
                "reconstructed_trades": 0,
            }
        else:
            # LogReg-слот: старая логика без изменений
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
                "reconstructed_trades": reconstructed_trades_count.get((asset, version), 0),
            }

    result_map["_unattributed"] = {
        "total_trades": unattributed_trades,
        "pnl": round(float(unattributed_pnl), 2),
    }

    response_data = {"status": "success", "data": result_map}
    _model_pnl_cache[cache_key] = {"time": current_time, "data": response_data}

    return response_data
