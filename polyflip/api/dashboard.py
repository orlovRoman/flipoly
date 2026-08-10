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
from fastapi import APIRouter, Request, Depends, Query
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
        "index.html",
        {
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
        "execution.html",
        {
            "request": request,
            "timestamp": int(time.time()),
            "static_version": STATIC_VERSION,
            "root_path": request.scope.get("root_path", ""),
        },
    )


_dashboard_cache = {}
_DASHBOARD_CACHE_TTL = 30  # 30 секунд кэша

_model_pnl_cache = {}
_MODEL_PNL_CACHE_TTL = 300  # 300 секунд кэша


def invalidate_dashboard_cache():
    _dashboard_cache.clear()
    _model_pnl_cache.clear()


_logs_cache = {}
_LOGS_CACHE_TTL = 10  # 10 секунд кэша для логов торговли
_logs_cache_lock = asyncio.Lock()


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

    settings_res = await db.execute(
        select(RuntimeSettings.key, RuntimeSettings.value)
    )
    settings_dict = {k: v for k, v in settings_res.all()}

    items = []
    for log, question, end_time_est in logs_with_questions:
        funnel = funnel_map.get(log.decision_run_id)
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
                "error_msg": log.error_msg,
                "mode": getattr(log, "mode", "LIVE"),
                "pnl": getattr(log, "pnl", None),
                "kelly_fraction": getattr(log, "kelly_fraction", None),
                "kelly_multiplier": getattr(log, "kelly_multiplier", None),
                "edge": getattr(log, "edge", None),
                "stop_loss_status": getattr(log, "stop_loss_status", None),
                "take_profit_status": getattr(log, "take_profit_status", None),
                "take_profit_hit_at": log.take_profit_hit_at.isoformat() if getattr(log, "take_profit_hit_at", None) else None,
                "take_profit_sell_price": getattr(log, "take_profit_sell_price", None),
                "take_profit_sell_size": getattr(log, "take_profit_sell_size", None),
                "created_at": log.created_at.isoformat(),
                "updated_at": (
                    log.updated_at.isoformat()
                    if getattr(log, "updated_at", None)
                    else None
                ),
                "direction_value": getattr(log, "direction_value", None),
            }
        
        if funnel:
            item["funnel_log"] = {
                "direction_model_key": funnel.direction_model_key,
                "direction_model_version": funnel.direction_model_version,
                "direction_status": funnel.direction_status,
                "direction_p_up": funnel.direction_p_up,
                "direction_p_down": funnel.direction_p_down,
                "direction_probability": funnel.direction_probability,
                "direction_threshold_up": funnel.direction_threshold_up,
                "direction_threshold_down": funnel.direction_threshold_down,
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
            }
            
        item["direction_display"] = direction_display_value(
            getattr(log, "direction_value", None),
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

    # 2. Выполняем точный запрос к TradeHistory
    pnl_expr = func.coalesce(
        TradeHistory.realized_pnl_usdc, cast(TradeHistory.pnl, Numeric)
    )
    trades_stmt = select(
        TradeHistory.model_key,
        TradeHistory.model_version,
        pnl_expr.label("pnl"),
        TradeHistory.model_attribution_source,
    ).where(
        TradeHistory.position_status == "CLOSED",
        TradeHistory.mode == requested_mode,
        pnl_expr.is_not(None),
    )
    trades_rows = (await db.execute(trades_stmt)).all()

    # Группируем сделки по (model_key, model_version)
    from collections import defaultdict

    trades_by_model = defaultdict(list)
    exact_trades_count = defaultdict(int)
    reconstructed_trades_count = defaultdict(int)

    unattributed_trades = 0
    unattributed_pnl = 0.0

    for row in trades_rows:
        attr_src = row.model_attribution_source
        m_key = row.model_key
        pnl_val = float(row.pnl) if row.pnl is not None else 0.0

        if m_key is not None and attr_src in ("EXACT", "RECONSTRUCTED"):
            trades_by_model[(m_key, row.model_version)].append(pnl_val)
            if attr_src == "EXACT":
                exact_trades_count[(m_key, row.model_version)] += 1
            else:
                reconstructed_trades_count[(m_key, row.model_version)] += 1
        else:
            unattributed_trades += 1
            unattributed_pnl += pnl_val

    # 3. Агрегируем результаты для каждой модели из реестра
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
            "reconstructed_trades": reconstructed_trades_count.get((asset, version), 0),
        }

    result_map["_unattributed"] = {
        "total_trades": unattributed_trades,
        "pnl": round(float(unattributed_pnl), 2),
    }

    response_data = {"status": "success", "data": result_map}
    _model_pnl_cache[cache_key] = {"time": current_time, "data": response_data}

    return response_data
