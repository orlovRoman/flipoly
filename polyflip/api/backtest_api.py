# polyflip/api/backtest_api.py
"""
Backtest API: запуск, получение результатов, история, UI-страница.
"""
from __future__ import annotations
import os
import uuid
import time
import pickle
import asyncio
import statistics
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from enum import Enum
import concurrent.futures
import functools

from polyflip.db.connection import get_db_session, async_session
from polyflip.db.models import MarketSnapshot, ModelRegistry
from polyflip.api.auth import verify_api_key
from polyflip.config import settings
from polyflip.api.backtest_schemas import (
    BacktestConfig, BacktestResult, BacktestRunResponse,
    StrategyBreakdown, AssetBreakdown, EquityCurvePoint
)
from polyflip.backtesting.market_replay import group_snapshots_into_replays, rows_to_replays
from polyflip.backtesting.runner import BacktestRunner
from polyflip.backtesting.metrics import compute_trade_pnl

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Backtest"])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Кэш результатов в памяти (run_id → BacktestResult)
# В продакшне заменить на Redis, но для MVP достаточно
_results_cache: dict[str, BacktestResult] = {}
_MAX_CACHE_SIZE = 20  # храним последние 20 прогонов


# ─── UI Page ───────────────────────────────────────────────────────────────

@router.get("/backtest")
async def backtest_page(request: Request):
    """Страница дашборда бэктестов."""
    return templates.TemplateResponse(
        "backtest.html",
        {
            "request": request,
            "api_key": settings.API_KEY
        }
    )


# ─── Run Backtest ───────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"

class Job:
    def __init__(self, run_id: str):
        self.run_id    = run_id
        self.status    = JobStatus.PENDING
        self.progress  = 0
        self.result    = None
        self.error     = None
        self.started   = datetime.now(timezone.utc)
        self.finished  = None

_jobs: dict[str, Job] = {}
_MAX_JOBS = 10
_semaphore = asyncio.Semaphore(1)

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="backtest"
)

async def _run_cpu_task(fn, *args, timeout_sec=120, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        fn = functools.partial(fn, **kwargs)
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, fn, *args),
            timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Backtest exceeded {timeout_sec}s time limit. Reduce dataset size."
        )

@router.post("/api/backtest/submit", dependencies=[Depends(verify_api_key)])
async def submit_backtest(
    config: BacktestConfig,
    background_tasks: BackgroundTasks,
):
    """Принимает задачу, отвечает мгновенно, запускает в фоне."""
    if any(j.status == JobStatus.RUNNING for j in list(_jobs.values())):
        raise HTTPException(status_code=429, detail="Another backtest is already running. Please wait.")

    run_id = str(uuid.uuid4())
    job    = Job(run_id)
    _jobs[run_id] = job

    if len(_jobs) > _MAX_JOBS:
        oldest = min(
            (j for j in list(_jobs.values()) if j.status != JobStatus.RUNNING),
            key=lambda j: j.started, default=None
        )
        if oldest:
            del _jobs[oldest.run_id]

    background_tasks.add_task(_run_backtest_bg, run_id, config)
    return {"run_id": run_id, "status": "pending", "poll_url": f"/api/backtest/status/{run_id}"}


@router.get("/api/backtest/status/{run_id}", dependencies=[Depends(verify_api_key)])
async def get_job_status(run_id: str):
    job = _jobs.get(run_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    return {
        "run_id":   run_id,
        "status":   job.status,
        "progress": job.progress,
        "error":    job.error,
        "result":   job.result if job.status == JobStatus.COMPLETED else None,
        "elapsed_sec": ((job.finished or datetime.now(timezone.utc)) - job.started).total_seconds(),
    }


async def _run_backtest_bg(run_id: str, config: BacktestConfig):
    job = _jobs[run_id]
    job.status = JobStatus.RUNNING

    async with _semaphore:
        try:
            async with async_session() as db:
                job.progress = 10
                result = await _execute_backtest_logic(db, config, run_id, job)
                job.result   = result
                job.status   = JobStatus.COMPLETED
                job.progress = 100
                
                # Кэшируем для совместимости с get_backtest_result
                if len(_results_cache) >= _MAX_CACHE_SIZE:
                    oldest_key = next(iter(_results_cache))
                    del _results_cache[oldest_key]
                _results_cache[run_id] = result
        except Exception as e:
            logger.exception("backtest_bg_failed", run_id=run_id, error=str(e))
            job.status = JobStatus.FAILED
            job.error  = str(e)
        finally:
            job.finished = datetime.now(timezone.utc)


async def _execute_backtest_logic(db: AsyncSession, config: BacktestConfig, run_id: str, job: Job):
    started_at = datetime.now(timezone.utc)

    # 1. Загружаем стартовые тики для каждого рынка в торговом окне
    base_filters = [
        MarketSnapshot.asset.in_(config.assets),
        MarketSnapshot.final_outcome.in_(["YES", "NO"]),
        MarketSnapshot.time_left_min >= config.min_time_left_min,
        MarketSnapshot.time_left_min <= config.max_time_left_min,
    ]
    if config.date_from:
        base_filters.append(MarketSnapshot.recorded_at >= config.date_from)
    if config.date_to:
        base_filters.append(MarketSnapshot.recorded_at <= config.date_to)

    count_cte = (
        select(MarketSnapshot.market_id, func.count().label("total_snaps"))
        .where(*base_filters)
        .group_by(MarketSnapshot.market_id)
        .cte("market_counts")
    )

    rank_sub = (
        select(
            MarketSnapshot.id.label("snap_id"),
            MarketSnapshot.market_id.label("market_id"),
            func.row_number().over(
                partition_by=MarketSnapshot.market_id,
                order_by=MarketSnapshot.time_left_min.desc()
            ).label("rn"),
            count_cte.c.total_snaps,
        )
        .join(count_cte, MarketSnapshot.market_id == count_cte.c.market_id)
        .where(*base_filters)
        .subquery("ranked_snaps")
    )

    qualified_stmt = select(rank_sub.c.market_id).where(
        rank_sub.c.rn == 1,
        rank_sub.c.total_snaps >= config.min_snapshots_per_market,
    ).limit(config.max_markets)
    
    qualified_res = await db.execute(qualified_stmt)
    market_ids = [row[0] for row in qualified_res.all()]
    job.progress = 20

    if not market_ids:
        raise ValueError("No resolved snapshots found for given filters. Check assets and date range.")

    SNAPSHOT_COLS = [
        MarketSnapshot.id,
        MarketSnapshot.market_id,
        MarketSnapshot.asset,
        MarketSnapshot.recorded_at,
        MarketSnapshot.mid_price,
        MarketSnapshot.price_velocity,
        MarketSnapshot.time_left_min,
        MarketSnapshot.final_outcome,
        MarketSnapshot.volume_5min,
        MarketSnapshot.spread,
        MarketSnapshot.hour_of_day,
    ]

    stmt = select(*SNAPSHOT_COLS).where(MarketSnapshot.market_id.in_(market_ids))
    if config.date_from:
        stmt = stmt.where(MarketSnapshot.recorded_at >= config.date_from)
    if config.date_to:
        stmt = stmt.where(MarketSnapshot.recorded_at <= config.date_to)
    stmt = stmt.order_by(MarketSnapshot.market_id, MarketSnapshot.recorded_at)

    result = await db.stream(stmt.execution_options(yield_per=1000))
    rows = []
    async for partition in result.partitions(1000):
        rows.extend(partition)
        await asyncio.sleep(0)
    
    job.progress = 40
    total_loaded = len(rows)

    replays = await _run_cpu_task(rows_to_replays, rows, 1, timeout_sec=60)
    tradeable = len(replays)
    job.progress = 60

    total_markets_in_window = await db.scalar(
        select(func.count(func.distinct(MarketSnapshot.market_id))).where(*base_filters)
    )
    skipped = max(0, (total_markets_in_window or 0) - tradeable)

    model_blob: Optional[bytes] = None
    features_str: str = ""
    if config.strategy_mode == "ML":
        model_stmt = select(ModelRegistry).where(ModelRegistry.is_active == True)
        if config.model_id:
            model_stmt = select(ModelRegistry).where(ModelRegistry.id == config.model_id)
        elif config.assets:
            model_stmt = model_stmt.where(ModelRegistry.asset == config.assets[0])
        
        model_row = (await db.execute(model_stmt)).scalars().first()
        if model_row:
            model_blob = model_row.model_blob
            features_str = model_row.features or ""

    runner_config = config.to_runner_config()
    runner = BacktestRunner(runner_config, model_blob, features_str)
    
    job.progress = 70
    trades = await _run_cpu_task(runner.run_all, replays, timeout_sec=180)
    job.progress = 90

    finished_at = datetime.now(timezone.utc)
    backtest_result = await _run_cpu_task(
        _build_result,
        run_id, config, started_at, finished_at,
        total_loaded, tradeable, skipped, trades, replays,
        timeout_sec=60
    )
    return backtest_result


# ─── Get Result ─────────────────────────────────────────────────────────────

@router.get(
    "/api/backtest/result/{run_id}",
    response_model=BacktestRunResponse,
    dependencies=[Depends(verify_api_key)]
)
async def get_backtest_result(run_id: str):
    """Возвращает результат прогона по run_id."""
    result = _results_cache.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found in cache")
    return BacktestRunResponse(
        run_id=run_id,
        status="completed",
        message="Cached result",
        result=result
    )


# ─── List Recent Runs ────────────────────────────────────────────────────────

@router.get(
    "/api/backtest/history",
    dependencies=[Depends(verify_api_key)]
)
async def list_backtest_history():
    """Список последних прогонов (без equity_curve для экономии трафика)."""
    history = []
    for run_id, res in reversed(list(_results_cache.items())):
        history.append({
            "run_id": run_id,
            "started_at": res.started_at.isoformat(),
            "duration_sec": res.duration_sec,
            "assets": res.config.assets,
            "strategy_mode": res.config.strategy_mode,
            "total_trades": res.total_trades,
            "net_profit": res.net_profit,
            "roi_pct": res.roi_pct,
            "win_rate_pct": res.win_rate_pct,
        })
    return {"runs": history}


# ─── Available Models ────────────────────────────────────────────────────────

@router.get(
    "/api/backtest/models",
    dependencies=[Depends(verify_api_key)]
)
async def list_available_models(db: AsyncSession = Depends(get_db_session)):
    """Список доступных моделей для выбора в UI."""
    stmt = select(
        ModelRegistry.id,
        ModelRegistry.asset,
        ModelRegistry.version,
        ModelRegistry.is_active,
        ModelRegistry.trained_at,
        ModelRegistry.features,
    ).order_by(ModelRegistry.trained_at.desc())
    rows = (await db.execute(stmt)).all()
    return {
        "models": [
            {
                "id": r.id,
                "asset": r.asset,
                "version": r.version,
                "is_active": r.is_active,
                "trained_at": r.trained_at.isoformat() if r.trained_at else None,
                "features": r.features,
            }
            for r in rows
        ]
    }


# ─── Dataset Stats ────────────────────────────────────────────────────────────

@router.get(
    "/api/backtest/dataset_stats",
    dependencies=[Depends(verify_api_key)]
)
async def get_dataset_stats(db: AsyncSession = Depends(get_db_session)):
    """Статистика доступных данных: кол-во снепшотов, рынков, период по каждому ассету."""
    from sqlalchemy import func, distinct
    stmt = select(
        MarketSnapshot.asset,
        MarketSnapshot.final_outcome,
        func.count(distinct(MarketSnapshot.market_id)).label("markets"),
        func.count(MarketSnapshot.id).label("snapshots"),
        func.min(MarketSnapshot.recorded_at).label("date_from"),
        func.max(MarketSnapshot.recorded_at).label("date_to"),
    ).group_by(MarketSnapshot.asset, MarketSnapshot.final_outcome)
    rows = (await db.execute(stmt)).all()

    result: dict = {}
    for r in rows:
        if r.asset not in result:
            result[r.asset] = {}
        result[r.asset][r.final_outcome] = {
            "markets": r.markets,
            "snapshots": r.snapshots,
            "date_from": r.date_from.isoformat() if r.date_from else None,
            "date_to": r.date_to.isoformat() if r.date_to else None,
        }
    return {"stats": result}


# ─── Internal: build BacktestResult ──────────────────────────────────────────

def _build_result(
    run_id: str,
    config: BacktestConfig,
    started_at: datetime,
    finished_at: datetime,
    total_loaded: int,
    tradeable: int,
    skipped: int,
    trades,
    replays,
) -> BacktestResult:
    """Собирает BacktestResult из raw trades + replays."""
    duration = (finished_at - started_at).total_seconds()

    if not trades:
        return BacktestResult(
            run_id=run_id, config=config,
            started_at=started_at, finished_at=finished_at,
            duration_sec=duration,
            total_markets_loaded=total_loaded,
            tradeable_markets=tradeable, skipped_markets=skipped,
            total_trades=0, total_invested=0, net_profit=0,
            roi_pct=0, win_rate_pct=0, avg_trade_pnl=0,
            max_drawdown_pct=0, sharpe_ratio=None, profit_factor=0,
            strategies=[], assets=[], equity_curve=[],
            top_trades=[], worst_trades=[]
        )

    # Считаем PnL по каждой сделке
    trade_results = []
    for t in trades:
        replay = replays.get(t.market_id)
        if not replay:
            continue
        pnl = compute_trade_pnl(t, replay)
        won = pnl > 0
        trade_results.append({
            "trade": t, "pnl": pnl, "won": won,
            "replay": replay
        })

    if not trade_results:
        return BacktestResult(
            run_id=run_id, config=config,
            started_at=started_at, finished_at=finished_at,
            duration_sec=duration,
            total_markets_loaded=total_loaded,
            tradeable_markets=tradeable, skipped_markets=skipped,
            total_trades=0, total_invested=0, net_profit=0,
            roi_pct=0, win_rate_pct=0, avg_trade_pnl=0,
            max_drawdown_pct=0, sharpe_ratio=None, profit_factor=0,
            strategies=[], assets=[], equity_curve=[],
            top_trades=[], worst_trades=[]
        )

    # Базовые метрики
    total_invested = sum(tr["trade"].bet_size for tr in trade_results)
    net_profit = sum(tr["pnl"] for tr in trade_results)
    wins = [tr for tr in trade_results if tr["won"]]
    losses = [tr for tr in trade_results if not tr["won"]]
    win_rate = len(wins) / len(trade_results) * 100
    roi = (net_profit / total_invested * 100) if total_invested > 0 else 0
    avg_pnl = net_profit / len(trade_results)

    # Equity curve (накопленный PnL)
    equity_curve = []
    cumulative = 0.0
    for i, tr in enumerate(trade_results):
        t = tr["trade"]
        cumulative += tr["pnl"]
        equity_curve.append(EquityCurvePoint(
            trade_index=i,
            cumulative_pnl=round(cumulative, 4),
            trade_pnl=round(tr["pnl"], 4),
            market_id=t.market_id,
            asset=t.asset,
            strategy=t.decision.strategy_type,
            outcome="WIN" if tr["won"] else "LOSS",
            p_flip=t.p_flip,
            edge=t.decision.edge,
            bet_size=t.bet_size,
            executed_price=t.executed_price,
        ))

    # Max Drawdown
    max_dd = _compute_max_drawdown(equity_curve)

    # Sharpe (если > 1 сделка)
    sharpe = None
    if len(trade_results) > 1:
        pnls = [tr["pnl"] for tr in trade_results]
        try:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            sharpe = round(mean_pnl / std_pnl, 3) if std_pnl > 0 else None
        except Exception:
            pass

    # Profit factor
    gross_profit = sum(tr["pnl"] for tr in wins)
    gross_loss = abs(sum(tr["pnl"] for tr in losses))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else 999.0

    # По стратегиям
    strat_map: dict[str, list] = {}
    for tr in trade_results:
        s = tr["trade"].decision.strategy_type
        strat_map.setdefault(s, []).append(tr)

    strategies = []
    for strat, items in strat_map.items():
        s_wins = [x for x in items if x["won"]]
        edges = [x["trade"].decision.edge for x in items if x["trade"].decision.edge is not None]
        strategies.append(StrategyBreakdown(
            strategy=strat,
            trades=len(items),
            net_pnl=round(sum(x["pnl"] for x in items), 4),
            win_rate_pct=round(len(s_wins) / len(items) * 100, 2),
            avg_edge=round(statistics.mean(edges), 4) if edges else None,
        ))

    # По ассетам
    asset_map: dict[str, list] = {}
    for tr in trade_results:
        a = tr["trade"].asset
        asset_map.setdefault(a, []).append(tr)

    assets_breakdown = []
    for asset, items in asset_map.items():
        a_wins = [x for x in items if x["won"]]
        assets_breakdown.append(AssetBreakdown(
            asset=asset,
            trades=len(items),
            net_pnl=round(sum(x["pnl"] for x in items), 4),
            win_rate_pct=round(len(a_wins) / len(items) * 100, 2),
        ))

    # Топ/Худшие
    sorted_desc = sorted(equity_curve, key=lambda x: x.trade_pnl, reverse=True)
    sorted_asc  = sorted(equity_curve, key=lambda x: x.trade_pnl)

    return BacktestResult(
        run_id=run_id, config=config,
        started_at=started_at, finished_at=finished_at,
        duration_sec=round(duration, 3),
        total_markets_loaded=total_loaded,
        tradeable_markets=tradeable, skipped_markets=skipped,
        total_trades=len(trade_results),
        total_invested=round(total_invested, 2),
        net_profit=round(net_profit, 4),
        roi_pct=round(roi, 2),
        win_rate_pct=round(win_rate, 2),
        avg_trade_pnl=round(avg_pnl, 4),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        strategies=strategies,
        assets=assets_breakdown,
        equity_curve=equity_curve,
        top_trades=sorted_desc[:10],
        worst_trades=sorted_asc[:10],
    )


def _compute_max_drawdown(equity_curve: list[EquityCurvePoint]) -> float:
    """Максимальная просадка от пика к впадине в %."""
    if not equity_curve:
        return 0.0
    values = [p.cumulative_pnl for p in equity_curve]
    peak = values[0]   # FIX: инициализируем первым реальным значением, не нулём
    max_dd = 0.0
    for val in values:
        if val > peak:
            peak = val
        if abs(peak) > 1e-9:
            dd = (peak - val) / abs(peak) * 100
        else:
            dd = 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd
