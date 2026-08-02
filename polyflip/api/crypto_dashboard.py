# polyflip/api/crypto_dashboard.py
"""
Дашборд для крипто-домена (LightGBM Up/Down).
Полностью изолирован от Polymarket-дашборда.
Подключается в main.py одной строкой.

"""

from __future__ import annotations

import asyncio
import os

STATIC_VERSION = os.getenv("POLYFLIP_BUILD_SHA", "dev")
import json
import time
from datetime import datetime, timezone
import numpy as np

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Query, HTTPException
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, update, delete, func, cast, Numeric, text
from collections import defaultdict
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.api.auth import verify_api_key
import polyflip.constants as C
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.feature_builder import build_features
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.trainer import CryptoModelTrainer
from polyflip.db.connection import async_session, get_db_session
from polyflip.db.models import ModelRegistry, TradeHistory, RuntimeSettings
from polyflip.settings_registry import registry_defaults

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/crypto", tags=["Crypto"])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "XRPUSDT", "SOLUSDT"]

# Кэш и активные процессы обучения
_cache: dict = {}
_CACHE_TTL = 10  # снизим до 10 секунд для лучшей отзывчивости настроек
_active_trainings: dict[str, dict] = {}


@router.get("")
async def crypto_page(request: Request):
    """HTML-страница крипто-дашборда."""
    defs = registry_defaults()
    api_key = request.cookies.get("api_key", "")
    return templates.TemplateResponse(
        request=request,
        name="crypto.html",
        context={
            "symbols": CRYPTO_SYMBOLS,
            "root_path": request.scope.get("root_path", ""),
            "api_key": api_key,
            "defaults": {
                "n_estimators": int(defs.get("CRYPTO_LGBM_N_ESTIMATORS", "300")),
                "learning_rate": float(defs.get("CRYPTO_LGBM_LEARNING_RATE", "0.05")),
                "num_leaves": int(defs.get("CRYPTO_LGBM_NUM_LEAVES", "31")),
                "max_depth": int(defs.get("CRYPTO_LGBM_MAX_DEPTH", "5")),
                "min_child_samples": int(
                    defs.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20")
                ),
                "subsample": float(defs.get("CRYPTO_LGBM_SUBSAMPLE", "0.8")),
                "colsample_bytree": float(
                    defs.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8")
                ),
                "reg_alpha": float(defs.get("CRYPTO_LGBM_REG_ALPHA", "0.1")),
                "reg_lambda": float(defs.get("CRYPTO_LGBM_REG_LAMBDA", "1.0")),
                "min_edge": float(defs.get("BACKTEST_MIN_EDGE", "0.04")),
            },
        },
    )


@router.get("/api/status", dependencies=[Depends(verify_api_key)])
async def crypto_status(db: AsyncSession = Depends(get_db_session)):
    """
    Возвращает текущее состояние крипто-моделей:
    версию, AUC, ECE, порог, список фич, дату обучения, важность фичей и гиперпараметры.
    """
    now = time.time()
    if "status" in _cache and now - _cache["status"]["ts"] < _CACHE_TTL:
        res = dict(_cache["status"]["data"])
        res["active_trainings"] = _active_trainings
        return res

    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    stmt = (
        select(ModelRegistry)
        .where(
            ModelRegistry.asset.in_(allowed_assets),
        )
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()

    # Пороги из RuntimeSettings
    thr_keys = [f"CRYPTO_THRESHOLD_{a}" for a in allowed_assets]
    thr_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(thr_keys))
    thr_rows = (await db.execute(thr_stmt)).scalars().all()
    thresholds = {
        r.key.replace("CRYPTO_THRESHOLD_", ""): float(r.value) for r in thr_rows
    }

    # Важность признаков из RuntimeSettings
    fi_keys = [f"CRYPTO_FI_{a}" for a in allowed_assets]
    fi_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(fi_keys))
    fi_rows = (await db.execute(fi_stmt)).scalars().all()
    feature_importances = {}
    for r in fi_rows:
        try:
            sym = r.key.replace("CRYPTO_FI_", "")
            feature_importances[sym] = json.loads(r.value)
        except Exception:
            pass

    # Текущие гиперпараметры обучения из БД
    settings_keys = [
        "CRYPTO_LGBM_N_ESTIMATORS",
        "CRYPTO_LGBM_LEARNING_RATE",
        "CRYPTO_LGBM_NUM_LEAVES",
        "CRYPTO_LGBM_MAX_DEPTH",
        "CRYPTO_LGBM_MIN_CHILD_SAMPLES",
        "CRYPTO_LGBM_SUBSAMPLE",
        "CRYPTO_LGBM_COLSAMPLE_BYTREE",
        "CRYPTO_LGBM_REG_ALPHA",
        "CRYPTO_LGBM_REG_LAMBDA",
        "BACKTEST_MIN_EDGE",
        "LGBM_EPSILON_QUANTILE",
    ]
    set_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    set_rows = (await db.execute(set_stmt)).scalars().all()
    db_settings = {r.key: r.value for r in set_rows}

    defs = registry_defaults()
    active_settings = {
        "n_estimators": int(
            db_settings.get(
                "CRYPTO_LGBM_N_ESTIMATORS", defs.get("CRYPTO_LGBM_N_ESTIMATORS", "300")
            )
        ),
        "learning_rate": float(
            db_settings.get(
                "CRYPTO_LGBM_LEARNING_RATE",
                defs.get("CRYPTO_LGBM_LEARNING_RATE", "0.05"),
            )
        ),
        "num_leaves": int(
            db_settings.get(
                "CRYPTO_LGBM_NUM_LEAVES", defs.get("CRYPTO_LGBM_NUM_LEAVES", "31")
            )
        ),
        "max_depth": int(
            db_settings.get(
                "CRYPTO_LGBM_MAX_DEPTH", defs.get("CRYPTO_LGBM_MAX_DEPTH", "5")
            )
        ),
        "min_child_samples": int(
            db_settings.get(
                "CRYPTO_LGBM_MIN_CHILD_SAMPLES",
                defs.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20"),
            )
        ),
        "subsample": float(
            db_settings.get(
                "CRYPTO_LGBM_SUBSAMPLE", defs.get("CRYPTO_LGBM_SUBSAMPLE", "0.8")
            )
        ),
        "colsample_bytree": float(
            db_settings.get(
                "CRYPTO_LGBM_COLSAMPLE_BYTREE",
                defs.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8"),
            )
        ),
        "reg_alpha": float(
            db_settings.get(
                "CRYPTO_LGBM_REG_ALPHA", defs.get("CRYPTO_LGBM_REG_ALPHA", "0.1")
            )
        ),
        "reg_lambda": float(
            db_settings.get(
                "CRYPTO_LGBM_REG_LAMBDA", defs.get("CRYPTO_LGBM_REG_LAMBDA", "1.0")
            )
        ),
        "min_edge": float(
            db_settings.get("BACKTEST_MIN_EDGE", defs.get("BACKTEST_MIN_EDGE", "0.04"))
        ),
        "epsilon_quantile": float(
            db_settings.get(
                "LGBM_EPSILON_QUANTILE", defs.get("LGBM_EPSILON_QUANTILE", "0.6")
            )
        ),
    }

    models_info = {}
    for m in rows:
        key = f"{m.asset}_v{m.version}"
        models_info[key] = {
            "asset": m.asset,
            "version": m.version,
            "is_active": m.is_active,
            "auc": round(m.accuracy, 4),
            "baseline": round(m.baseline, 4),
            "ece": round(m.ece, 4) if getattr(m, "ece", None) else None,
            "threshold": thresholds.get(m.asset),
            "features": m.features.split(",") if getattr(m, "features", None) else [],
            "trained_at": (
                m.trained_at.isoformat() if getattr(m, "trained_at", None) else None
            ),
            "feature_importance": feature_importances.get(m.asset, {}),
        }

    result = {
        "models": models_info,
        "symbols": CRYPTO_SYMBOLS,
        "settings": active_settings,
        "active_trainings": _active_trainings,
        "feature_importances": {
            asset: feature_importances.get(asset, {})
            for asset in set(m.asset for m in rows if m.is_active)
        },
    }
    _cache["status"] = {"ts": now, "data": result}
    return result


@router.post("/api/settings", dependencies=[Depends(verify_api_key)])
async def save_crypto_settings(
    settings: dict, db: AsyncSession = Depends(get_db_session)
):
    """Сохраняет измененные гиперпараметры в RuntimeSettings."""
    now = datetime.now(timezone.utc)
    keys_map = {
        "n_estimators": "CRYPTO_LGBM_N_ESTIMATORS",
        "learning_rate": "CRYPTO_LGBM_LEARNING_RATE",
        "num_leaves": "CRYPTO_LGBM_NUM_LEAVES",
        "max_depth": "CRYPTO_LGBM_MAX_DEPTH",
        "min_child_samples": "CRYPTO_LGBM_MIN_CHILD_SAMPLES",
        "subsample": "CRYPTO_LGBM_SUBSAMPLE",
        "colsample_bytree": "CRYPTO_LGBM_COLSAMPLE_BYTREE",
        "reg_alpha": "CRYPTO_LGBM_REG_ALPHA",
        "reg_lambda": "CRYPTO_LGBM_REG_LAMBDA",
        "min_edge": "BACKTEST_MIN_EDGE",
        "epsilon_quantile": "LGBM_EPSILON_QUANTILE",
    }

    for key, db_key in keys_map.items():
        if key in settings:
            val_str = str(settings[key])
            row = (
                await db.execute(
                    select(RuntimeSettings).where(RuntimeSettings.key == db_key)
                )
            ).scalar_one_or_none()
            if row:
                row.value = val_str
                row.updated_at = now
                row.updated_by = "crypto_dashboard_ui"
            else:
                db.add(
                    RuntimeSettings(
                        key=db_key,
                        value=val_str,
                        updated_at=now,
                        updated_by="crypto_dashboard_ui",
                    )
                )

    await db.commit()
    _cache.pop("status", None)
    return {"status": "success", "message": "Настройки успешно сохранены!"}


@router.get("/api/backtest", dependencies=[Depends(verify_api_key)])
async def crypto_backtest(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    min_edge: float | None = Query(None),
    commission: float | None = Query(None),
):
    """
    Запускает walk-forward backtest и возвращает детальные метрики и PnL-кривую.
    Результат кэшируется на 5 минут для предотвращения перегрузки CPU.
    """
    cache_key = f"backtest_{symbol}_{interval}_{min_edge}_{commission}"
    now = time.time()
    if cache_key in _cache and now - _cache[cache_key]["ts"] < 300:
        return _cache[cache_key]["data"]

    from polyflip.services.settings_service import get_float, get_int

    async with async_session() as session:
        if min_edge is None:
            min_edge = await get_float(session, "BACKTEST_MIN_EDGE")

        lgbm_params = {
            "subsample": await get_float(session, "CRYPTO_LGBM_SUBSAMPLE"),
            "colsample_bytree": await get_float(
                session, "CRYPTO_LGBM_COLSAMPLE_BYTREE"
            ),
            "num_leaves": await get_int(session, "CRYPTO_LGBM_NUM_LEAVES"),
            "max_depth": await get_int(session, "CRYPTO_LGBM_MAX_DEPTH"),
            "min_child_samples": await get_int(
                session, "CRYPTO_LGBM_MIN_CHILD_SAMPLES"
            ),
            "n_estimators": await get_int(session, "CRYPTO_LGBM_N_ESTIMATORS"),
            "reg_alpha": await get_float(session, "CRYPTO_LGBM_REG_ALPHA"),
            "reg_lambda": await get_float(session, "CRYPTO_LGBM_REG_LAMBDA"),
        }

        candles = await get_recent_candles(session, symbol, interval, limit=10_000)

    if len(candles) < 600:
        return {
            "error": f"Недостаточно свечей: {len(candles)} < 600. Пожалуйста, сделайте backfill.",
            "symbol": symbol,
        }

    df = build_features(candles)

    # Запускаем backtest в пуле потоков (CPU-bound)
    result = await asyncio.to_thread(
        run_backtest, df, symbol, min_edge, commission, lgbm_params=lgbm_params
    )

    data = {
        "symbol": result.symbol,
        "n_candles_total": result.n_candles_total,
        "n_candles_test": result.n_candles_test,
        "n_trades": result.n_trades,
        "win_rate": round(result.win_rate, 4),
        "total_return_net": round(result.total_return_net, 5),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "max_drawdown": round(result.max_drawdown, 5),
        "edge_rate": round(result.edge_rate, 4),
        "epsilon": round(result.epsilon, 6),
        "train_auc": round(result.train_auc, 4),
        "is_profitable": result.is_profitable(),
        "summary": result.summary(),
        "pnl_curve": result.pnl_curve,
    }

    _cache[cache_key] = {"ts": now, "data": data}
    return data


@router.post("/api/train", dependencies=[Depends(verify_api_key)])
async def crypto_train(
    background_tasks: BackgroundTasks,
    symbol: str = "BTCUSDT",
    interval: str = "15m",
):
    """
    Запускает переобучение LightGBM-модели в фоне.
    Не блокирует HTTP-ответ — обучение идёт в background task.
    """
    if symbol in _active_trainings:
        return {
            "status": "already_running",
            "symbol": symbol,
            "message": f"Обучение модели {symbol} уже выполняется.",
        }

    _active_trainings[symbol] = {
        "status": "training",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
    }
    _cache.pop("status", None)

    async def _train():
        try:
            async with async_session() as session:
                trainer = CryptoModelTrainer(session)
                ok = await trainer.train(symbol, interval)
                logger.info("crypto_retrain_done", symbol=symbol, success=ok)
        except Exception as exc:
            logger.exception("crypto_retrain_error", symbol=symbol, error=str(exc))
        finally:
            _active_trainings.pop(symbol, None)
            _cache.pop("status", None)
            for k in list(_cache.keys()):
                if k.startswith(f"backtest_{symbol}"):
                    _cache.pop(k, None)

    background_tasks.add_task(_train)
    return {
        "status": "started",
        "symbol": symbol,
        "message": f"Переобучение {symbol} запущено в фоне.",
    }


@router.get("/api/models/analytics", dependencies=[Depends(verify_api_key)])
async def crypto_models_analytics(
    requested_mode: str = "PAPER",
    date_from: str = None,
    date_to: str = None,
    db: AsyncSession = Depends(get_db_session)
):
    cache_key = f"crypto_model_analytics_{requested_mode}_{date_from}_{date_to}"
    now = time.time()
    if cache_key in _cache:
        c = _cache[cache_key]
        if now - c["ts"] < 30:
            return c["data"]

    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    # 1. Запрашиваем модели
    stmt = (
        select(ModelRegistry)
        .where(
            ModelRegistry.asset.in_(allowed_assets),
        )
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    models = (await db.execute(stmt)).scalars().all()

    # Фильтры дат
    date_filter = ""
    if date_from:
        date_filter += f" AND closed_at >= :date_from"
    if date_to:
        date_filter += f" AND closed_at <= :date_to"

    # 2. PRIMARY CTE
    primary_sql = text(f"""
        WITH trades AS (
            SELECT 
                model_key,
                model_version,
                COALESCE(realized_pnl_usdc, pnl) as pnl,
                closed_at,
                id
            FROM trade_history
            WHERE mode = :mode
              AND position_status = 'CLOSED'
              AND model_key IS NOT NULL
              AND model_attribution_source IN ('EXACT', 'RECONSTRUCTED')
              AND COALESCE(realized_pnl_usdc, pnl) IS NOT NULL
              {date_filter}
        ),
        zero_trade AS (
            SELECT DISTINCT model_key, model_version, 0 as pnl, '1970-01-01' as closed_at, 0 as id FROM trades
        ),
        all_trades AS (
            SELECT model_key, model_version, pnl, closed_at, id FROM trades
            UNION ALL
            SELECT model_key, model_version, pnl, closed_at, id FROM zero_trade
        ),
        cumulatives AS (
            SELECT 
                model_key, model_version, pnl,
                SUM(pnl) OVER (PARTITION BY model_key, model_version ORDER BY closed_at ASC, id ASC) as equity
            FROM all_trades
        ),
        peaks AS (
            SELECT 
                model_key, model_version, pnl, equity,
                MAX(equity) OVER (PARTITION BY model_key, model_version ORDER BY closed_at ASC, id ASC) as running_peak
            FROM cumulatives
        ),
        drawdowns AS (
            SELECT 
                model_key, model_version, pnl, equity, running_peak,
                (equity - running_peak) as drawdown
            FROM peaks
        ),
        agg_trades AS (
            SELECT 
                model_key, model_version,
                COUNT(*) as total_trades,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_count,
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)) as gross_loss
            FROM trades
            GROUP BY model_key, model_version
        ),
        agg_drawdowns AS (
            SELECT 
                model_key, model_version,
                MIN(drawdown) as max_drawdown
            FROM drawdowns
            GROUP BY model_key, model_version
        )
        SELECT 
            t.model_key, t.model_version,
            t.total_trades, t.total_pnl, t.win_count, t.gross_profit, t.gross_loss,
            d.max_drawdown
        FROM agg_trades t
        LEFT JOIN agg_drawdowns d ON t.model_key = d.model_key AND t.model_version = d.model_version
    """)
    
    params = {"mode": requested_mode}
    if date_from: params["date_from"] = date_from
    if date_to: params["date_to"] = date_to
    
    primary_rows = (await db.execute(primary_sql, params)).fetchall()
    primary_stats = {(row.model_key, row.model_version): row for row in primary_rows}

    # 3. CONFIRM CTE
    confirm_sql = text(f"""
        WITH trades AS (
            SELECT 
                confirm_model_key as model_key,
                confirm_model_version as model_version,
                COALESCE(realized_pnl_usdc, pnl) as pnl,
                closed_at,
                id
            FROM trade_history
            WHERE mode = :mode
              AND position_status = 'CLOSED'
              AND confirm_model_key IS NOT NULL
              AND model_attribution_source IN ('EXACT', 'RECONSTRUCTED')
              AND COALESCE(realized_pnl_usdc, pnl) IS NOT NULL
              {date_filter}
        ),
        zero_trade AS (
            SELECT DISTINCT model_key, model_version, 0 as pnl, '1970-01-01' as closed_at, 0 as id FROM trades
        ),
        all_trades AS (
            SELECT model_key, model_version, pnl, closed_at, id FROM trades
            UNION ALL
            SELECT model_key, model_version, pnl, closed_at, id FROM zero_trade
        ),
        cumulatives AS (
            SELECT 
                model_key, model_version, pnl,
                SUM(pnl) OVER (PARTITION BY model_key, model_version ORDER BY closed_at ASC, id ASC) as equity
            FROM all_trades
        ),
        peaks AS (
            SELECT 
                model_key, model_version, pnl, equity,
                MAX(equity) OVER (PARTITION BY model_key, model_version ORDER BY closed_at ASC, id ASC) as running_peak
            FROM cumulatives
        ),
        drawdowns AS (
            SELECT 
                model_key, model_version, pnl, equity, running_peak,
                (equity - running_peak) as drawdown
            FROM peaks
        ),
        agg_trades AS (
            SELECT 
                model_key, model_version,
                COUNT(*) as total_trades,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_count,
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)) as gross_loss
            FROM trades
            GROUP BY model_key, model_version
        ),
        agg_drawdowns AS (
            SELECT 
                model_key, model_version,
                MIN(drawdown) as max_drawdown
            FROM drawdowns
            GROUP BY model_key, model_version
        )
        SELECT 
            t.model_key, t.model_version,
            t.total_trades, t.total_pnl, t.win_count, t.gross_profit, t.gross_loss,
            d.max_drawdown
        FROM agg_trades t
        LEFT JOIN agg_drawdowns d ON t.model_key = d.model_key AND t.model_version = d.model_version
    """)
    confirm_rows = (await db.execute(confirm_sql, params)).fetchall()
    confirm_stats = {(row.model_key, row.model_version): row for row in confirm_rows}

    # 3.5 VETO CTE
    veto_sql = text("""
        SELECT 
            confirm_model_key as model_key,
            confirm_model_version as model_version,
            SUM(CASE WHEN confirm_passed = true THEN 1 ELSE 0 END) as confirm_passed_count,
            SUM(CASE WHEN confirm_passed = false THEN 1 ELSE 0 END) as veto_count
        FROM decision_funnel_log
        WHERE confirm_model_key IS NOT NULL
          AND execution_mode = :mode
        GROUP BY confirm_model_key, confirm_model_version
    """)
    veto_rows = (await db.execute(veto_sql, {"mode": requested_mode})).fetchall()
    veto_stats = {(row.model_key, row.model_version): row for row in veto_rows}

    # 4. Считаем метрики для каждой версии в реестре
    result = {}
    for m in models:
        key = f"{m.asset}_v{m.version}"
        p = primary_stats.get((m.asset, m.version))
        c = confirm_stats.get((m.asset, m.version))
        v = veto_stats.get((m.asset, m.version))

        def calc_pf(profit, loss):
            if not profit and not loss: return 0.0
            if not loss: return None # Infinity equivalent when there's profit but no losses
            return round(float(profit) / float(loss), 2)

        metrics = {
            "pnl": round(float(p.total_pnl), 4) if p else 0.0,
            "win_rate": round(float(p.win_count) / float(p.total_trades) * 100, 1) if p and p.total_trades > 0 else None,
            "total_trades": int(p.total_trades) if p else 0,
            "max_drawdown": round(float(p.max_drawdown), 4) if p and p.max_drawdown is not None else 0.0,
            "profit_factor": calc_pf(p.gross_profit, p.gross_loss) if p else 0.0,
            
            "confirmed_pnl": round(float(c.total_pnl), 4) if c else 0.0,
            "confirmed_win_rate": round(float(c.win_count) / float(c.total_trades) * 100, 1) if c and c.total_trades > 0 else None,
            "confirmed_trades": int(c.total_trades) if c else 0,
            "confirmed_max_drawdown": round(float(c.max_drawdown), 4) if c and c.max_drawdown is not None else 0.0,
            "confirmed_profit_factor": calc_pf(c.gross_profit, c.gross_loss) if c else 0.0,
            
            "confirm_passed_count": int(v.confirm_passed_count) if v else 0,
            "veto_count": int(v.veto_count) if v else 0,
        }
        
        metrics.update({
            "auc_lift": round(m.accuracy - 0.5, 4) if m.accuracy else None,
            "precision": round(m.precision_at_threshold, 4) if getattr(m, 'precision_at_threshold', None) is not None else None,
            "recall": round(m.recall_at_threshold, 4) if getattr(m, 'recall_at_threshold', None) is not None else None,
            "f1": round(m.f1_at_threshold, 4) if getattr(m, 'f1_at_threshold', None) is not None else None,
            "brier_score": round(m.brier_score, 4) if getattr(m, 'brier_score', None) is not None else None,
        })

        result[key] = metrics

    _cache[cache_key] = {"ts": now, "data": result}
    return result


@router.post(
    "/api/models/{asset}/activate/{version}", dependencies=[Depends(verify_api_key)]
)
async def activate_crypto_model(
    asset: str, version: int, db: AsyncSession = Depends(get_db_session)
):
    """Активирует указанную версию крипто-модели, деактивируя остальные."""
    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    if asset not in allowed_assets:
        raise HTTPException(status_code=404, detail=f"Актив {asset} не найден")

    # Деактивировать все версии этого актива
    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset)
        .values(is_active=False)
    )
    # Активировать нужную версию
    result = await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset, ModelRegistry.version == version)
        .values(is_active=True)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Версия {version} не найдена")

    await db.commit()
    _cache.clear()  # сбросить весь кэш
    return {"status": "success", "asset": asset, "version": version}


@router.delete("/api/models/{asset}/{version}", dependencies=[Depends(verify_api_key)])
async def delete_crypto_model(
    asset: str, version: int, db: AsyncSession = Depends(get_db_session)
):
    """Удаляет указанную версию крипто-модели из БД."""
    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    if asset not in allowed_assets:
        raise HTTPException(status_code=404, detail=f"Актив {asset} не найден")

    # Check if active
    stmt = select(ModelRegistry).where(
        ModelRegistry.asset == asset, ModelRegistry.version == version
    )
    model = (await db.execute(stmt)).scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=404, detail=f"Модель {asset} v{version} не найдена"
        )

    if model.is_active:
        raise HTTPException(
            status_code=400,
            detail="Нельзя удалить активную модель. Сначала активируйте другую.",
        )

    # Delete
    del_stmt = delete(ModelRegistry).where(
        ModelRegistry.asset == asset, ModelRegistry.version == version
    )
    await db.execute(del_stmt)
    await db.commit()

    _cache.clear()
    return {"status": "success", "detail": f"Модель {asset} v{version} удалена"}
