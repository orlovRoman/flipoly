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
from fastapi import APIRouter, BackgroundTasks, Body, Depends, Request, Query, HTTPException
from pydantic import BaseModel
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
from polyflip.crypto.predictor import CryptoPredictor
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
def round_optional(value, digits=4):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None

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
        select(
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.is_active,
            ModelRegistry.accuracy,
            ModelRegistry.baseline,
            ModelRegistry.ece,
            ModelRegistry.features,
            ModelRegistry.trained_at,
            ModelRegistry.quality_gate_passed,
            ModelRegistry.quality_gate_reasons,
            ModelRegistry.activation_source,
            ModelRegistry.quality_override,
            ModelRegistry.activated_at,
            ModelRegistry.activation_reason,
            ModelRegistry.precision_at_threshold,
            ModelRegistry.recall_at_threshold,
            ModelRegistry.f1_at_threshold,
            ModelRegistry.brier_score,
        )
        .where(ModelRegistry.asset.in_(allowed_assets))
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    rows = (await db.execute(stmt)).all()

    # Пороги из RuntimeSettings
    thr_keys = [f"CRYPTO_THRESHOLD_{a}" for a in allowed_assets]
    thr_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(thr_keys))
    thr_rows = (await db.execute(thr_stmt)).scalars().all()
    
    thresholds = {}
    for row in thr_rows:
        asset = row.key.replace("CRYPTO_THRESHOLD_", "")
        try:
            thresholds[asset] = float(row.value)
        except (TypeError, ValueError):
            logger.warning(
                "invalid_crypto_threshold",
                key=row.key,
                value=row.value,
            )
            thresholds[asset] = None

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

    def _safe_int(key, default):
        val = db_settings.get(key, defs.get(key, default))
        if val == "None" or val is None:
            return int(default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return int(default)

    def _safe_float(key, default):
        val = db_settings.get(key, defs.get(key, default))
        if val == "None" or val is None:
            return float(default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    active_settings = {
        "n_estimators": _safe_int("CRYPTO_LGBM_N_ESTIMATORS", "300"),
        "learning_rate": _safe_float("CRYPTO_LGBM_LEARNING_RATE", "0.05"),
        "num_leaves": _safe_int("CRYPTO_LGBM_NUM_LEAVES", "31"),
        "max_depth": _safe_int("CRYPTO_LGBM_MAX_DEPTH", "5"),
        "min_child_samples": _safe_int("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20"),
        "subsample": _safe_float("CRYPTO_LGBM_SUBSAMPLE", "0.8"),
        "colsample_bytree": _safe_float("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8"),
        "reg_alpha": _safe_float("CRYPTO_LGBM_REG_ALPHA", "0.1"),
        "reg_lambda": _safe_float("CRYPTO_LGBM_REG_LAMBDA", "1.0"),
        "min_edge": _safe_float("BACKTEST_MIN_EDGE", "0.04"),
        "epsilon_quantile": _safe_float("LGBM_EPSILON_QUANTILE", "0.6"),
    }

    models_info = {}
    for m in rows:
        key = f"{m.asset}_v{m.version}"
        try:
            models_info[key] = {
                "asset": m.asset,
                "version": m.version,
                "is_active": m.is_active,
                "auc": round_optional(m.accuracy),
                "baseline": round_optional(m.baseline),
                "ece": round_optional(m.ece) if getattr(m, "ece", None) else None,
                "threshold": thresholds.get(m.asset),
                "features": m.features.split(",") if getattr(m, "features", None) else [],
                "trained_at": (
                    m.trained_at.isoformat() if getattr(m, "trained_at", None) else None
                ),
                "feature_importance": feature_importances.get(m.asset, {}),
                # Аудит Quality Gate и активации
                "quality_gate_passed": m.quality_gate_passed,
                "quality_gate_reasons": m.quality_gate_reasons,
                "activation_source": m.activation_source,
                "quality_override": getattr(m, "quality_override", None),
                "activated_at": (
                    m.activated_at.isoformat() if getattr(m, "activated_at", None) else None
                ),
                "activation_reason": getattr(m, "activation_reason", None),
                # Precision / Recall / F1 / Brier из реестра
                "precision": round_optional(m.precision_at_threshold) if getattr(m, "precision_at_threshold", None) is not None else None,
                "recall": round_optional(m.recall_at_threshold) if getattr(m, "recall_at_threshold", None) is not None else None,
                "f1": round_optional(m.f1_at_threshold) if getattr(m, "f1_at_threshold", None) is not None else None,
                "brier_score": round_optional(m.brier_score) if getattr(m, "brier_score", None) is not None else None,
            }
        except Exception as e:
            logger.error("crypto_status_model_parse_error", key=key, error=str(e))

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



@router.get("/api/models/coverage", dependencies=[Depends(verify_api_key)])
async def crypto_models_coverage(db: AsyncSession = Depends(get_db_session)):
    """
    P1: Таблица покрытия LightGBM-моделей по режимам волатильности.
    Показывает для каждого (актив × режим): активная версия или None.
    Используется для диагностики REGIME_UNAVAILABLE / MODEL_NOT_LOADED.
    """
    cache_key = "crypto_models_coverage"
    now = time.time()
    if cache_key in _cache and now - _cache[cache_key]["ts"] < 60:
        return _cache[cache_key]["data"]

    regimes = ["low_vol", "mid_vol", "high_vol"]
    result = {}

    allowed_assets = [f"{sym.upper()}_{regime}" for sym in CRYPTO_SYMBOLS for regime in regimes]
    stmt = (
        select(
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.is_active,
            ModelRegistry.trained_at,
            ModelRegistry.quality_gate_passed,
            ModelRegistry.activation_source,
        )
        .where(ModelRegistry.asset.in_(allowed_assets))
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    all_rows = (await db.execute(stmt)).all()

    asset_groups = {}
    for r in all_rows:
        asset_groups.setdefault(r.asset, []).append(r)

    for sym in CRYPTO_SYMBOLS:
        sym_upper = sym.upper()
        result[sym_upper] = {}
        for regime in regimes:
            asset_key = f"{sym_upper}_{regime}"
            rows = asset_groups.get(asset_key, [])[:3]
            
            active_row = next((r for r in rows if r.is_active), None)
            all_versions = [r.version for r in rows]

            result[sym_upper][regime] = {
                "active_version": active_row.version if active_row else None,
                "active_trained_at": (
                    active_row.trained_at.isoformat() if active_row and active_row.trained_at else None
                ),
                "quality_gate_passed": active_row.quality_gate_passed if active_row else None,
                "activation_source": active_row.activation_source if active_row else None,
                "recent_versions": all_versions,
                "status": (
                    "ACTIVE" if active_row else
                    ("HAS_INACTIVE" if all_versions else "MISSING")
                ),
            }

    data = {"coverage": result, "regimes": regimes, "symbols": list(CRYPTO_SYMBOLS)}
    _cache[cache_key] = {"data": data, "ts": now}
    return data


@router.get("/api/models/analytics", dependencies=[Depends(verify_api_key)])
async def crypto_models_analytics(
    requested_mode: str = "PAPER",
    date_from: str = None,
    date_to: str = None,
    db: AsyncSession = Depends(get_db_session)
):
    if requested_mode not in {"PAPER", "LIVE", "SHADOW"}:
        raise HTTPException(
            status_code=422,
            detail="requested_mode должен быть PAPER, SHADOW или LIVE",
        )

    # ── Валидация дат (баг #2) ──────────────────────────────────────────────
    from datetime import timedelta
    import re as _re

    def _parse_date(s: str, field: str) -> datetime:
        if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise HTTPException(status_code=422, detail=f"{field}: ожидается формат YYYY-MM-DD")
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=422, detail=f"{field}: несуществующая дата '{s}'")

    dt_from = _parse_date(date_from, "date_from") if date_from else None
    dt_to   = _parse_date(date_to,   "date_to")   if date_to   else None
    if dt_from and dt_to and dt_from > dt_to:
        raise HTTPException(status_code=422, detail="date_from не может быть позже date_to")
    # ────────────────────────────────────────────────────────────────────────

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
        select(
            ModelRegistry.asset,
            ModelRegistry.version,
            ModelRegistry.accuracy,
            ModelRegistry.precision_at_threshold,
            ModelRegistry.recall_at_threshold,
            ModelRegistry.f1_at_threshold,
            ModelRegistry.brier_score,
            ModelRegistry.activation_source,
            ModelRegistry.quality_gate_passed,
            ModelRegistry.quality_override,
        )
        .where(ModelRegistry.asset.in_(allowed_assets))
        .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    )
    models = (await db.execute(stmt)).all()

    # Фильтры дат
    params = {"mode": requested_mode}
    date_filter = ""
    veto_date_filter = ""

    if dt_from:
        params["date_from"] = dt_from
        date_filter     += " AND created_at >= :date_from"
        veto_date_filter += " AND created_at >= :date_from"

    if dt_to:
        dt_to_exclusive = dt_to + timedelta(days=1)
        params["date_to_exclusive"] = dt_to_exclusive
        date_filter     += " AND created_at < :date_to_exclusive"
        veto_date_filter += " AND created_at < :date_to_exclusive"

    # 2. PRIMARY CTE
    primary_sql = text(f"""
        WITH trades AS (
            SELECT 
                model_key,
                model_version,
                COALESCE(realized_pnl_usdc, pnl) as pnl,
                created_at,
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
            SELECT DISTINCT model_key, model_version, 0 as pnl, '1970-01-01' as created_at, 0 as id FROM trades
        ),
        all_trades AS (
            SELECT model_key, model_version, pnl, created_at, id FROM trades
            UNION ALL
            SELECT model_key, model_version, pnl, created_at, id FROM zero_trade
        ),
        cumulatives AS (
            SELECT 
                model_key, model_version, pnl, created_at, id,
                SUM(pnl) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as equity
            FROM all_trades
        ),
        peaks AS (
            SELECT 
                model_key, model_version, pnl, equity,
                MAX(equity) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as running_peak
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
    
    primary_rows = (await db.execute(primary_sql, params)).fetchall()
    primary_stats = {(row.model_key, row.model_version): row for row in primary_rows}

    # 3. DIRECTION CTE
    direction_sql = text(f"""
        WITH trades AS (
            SELECT 
                COALESCE(direction_model_key, confirm_model_key) as model_key,
                COALESCE(direction_model_version, confirm_model_version) as model_version,
                COALESCE(realized_pnl_usdc, pnl) as pnl,
                created_at,
                id
            FROM trade_history
            WHERE mode = :mode
              AND position_status = 'CLOSED'
              AND COALESCE(direction_model_key, confirm_model_key) IS NOT NULL
              AND COALESCE(realized_pnl_usdc, pnl) IS NOT NULL
              {date_filter}
        ),
        unique_trades AS (
            SELECT DISTINCT id, model_key, model_version, pnl, created_at FROM trades
        ),
        zero_trade AS (
            SELECT DISTINCT model_key, model_version, 0 as pnl, '1970-01-01' as created_at, 0 as id FROM trades
        ),
        all_trades AS (
            SELECT model_key, model_version, pnl, created_at, id FROM unique_trades
            UNION ALL
            SELECT model_key, model_version, pnl, created_at, id FROM zero_trade
        ),
        cumulatives AS (
            SELECT 
                model_key, model_version, pnl, created_at, id,
                SUM(pnl) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as equity
            FROM all_trades
        ),
        peaks AS (
            SELECT 
                model_key, model_version, pnl, equity,
                MAX(equity) OVER (PARTITION BY model_key, model_version ORDER BY created_at ASC, id ASC) as running_peak
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
            FROM unique_trades
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
    direction_rows = (await db.execute(direction_sql, params)).fetchall()
    direction_stats = {(row.model_key, row.model_version): row for row in direction_rows}

    # 3.5 VETO CTE
    # Используем COUNT(DISTINCT decision_run_id) для дедупликации исторических записей:
    # старые строки могли иметь дубли ML + COMBINED для одного цикла решения (баг #3)
    veto_params = dict(params)
    veto_params["decision_mode"] = requested_mode

    veto_sql = text(f"""
        SELECT
            COALESCE(direction_model_key, confirm_model_key) as model_key,
            COALESCE(direction_model_version, confirm_model_version) as model_version,
            COUNT(DISTINCT CASE WHEN confirm_passed = true  THEN decision_run_id END) as confirm_passed_count,
            COUNT(DISTINCT CASE WHEN confirm_passed = false THEN decision_run_id END) as veto_count
        FROM decision_funnel_log
        WHERE COALESCE(direction_model_key, confirm_model_key) IS NOT NULL
          AND execution_mode = :decision_mode
          AND decision_run_id IS NOT NULL
          {veto_date_filter}
        GROUP BY
            COALESCE(direction_model_key, confirm_model_key),
            COALESCE(direction_model_version, confirm_model_version)
    """)
    veto_rows = (await db.execute(veto_sql, veto_params)).fetchall()
    veto_stats = {(row.model_key, row.model_version): row for row in veto_rows}

    # 4. Считаем метрики для каждой версии в реестре
    result = {}
    for m in models:
        key = f"{m.asset}_v{m.version}"
        p = primary_stats.get((m.asset, m.version))
        c = direction_stats.get((m.asset, m.version))
        v = veto_stats.get((m.asset, m.version))

        def calc_pf(profit, loss):
            if not profit and not loss: return 0.0
            if not loss: return None  # Infinity: прибыль есть, убытков нет
            return round(float(profit) / float(loss), 2)

        metrics = {
            "pnl": round(float(p.total_pnl), 4) if p else 0.0,
            "win_rate": round(float(p.win_count) / float(p.total_trades) * 100, 1) if p and p.total_trades > 0 else None,
            "total_trades": int(p.total_trades) if p else 0,
            "max_drawdown": round(float(p.max_drawdown), 4) if p and p.max_drawdown is not None else 0.0,
            "profit_factor": calc_pf(p.gross_profit, p.gross_loss) if p else 0.0,

            "direction_pnl": round(float(c.total_pnl), 4) if c else 0.0,
            "direction_win_rate": round(float(c.win_count) / float(c.total_trades) * 100, 1) if c and c.total_trades > 0 else None,
            "direction_trades": int(c.total_trades) if c else 0,
            "direction_max_drawdown": round(float(c.max_drawdown), 4) if c and c.max_drawdown is not None else 0.0,
            "direction_profit_factor": calc_pf(c.gross_profit, c.gross_loss) if c else 0.0,

            "confirm_passed_count": int(v.confirm_passed_count) if v else 0,
            "veto_count": int(v.veto_count) if v else 0,

            # Метрики качества из ModelRegistry
            "auc_lift": round(m.accuracy - 0.5, 4) if m.accuracy else None,
            "precision": round(m.precision_at_threshold, 4) if getattr(m, "precision_at_threshold", None) is not None else None,
            "recall": round(m.recall_at_threshold, 4) if getattr(m, "recall_at_threshold", None) is not None else None,
            "f1": round(m.f1_at_threshold, 4) if getattr(m, "f1_at_threshold", None) is not None else None,
            "brier_score": round(m.brier_score, 4) if getattr(m, "brier_score", None) is not None else None,

            # Аудит активации
            "activation_source": m.activation_source,
            "quality_gate_passed": m.quality_gate_passed,
            "quality_override": getattr(m, "quality_override", None),
        }

        result[key] = metrics

    _cache[cache_key] = {"ts": now, "data": result}
    return result


class ActivateModelRequest(BaseModel):
    force: bool = False
    reason: str | None = None


@router.post(
    "/api/models/{asset}/activate/{version}", dependencies=[Depends(verify_api_key)]
)
async def activate_crypto_model(
    asset: str,
    version: int,
    payload: ActivateModelRequest = Body(default_factory=ActivateModelRequest),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Активирует указанную версию крипто-модели, деактивируя остальные.

    Если модель не прошла Quality Gate и force=False → HTTP 409.
    Если force=True → активация помечается как MANUAL.
    """
    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_mid_vol", f"{s}_high_vol", s])

    if asset not in allowed_assets:
        raise HTTPException(status_code=404, detail=f"Актив {asset} не найден")

    # 1. Загружаем целевую модель
    target_stmt = select(ModelRegistry).where(
        ModelRegistry.asset == asset,
        ModelRegistry.version == version,
    )
    model = (await db.execute(target_stmt)).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail=f"Версия {version} не найдена")

    # 1.5 Smoke Test: Проверка совместимости формата признаков
    if model.model_blob:
        import pickle
        import numpy as np
        from polyflip.crypto.trainer import CRYPTO_FEATURES
        try:
            clf = pickle.loads(model.model_blob)
            fv_array = np.zeros(len(CRYPTO_FEATURES), dtype=np.float64)
            clf.predict_proba([fv_array])
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Smoke Test Failed: Модель несовместима с текущим форматом признаков ({len(CRYPTO_FEATURES)}). "
                       f"Ошибка инференса: {str(e)}. Необходимо переобучить модель.",
            )

    # 2. Quality Gate check — только если поле явно False (None = legacy, не блокируем)
    if model.quality_gate_passed is False and not payload.force:
        reasons = model.quality_gate_reasons or {}
        raise HTTPException(
            status_code=409,
            detail={
                "code": "QUALITY_GATE_OVERRIDE_REQUIRED",
                "message": f"Модель {asset} v{version} не прошла Quality Gate. Передайте force=true для ручной активации.",
                "metrics": {
                    "auc": model.accuracy,
                    "ece": model.ece,
                    "precision": model.precision_at_threshold,
                    "recall": model.recall_at_threshold,
                    "reasons": reasons.get("reasons", []),
                },
            },
        )

    # 3. Запоминаем предыдущую активную версию
    prev_stmt = select(ModelRegistry).where(
        ModelRegistry.asset == asset,
        ModelRegistry.is_active == True,
    )
    prev_model = (await db.execute(prev_stmt)).scalar_one_or_none()
    previous_version = prev_model.version if prev_model else None

    # 4. Атомарная активация
    # Семантика (баг #4): activation_source показывает КТО активировал,
    # quality_override показывает, был ли обойдён Quality Gate
    now = datetime.now(timezone.utc)
    is_quality_override = bool(payload.force or model.quality_gate_passed is False)

    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset)
        .values(is_active=False)
    )
    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.asset == asset, ModelRegistry.version == version)
        .values(
            is_active=True,
            activation_source="DASHBOARD",
            quality_override=is_quality_override,
            activated_at=now,
            activated_by="dashboard",
            activation_reason=payload.reason,
        )
    )
    await db.commit()

    # 5. Инвалидируем кэш предиктора (в рамках того же процесса API)
    symbol = asset.split("_")[0]
    try:
        CryptoPredictor.invalidate_all(symbol)
        logger.info("predictor_cache_invalidated_after_manual_activation", asset=asset, version=version)
    except Exception as exc:
        logger.warning("predictor_invalidate_failed", error=str(exc))

    _cache.clear()

    # 6. Формируем ответ
    response: dict = {
        "status": "success",
        "asset": asset,
        "active_version": version,
        "previous_version": previous_version,
        "activation_source": "DASHBOARD",
        "quality_gate_passed": model.quality_gate_passed,
        "quality_override": is_quality_override,
    }
    if is_quality_override:
        response["warning"] = (
            f"Модель {asset} v{version} активирована через DASHBOARD с обходом Quality Gate."
        )
    return response


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
