# polyflip/api/crypto_dashboard.py
"""
Дашборд для крипто-домена (LightGBM Up/Down).
Полностью изолирован от Polymarket-дашборда.
Подключается в main.py одной строкой.

"""
from __future__ import annotations

import asyncio
import os
import json
import time
from datetime import datetime, timezone
import numpy as np

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Query, HTTPException
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, update, delete
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

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/crypto", tags=["Crypto"])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "XRPUSDT", "SOLUSDT"]

# Кэш
_cache: dict = {}
_CACHE_TTL = 10  # снизим до 10 секунд для лучшей отзывчивости настроек

@router.get("")
async def crypto_page(request: Request):
    """HTML-страница крипто-дашборда."""
    from polyflip.settings_registry import registry_defaults
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
                "min_child_samples": int(defs.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20")),
                "subsample": float(defs.get("CRYPTO_LGBM_SUBSAMPLE", "0.8")),
                "colsample_bytree": float(defs.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8")),
                "reg_alpha": float(defs.get("CRYPTO_LGBM_REG_ALPHA", "0.1")),
                "reg_lambda": float(defs.get("CRYPTO_LGBM_REG_LAMBDA", "1.0")),
                "min_edge": float(defs.get("BACKTEST_MIN_EDGE", "0.04")),
            }
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
        return _cache["status"]["data"]

    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_high_vol", s])

    stmt = select(ModelRegistry).where(
        ModelRegistry.asset.in_(allowed_assets),
    ).order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    rows = (await db.execute(stmt)).scalars().all()

    # Пороги из RuntimeSettings
    thr_keys = [f"CRYPTO_THRESHOLD_{a}" for a in allowed_assets]
    thr_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(thr_keys))
    thr_rows = (await db.execute(thr_stmt)).scalars().all()
    thresholds = {r.key.replace("CRYPTO_THRESHOLD_", ""): float(r.value) for r in thr_rows}

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
        "CRYPTO_LGBM_N_ESTIMATORS", "CRYPTO_LGBM_LEARNING_RATE", "CRYPTO_LGBM_NUM_LEAVES",
        "CRYPTO_LGBM_MAX_DEPTH", "CRYPTO_LGBM_MIN_CHILD_SAMPLES", "CRYPTO_LGBM_SUBSAMPLE",
        "CRYPTO_LGBM_COLSAMPLE_BYTREE", "CRYPTO_LGBM_REG_ALPHA", "CRYPTO_LGBM_REG_LAMBDA",
        "CRYPTO_BACKTEST_MIN_EDGE"
    ]
    set_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    set_rows = (await db.execute(set_stmt)).scalars().all()
    db_settings = {r.key: r.value for r in set_rows}

    defs = registry_defaults()
    active_settings = {
        "n_estimators": int(db_settings.get("CRYPTO_LGBM_N_ESTIMATORS", defs.get("CRYPTO_LGBM_N_ESTIMATORS", "300"))),
        "learning_rate": float(db_settings.get("CRYPTO_LGBM_LEARNING_RATE", defs.get("CRYPTO_LGBM_LEARNING_RATE", "0.05"))),
        "num_leaves": int(db_settings.get("CRYPTO_LGBM_NUM_LEAVES", defs.get("CRYPTO_LGBM_NUM_LEAVES", "31"))),
        "max_depth": int(db_settings.get("CRYPTO_LGBM_MAX_DEPTH", defs.get("CRYPTO_LGBM_MAX_DEPTH", "5"))),
        "min_child_samples": int(db_settings.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", defs.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20"))),
        "subsample": float(db_settings.get("CRYPTO_LGBM_SUBSAMPLE", defs.get("CRYPTO_LGBM_SUBSAMPLE", "0.8"))),
        "colsample_bytree": float(db_settings.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", defs.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8"))),
        "reg_alpha": float(db_settings.get("CRYPTO_LGBM_REG_ALPHA", defs.get("CRYPTO_LGBM_REG_ALPHA", "0.1"))),
        "reg_lambda": float(db_settings.get("CRYPTO_LGBM_REG_LAMBDA", defs.get("CRYPTO_LGBM_REG_LAMBDA", "1.0"))),
        "min_edge": float(db_settings.get("CRYPTO_BACKTEST_MIN_EDGE", defs.get("BACKTEST_MIN_EDGE", "0.04"))),
    }

    models_info = {}
    for m in rows:
        key = f"{m.asset}_v{m.version}"
        models_info[key] = {
            "asset":      m.asset,
            "version":    m.version,
            "is_active":  m.is_active,
            "auc":        round(m.accuracy, 4),
            "baseline":   round(m.baseline, 4),
            "ece":        round(m.ece, 4) if getattr(m, 'ece', None) else None,
            "threshold":  thresholds.get(m.asset),
            "features":   m.features.split(",") if getattr(m, 'features', None) else [],
            "trained_at": m.trained_at.isoformat() if getattr(m, 'trained_at', None) else None,
            "feature_importance": feature_importances.get(m.asset, {}),
        }

    result = {
        "models": models_info,
        "symbols": CRYPTO_SYMBOLS,
        "settings": active_settings,
        "feature_importances": {
            asset: feature_importances.get(asset, {})
            for asset in set(m.asset for m in rows if m.is_active)
        }
    }
    _cache["status"] = {"ts": now, "data": result}
    return result

@router.post("/api/settings", dependencies=[Depends(verify_api_key)])
async def save_crypto_settings(
    settings: dict,
    db: AsyncSession = Depends(get_db_session)
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
        "min_edge": "CRYPTO_BACKTEST_MIN_EDGE",
    }

    for key, db_key in keys_map.items():
        if key in settings:
            val_str = str(settings[key])
            row = (await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == db_key))).scalar_one_or_none()
            if row:
                row.value = val_str
                row.updated_at = now
                row.updated_by = "crypto_dashboard_ui"
            else:
                db.add(RuntimeSettings(
                    key=db_key,
                    value=val_str,
                    updated_at=now,
                    updated_by="crypto_dashboard_ui"
                ))

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

    async with async_session() as session:
        # Пытаемся получить настройки из БД для дефолта
        async def _get_rt(key: str, default: float) -> float:
            row = (await session.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == key)
            )).scalar_one_or_none()
            return float(row.value) if row else default

        defs = registry_defaults()
        if min_edge is None:
            min_edge = await _get_rt("CRYPTO_BACKTEST_MIN_EDGE", float(defs.get("BACKTEST_MIN_EDGE", "0.04")))

        lgbm_params = {
            "subsample":        await _get_rt("CRYPTO_LGBM_SUBSAMPLE", float(defs.get("CRYPTO_LGBM_SUBSAMPLE", "0.8"))),
            "colsample_bytree": await _get_rt("CRYPTO_LGBM_COLSAMPLE_BYTREE", float(defs.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", "0.8"))),
            "num_leaves":       int(await _get_rt("CRYPTO_LGBM_NUM_LEAVES", float(defs.get("CRYPTO_LGBM_NUM_LEAVES", "31")))),
            "max_depth":        int(await _get_rt("CRYPTO_LGBM_MAX_DEPTH", float(defs.get("CRYPTO_LGBM_MAX_DEPTH", "5")))),
            "min_child_samples":int(await _get_rt("CRYPTO_LGBM_MIN_CHILD_SAMPLES", float(defs.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", "20")))),
            "n_estimators":     int(await _get_rt("CRYPTO_LGBM_N_ESTIMATORS", float(defs.get("CRYPTO_LGBM_N_ESTIMATORS", "300")))),
            "reg_alpha":        await _get_rt("CRYPTO_LGBM_REG_ALPHA", float(defs.get("CRYPTO_LGBM_REG_ALPHA", "0.1"))),
            "reg_lambda":       await _get_rt("CRYPTO_LGBM_REG_LAMBDA", float(defs.get("CRYPTO_LGBM_REG_LAMBDA", "1.0"))),
        }

        candles = await get_recent_candles(session, symbol, interval, limit=10_000)

    if len(candles) < 600:
        return {"error": f"Недостаточно свечей: {len(candles)} < 600. Пожалуйста, сделайте backfill.", "symbol": symbol}

    df = build_features(candles)
    
    # Запускаем backtest в пуле потоков (CPU-bound)
    result = await asyncio.to_thread(
        run_backtest, df, symbol, min_edge, commission, 
        lgbm_params=lgbm_params
    )

    data = {
        "symbol":           result.symbol,
        "n_candles_total":  result.n_candles_total,
        "n_candles_test":   result.n_candles_test,
        "n_trades":         result.n_trades,
        "win_rate":         round(result.win_rate, 4),
        "total_return_net": round(result.total_return_net, 5),
        "sharpe_ratio":     round(result.sharpe_ratio, 3),
        "max_drawdown":     round(result.max_drawdown, 5),
        "edge_rate":        round(result.edge_rate, 4),
        "epsilon":          round(result.epsilon, 6),
        "train_auc":        round(result.train_auc, 4),
        "is_profitable":    result.is_profitable(),
        "summary":          result.summary(),
        "pnl_curve":        result.pnl_curve
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
    async def _train():
        async with async_session() as session:
            trainer = CryptoModelTrainer(session)
            ok = await trainer.train(symbol, interval)
            # Сбрасываем кэши
            _cache.pop("status", None)
            for k in list(_cache.keys()):
                if k.startswith(f"backtest_{symbol}"):
                    _cache.pop(k, None)
            logger.info("crypto_retrain_done", symbol=symbol, success=ok)

    background_tasks.add_task(_train)
    return {
        "status":  "started",
        "symbol":  symbol,
        "message": f"Переобучение {symbol} запущено в фоне. Процесс займет около 15–30 секунд. Вы можете отслеживать версию на вкладке Модели.",
    }

@router.get("/api/model_pnl", dependencies=[Depends(verify_api_key)])
async def crypto_model_pnl(db: AsyncSession = Depends(get_db_session)):
    cache_key = "crypto_model_pnl"
    now = time.time()
    if cache_key in _cache:
        c = _cache[cache_key]
        if now - c["ts"] < 30:
            return c["data"]

    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_high_vol", s])

    # 1. Запрашиваем модели
    stmt = select(ModelRegistry).where(
        ModelRegistry.asset.in_(allowed_assets),
    ).order_by(ModelRegistry.asset, ModelRegistry.version.desc())
    models = (await db.execute(stmt)).scalars().all()

    # 2. Запрашиваем все успешные сделки
    # Fetch crypto trades. DB stores asset as 'BTC', 'ETH' etc.
    db_assets = [s.replace("USDT", "") for s in CRYPTO_SYMBOLS]
    
    trades_stmt = select(
        TradeHistory.asset,
        TradeHistory.pnl,
        TradeHistory.created_at,
    ).where(
        TradeHistory.status == "SUCCESS",
        TradeHistory.pnl.is_not(None),
        TradeHistory.asset.in_(db_assets),
    )
    trades = (await db.execute(trades_stmt)).all()

    # 3. Группируем сделки по asset + времени
    asset_trades: dict[str, list] = defaultdict(list)
    for row in trades:
        # row.asset is 'BTC', map to 'BTCUSDT_low_vol', 'BTCUSDT_high_vol', etc.
        base = row.asset
        asset_trades[f"{base}USDT_low_vol"].append((row.created_at, row.pnl))
        asset_trades[f"{base}USDT_high_vol"].append((row.created_at, row.pnl))
        asset_trades[f"{base}USDT"].append((row.created_at, row.pnl))

    # Группируем модели по asset
    asset_versions: dict[str, list] = defaultdict(list)
    for m in models:
        asset_versions[m.asset].append(m)

    # 4. Считаем метрики для каждой версии
    result = {}
    for asset, versions in asset_versions.items():
        versions_asc = list(reversed(versions))
        
        for idx, m in enumerate(versions_asc):
            since = m.trained_at
            until = versions_asc[idx + 1].trained_at if idx + 1 < len(versions_asc) else None
            
            pnls = [
                pnl for (ts, pnl) in asset_trades[asset]
                if (since is None or ts >= since)
                and (until is None or ts < until)
            ]
            
            key = f"{asset}_v{m.version}"
            if pnls:
                total = sum(pnls)
                wins = sum(1 for p in pnls if p > 0)
                result[key] = {
                    "pnl": round(total, 4),
                    "win_rate": round(wins / len(pnls) * 100, 1),
                    "total_trades": len(pnls),
                }
            else:
                result[key] = {
                    "pnl": 0.0,
                    "win_rate": None,
                    "total_trades": 0,
                }

    _cache[cache_key] = {"ts": now, "data": result}
    return result

@router.post("/api/models/{asset}/activate/{version}", dependencies=[Depends(verify_api_key)])
async def activate_crypto_model(
    asset: str,
    version: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Активирует указанную версию крипто-модели, деактивируя остальные."""
    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_high_vol", s])
    
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
    asset: str,
    version: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Удаляет указанную версию крипто-модели из БД."""
    allowed_assets = []
    for s in CRYPTO_SYMBOLS:
        allowed_assets.extend([f"{s}_low_vol", f"{s}_high_vol", s])
        
    if asset not in allowed_assets:
        raise HTTPException(status_code=404, detail=f"Актив {asset} не найден")
        
    # Check if active
    stmt = select(ModelRegistry).where(ModelRegistry.asset == asset, ModelRegistry.version == version)
    model = (await db.execute(stmt)).scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Модель {asset} v{version} не найдена")
        
    if model.is_active:
        raise HTTPException(status_code=400, detail="Нельзя удалить активную модель. Сначала активируйте другую.")
        
    # Delete
    del_stmt = delete(ModelRegistry).where(ModelRegistry.asset == asset, ModelRegistry.version == version)
    await db.execute(del_stmt)
    await db.commit()
    
    _cache.clear()
    return {"status": "success", "detail": f"Модель {asset} v{version} удалена"}

