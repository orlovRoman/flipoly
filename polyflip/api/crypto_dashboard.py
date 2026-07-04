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

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Query
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.api.auth import verify_api_key
import polyflip.constants as C
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.feature_builder import build_features
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.trainer import CryptoModelTrainer
from polyflip.db.connection import async_session, get_db_session
from polyflip.db.models import ModelRegistry, RuntimeSettings

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/crypto", tags=["Crypto"])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Кэш
_cache: dict = {}
_CACHE_TTL = 10  # снизим до 10 секунд для лучшей отзывчивости настроек


@router.get("")
async def crypto_page(request: Request):
    """HTML-страница крипто-дашборда."""
    # Получаем API-ключ из куки для формы
    api_key = request.cookies.get("api_key", "")
    return templates.TemplateResponse(
        request=request,
        name="crypto.html",
        context={
            "symbols": CRYPTO_SYMBOLS,
            "api_key": api_key,
            "defaults": {
                "n_estimators": C.LGBM_N_ESTIMATORS,
                "learning_rate": C.LGBM_LEARNING_RATE,
                "num_leaves": C.LGBM_NUM_LEAVES,
                "max_depth": C.LGBM_MAX_DEPTH,
                "min_child_samples": C.LGBM_MIN_CHILD_SAMPLES,
                "subsample": C.LGBM_SUBSAMPLE,
                "colsample_bytree": C.LGBM_COLSAMPLE_BYTREE,
                "reg_alpha": C.LGBM_REG_ALPHA,
                "reg_lambda": C.LGBM_REG_LAMBDA,
                "epsilon_quantile": C.CANDLE_EPSILON_QUANTILE,
                "min_edge": C.BACKTEST_MIN_EDGE,
            }
        },
    )


@router.get("/api/crypto/status", dependencies=[Depends(verify_api_key)])
async def crypto_status(db: AsyncSession = Depends(get_db_session)):
    """
    Возвращает текущее состояние крипто-моделей:
    версию, AUC, ECE, порог, список фич, дату обучения, важность фичей и гиперпараметры.
    """
    now = time.time()
    if "status" in _cache and now - _cache["status"]["ts"] < _CACHE_TTL:
        return _cache["status"]["data"]

    stmt = select(ModelRegistry).where(
        ModelRegistry.is_active.is_(True),
        ModelRegistry.asset.in_(CRYPTO_SYMBOLS),
    )
    rows = (await db.execute(stmt)).scalars().all()

    # Пороги из RuntimeSettings
    thr_keys = [f"CRYPTO_THRESHOLD_{s}" for s in CRYPTO_SYMBOLS]
    thr_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(thr_keys))
    thr_rows = (await db.execute(thr_stmt)).scalars().all()
    thresholds = {r.key.replace("CRYPTO_THRESHOLD_", ""): float(r.value) for r in thr_rows}

    # Важность признаков из RuntimeSettings
    fi_keys = [f"CRYPTO_FI_{s}" for s in CRYPTO_SYMBOLS]
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
        "CRYPTO_CANDLE_EPSILON_QUANTILE", "CRYPTO_BACKTEST_MIN_EDGE"
    ]
    set_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    set_rows = (await db.execute(set_stmt)).scalars().all()
    db_settings = {r.key: r.value for r in set_rows}

    active_settings = {
        "n_estimators": int(db_settings.get("CRYPTO_LGBM_N_ESTIMATORS", C.LGBM_N_ESTIMATORS)),
        "learning_rate": float(db_settings.get("CRYPTO_LGBM_LEARNING_RATE", C.LGBM_LEARNING_RATE)),
        "num_leaves": int(db_settings.get("CRYPTO_LGBM_NUM_LEAVES", C.LGBM_NUM_LEAVES)),
        "max_depth": int(db_settings.get("CRYPTO_LGBM_MAX_DEPTH", C.LGBM_MAX_DEPTH)),
        "min_child_samples": int(db_settings.get("CRYPTO_LGBM_MIN_CHILD_SAMPLES", C.LGBM_MIN_CHILD_SAMPLES)),
        "subsample": float(db_settings.get("CRYPTO_LGBM_SUBSAMPLE", C.LGBM_SUBSAMPLE)),
        "colsample_bytree": float(db_settings.get("CRYPTO_LGBM_COLSAMPLE_BYTREE", C.LGBM_COLSAMPLE_BYTREE)),
        "reg_alpha": float(db_settings.get("CRYPTO_LGBM_REG_ALPHA", C.LGBM_REG_ALPHA)),
        "reg_lambda": float(db_settings.get("CRYPTO_LGBM_REG_LAMBDA", C.LGBM_REG_LAMBDA)),
        "epsilon_quantile": float(db_settings.get("CRYPTO_CANDLE_EPSILON_QUANTILE", C.CANDLE_EPSILON_QUANTILE)),
        "min_edge": float(db_settings.get("CRYPTO_BACKTEST_MIN_EDGE", C.BACKTEST_MIN_EDGE)),
    }

    models_info = {}
    for m in rows:
        models_info[m.asset] = {
            "version":    m.version,
            "auc":        round(m.accuracy, 4),
            "baseline":   round(m.baseline, 4),
            "ece":        round(m.ece, 4) if m.ece else None,
            "threshold":  thresholds.get(m.asset),
            "features":   m.features.split(",") if m.features else [],
            "trained_at": m.trained_at.isoformat() if m.trained_at else None,
            "feature_importance": feature_importances.get(m.asset, {}),
        }

    result = {
        "models": models_info,
        "symbols": CRYPTO_SYMBOLS,
        "settings": active_settings
    }
    _cache["status"] = {"ts": now, "data": result}
    return result


@router.post("/api/crypto/settings", dependencies=[Depends(verify_api_key)])
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
        "epsilon_quantile": "CRYPTO_CANDLE_EPSILON_QUANTILE",
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


@router.get("/api/crypto/backtest", dependencies=[Depends(verify_api_key)])
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
    # Если параметры переданы явно, переопределяем их в константах
    old_min_edge = C.BACKTEST_MIN_EDGE
    old_commission = C.BACKTEST_COMMISSION

    if min_edge is not None:
        C.BACKTEST_MIN_EDGE = min_edge
    if commission is not None:
        C.BACKTEST_COMMISSION = commission

    cache_key = f"backtest_{symbol}_{interval}_{min_edge}_{commission}"
    now = time.time()
    if cache_key in _cache and now - _cache[cache_key]["ts"] < 300:
        # Восстанавливаем старые настройки перед выходом
        C.BACKTEST_MIN_EDGE = old_min_edge
        C.BACKTEST_COMMISSION = old_commission
        return _cache[cache_key]["data"]

    async with async_session() as session:
        # Пытаемся получить настройки из БД для дефолта
        if min_edge is None:
            thr_row = (await session.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == "CRYPTO_BACKTEST_MIN_EDGE")
            )).scalar_one_or_none()
            if thr_row:
                C.BACKTEST_MIN_EDGE = float(thr_row.value)

        candles = await get_recent_candles(session, symbol, interval, limit=10_000)

    if len(candles) < 600:
        C.BACKTEST_MIN_EDGE = old_min_edge
        C.BACKTEST_COMMISSION = old_commission
        return {"error": f"Недостаточно свечей: {len(candles)} < 600. Пожалуйста, сделайте backfill.", "symbol": symbol}

    df = build_features(candles)
    
    # Запускаем backtest в пуле потоков (CPU-bound)
    result = await asyncio.to_thread(run_backtest, df, symbol)

    # Дополнительно сгенерируем кривую доходности по сделкам для графика
    # Для этого воспроизведем логику симуляции, чтобы получить временной ряд PnL
    n_train = int(len(df) * C.BACKTEST_TRAIN_RATIO)
    df_test_raw = df.iloc[n_train:].copy()
    
    # Epsilon из train части
    df_train_raw = df.iloc[:n_train]
    epsilon = float(df_train_raw["ret_1"].abs().quantile(C.CANDLE_EPSILON_QUANTILE))

    # Для построения графика восстановим модель
    try:
        from polyflip.crypto.trainer import _fit_lgbm_and_serialize
        from sklearn.calibration import CalibratedClassifierCV
        import pickle
        
        available = [f for f in C.CRYPTO_FEATURES if f in df.columns]
        df_train = df_train_raw.copy()
        df_train["target"] = (df_train["ret_1"].shift(-1) > 0).astype(int)
        df_train = df_train.dropna(subset=["target", "ret_1"])
        # Применяем фильтр по epsilon к train
        df_train["abs_ret_next"] = df_train["ret_1"].shift(-1).abs()
        df_train = df_train[df_train["abs_ret_next"] >= epsilon]
        
        X_train = df_train[available]
        y_train = df_train["target"]
        
        model_bytes, _, _, _, _, _ = _fit_lgbm_and_serialize(X_train, y_train, n_splits=3)
        model = pickle.loads(model_bytes)
        
        df_test = df_test_raw.copy()
        X_test = df_test[available]
        probas = model.predict_proba(X_test)[:, 1]
        
        df_test["prob_up"] = probas
        df_test["edge"] = probas - 0.5
        df_test["signal"] = df_test["edge"].abs() >= C.BACKTEST_MIN_EDGE
        df_test["ret_next"] = df_test["ret_1"].shift(-1)
        df_test = df_test.dropna(subset=["ret_next"])
        
        trades = df_test[df_test["signal"]].copy()
        if len(trades) > 0:
            trades["direction"] = np.where(trades["edge"] > 0, 1, -1)
            trades["pnl_net"] = (trades["direction"] * trades["ret_next"]) - C.BACKTEST_COMMISSION * 2
            cum_pnl = trades["pnl_net"].cumsum()
            
            # Сократим количество точек до 100 для быстроты рендеринга
            step = max(1, len(trades) // 100)
            pnl_curve = [
                {
                    "time": trades["open_time"].iloc[i].isoformat() if hasattr(trades["open_time"].iloc[i], "isoformat") else str(trades["open_time"].iloc[i]),
                    "pnl": round(float(cum_pnl.iloc[i]) * 100, 2)  # в процентах
                }
                for i in range(0, len(trades), step)
            ]
            # Добавим последнюю точку обязательно
            if len(trades) % step != 0:
                pnl_curve.append({
                    "time": trades["open_time"].iloc[-1].isoformat() if hasattr(trades["open_time"].iloc[-1], "isoformat") else str(trades["open_time"].iloc[-1]),
                    "pnl": round(float(cum_pnl.iloc[-1]) * 100, 2)
                })
        else:
            pnl_curve = []
    except Exception as ex:
        logger.exception("pnl_curve_generation_failed", error=str(ex))
        pnl_curve = []

    import numpy as np
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
        "pnl_curve":        pnl_curve
    }
    
    # Восстанавливаем старые настройки
    C.BACKTEST_MIN_EDGE = old_min_edge
    C.BACKTEST_COMMISSION = old_commission

    _cache[cache_key] = {"ts": now, "data": data}
    return data


@router.post("/api/crypto/train", dependencies=[Depends(verify_api_key)])
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
