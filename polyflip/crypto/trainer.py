"""
LightGBM-тренер для крипто-модели Up/Down на OHLCV-свечах.

Таргет: Up (1) / Down (0) с фильтром |return_15m| >= ε (90-й перцентиль).
Фичи: из feature_builder.build_features().
Сериализация: pickle в ModelRegistry (та же схема, что и LogReg-модель).
"""
from __future__ import annotations

import asyncio
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator, calibration_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.constants import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
)
from polyflip.services.settings_service import get_float, get_int
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features, CRYPTO_FEATURE_COLUMNS
from polyflip.db.models import CryptoCandle, ModelRegistry, RuntimeSettings

logger = structlog.get_logger(__name__)

CRYPTO_FEATURES = [
    # Returns — только короткие горизонты
    "ret_1", "ret_3", "ret_6",
    # Volatility — только короткие
    "vol_6", "vol_24", "vol_trend",
    # Volume & CVD
    "vol_z_1", "taker_buy_ratio", "cvd_1", "cvd_6", "cvd_trend",
    # Technical
    "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position",
    # Position vs extremes — только 24h
    "dist_to_high_24", "dist_to_low_24",
    # Range
    "range_1", "range_avg_24",
    # Consecutive
    "consec_up", "consec_down",
    # Time (Cyclic)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


# Fail fast при старте: CRYPTO_FEATURES должен быть подмножеством CRYPTO_FEATURE_COLUMNS
_unknown = set(CRYPTO_FEATURES) - set(CRYPTO_FEATURE_COLUMNS)
assert not _unknown, (
    f"CRYPTO_FEATURES содержит фичи, которых нет в feature_builder: {_unknown}"
)


def _build_target(df: pd.DataFrame, epsilon_quantile: float = 0.70) -> pd.DataFrame:
    """
    Вычисляет таргет Up(1)/Down(0) и применяет epsilon-фильтрацию.
    """
    df = df.copy()
    next_ret = df["ret_1"].shift(-1)
    df["target"] = (next_ret > 0).astype(int)
    df["abs_ret_next"] = next_ret.abs()
    df = df.dropna(subset=["target", "abs_ret_next"])

    # Фильтр: обучаем только на "значимых" движениях
    epsilon = float(df["abs_ret_next"].quantile(epsilon_quantile))
    df_filtered = df[df["abs_ret_next"] >= epsilon].copy()

    if len(df_filtered) < 200:
        logger.warning("epsilon_too_aggressive", rows=len(df_filtered))
        return df  # fallback на unfiltered

    logger.info(
        "target_epsilon_filter",
        epsilon_quantile=epsilon_quantile,
        epsilon=round(epsilon, 6),
        before=len(df),
        after=len(df_filtered),
        kept_pct=round(len(df_filtered) / len(df) * 100, 1),
    )
    return df_filtered


def _make_lgbm(**params) -> LGBMClassifier:
    """Вспомогательная функция для создания квалифицированного LGBMClassifier."""
    defaults = {
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        15,
        "max_depth":         4,
        "min_child_samples": 50,
        "subsample":         0.8,
        "colsample_bytree":  1.0,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "random_state":      CV_RANDOM_STATE,
        "n_jobs":            1,
        "verbose":          -1,
    }
    defaults.update(params)
    return LGBMClassifier(**defaults)


def _fit_lgbm_and_serialize(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_N_SPLITS,
    min_precision: float = 0.52,
    min_valid_thr: float = 0.30,
    max_valid_thr: float = 0.75,
    thr_fallback: float = 0.55,
    **lgbm_params,
) -> tuple[bytes, float, float, float, float, dict[str, int]]:
    """
    CPU-bound. Обучает LightGBM с TimeSeriesSplit.
    Возвращает: (model_bytes, val_auc, baseline_auc, optimal_threshold, ece, feature_importance)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_scores = np.full(len(y), np.nan)
    aucs: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_lgbm = _make_lgbm(**lgbm_params)
        fold_lgbm.fit(X_train, y_train)

        # Калибруем на первой половине val, измеряем AUC на второй — без пересечения
        mid = len(val_idx) // 2
        cal_idx, eval_idx = val_idx[:mid], val_idx[mid:]

        if len(cal_idx) >= 20 and len(eval_idx) >= 20:
            X_cal,  y_cal  = X.iloc[cal_idx],  y.iloc[cal_idx]
            X_eval, y_eval = X.iloc[eval_idx], y.iloc[eval_idx]
            calibration_method = "sigmoid" if len(cal_idx) < 200 else "isotonic"
            fold_cal = CalibratedClassifierCV(
                estimator=FrozenEstimator(fold_lgbm), method=calibration_method, cv=None
            )
            fold_cal.fit(X_cal, y_cal)
            y_proba = fold_cal.predict_proba(X_eval)[:, 1]
            oof_scores[eval_idx] = y_proba
            aucs.append(roc_auc_score(y_eval, y_proba))
        else:
            # Фолд слишком мал — используем некалиброванную модель
            y_proba = fold_lgbm.predict_proba(X.iloc[val_idx])[:, 1]
            oof_scores[val_idx] = y_proba
            aucs.append(roc_auc_score(y.iloc[val_idx], y_proba))

    val_auc = float(np.mean(aucs))
    baseline_auc = float(max(y.mean(), 1.0 - y.mean()))

    # ECE через OOF
    valid_mask = ~np.isnan(oof_scores)
    try:
        if valid_mask.sum() > 10:
            frac_pos, mean_pred = calibration_curve(y[valid_mask], oof_scores[valid_mask], n_bins=10, strategy="uniform")
            ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        else:
            ece = 0.5
    except ValueError:
        ece = 0.5  # недостаточно данных для расчёта

    logger.info("crypto_calibration", ece=round(ece, 4))

    # Финальная модель на всех данных
    n_cal = max(50, int(len(X) * 0.15))
    X_fit, X_cal_final = X.iloc[:-n_cal], X.iloc[-n_cal:]
    y_fit, y_cal_final = y.iloc[:-n_cal], y.iloc[-n_cal:]

    final_lgbm = _make_lgbm(**lgbm_params)
    final_lgbm.fit(X_fit, y_fit)
    final_calibration_method = "sigmoid" if n_cal < 200 else "isotonic"
    final_cal = CalibratedClassifierCV(
        estimator=FrozenEstimator(final_lgbm), method=final_calibration_method, cv=None
    )
    final_cal.fit(X_cal_final, y_cal_final)

    # Оптимальный порог через OOF (без leakage)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y[valid_mask], oof_scores[valid_mask])
    if len(thr_arr) > 0:
        valid = prec_arr[:-1] >= min_precision
        f1 = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
        if valid.any():
            optimal_threshold = float(thr_arr[np.argmax(np.where(valid, f1, 0.0))])
        else:
            optimal_threshold = float(thr_arr[np.argmax(f1)])
    else:
        optimal_threshold = thr_fallback

    # Защита от leakage и неадекватных порогов
    if optimal_threshold >= max_valid_thr:
        logger.warning(
            "threshold_clipped_leakage",
            original=optimal_threshold,
            clipped=max_valid_thr,
        )
        optimal_threshold = max_valid_thr

    if optimal_threshold < min_valid_thr or optimal_threshold > max_valid_thr:
        logger.warning(
            "threshold_out_of_bounds",
            threshold=optimal_threshold,
            fallback=thr_fallback,
        )
        optimal_threshold = thr_fallback

    # Feature importance для логирования и дашборда
    fi = {
        col: int(imp)
        for col, imp in zip(X.columns, final_lgbm.feature_importances_)
    }
    logger.info("crypto_feature_importance", top5=dict(sorted(fi.items(), key=lambda x: -x[1])[:5]))

    return pickle.dumps(final_cal), val_auc, baseline_auc, optimal_threshold, ece, fi


async def _get_float_setting(db: AsyncSession, key: str, default: float = 0.0) -> float:
    try:
        return await get_float(db, key)
    except KeyError:
        row = (await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
        return float(row.value) if row else default


async def _get_int_setting(db: AsyncSession, key: str, default: int = 0) -> int:
    try:
        return await get_int(db, key)
    except KeyError:
        row = (await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
        return int(row.value) if row else default


class CryptoModelTrainer:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def train(self, symbol: str, interval: str = "15m") -> bool:
        logger.info("crypto_training_start", symbol=symbol, interval=interval)

        # Загружаем все свечи для символа
        candles = await get_recent_candles(
            self.db, symbol, interval, limit=10_000
        )
        if len(candles) < 500:
            logger.warning("not_enough_candles", symbol=symbol, count=len(candles))
            return False

        # Считываем динамические гиперпараметры из RuntimeSettings
        n_estimators = await _get_int_setting(self.db, "CRYPTO_LGBM_N_ESTIMATORS", 300)
        learning_rate = await _get_float_setting(self.db, "CRYPTO_LGBM_LEARNING_RATE", 0.05)
        num_leaves = await _get_int_setting(self.db, "CRYPTO_LGBM_NUM_LEAVES", 31)
        max_depth = await _get_int_setting(self.db, "CRYPTO_LGBM_MAX_DEPTH", 5)
        min_child_samples = await _get_int_setting(self.db, "CRYPTO_LGBM_MIN_CHILD_SAMPLES", 20)
        subsample = await _get_float_setting(self.db, "CRYPTO_LGBM_SUBSAMPLE", 0.8)
        colsample_bytree = await _get_float_setting(self.db, "CRYPTO_LGBM_COLSAMPLE_BYTREE", 0.8)
        reg_alpha = await _get_float_setting(self.db, "CRYPTO_LGBM_REG_ALPHA", 0.1)
        reg_lambda = await _get_float_setting(self.db, "CRYPTO_LGBM_REG_LAMBDA", 1.0)
        lgbm_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }

        min_precision = await get_float(self.db, "LGBM_MIN_PRECISION_FOR_THRESHOLD")
        min_valid_thr = await get_float(self.db, "LGBM_MIN_VALID_THRESHOLD")
        max_valid_thr = await get_float(self.db, "LGBM_MAX_VALID_THRESHOLD")
        thr_fallback = await get_float(self.db, "LGBM_THRESHOLD_FALLBACK")
        cv_n_splits = await get_int(self.db, "LGBM_CV_N_SPLITS")
        epsilon_quantile = await get_float(self.db, "LGBM_EPSILON_QUANTILE")

        # Читаем актуальные funding rates из БД
        fr_key = f"FUNDING_RATE_{symbol}"
        fr_ma3_key = f"FUNDING_RATE_MA3_{symbol}"
        fr_row = (await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == fr_key)
        )).scalar_one_or_none()
        fr_ma3_row = (await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == fr_ma3_key)
        )).scalar_one_or_none()

        funding_rate = float(fr_row.value) if fr_row else 0.0
        funding_rate_ma3 = float(fr_ma3_row.value) if fr_ma3_row else 0.0

        logger.info(
            "funding_rate_loaded_for_training",
            symbol=symbol,
            funding_rate=funding_rate,
            ma3=funding_rate_ma3,
        )

        # Строим фичи
        df = build_features(
            candles,
            funding_rate=funding_rate,
            funding_rate_ma3=funding_rate_ma3,
        )

        df_filtered = _build_target(df, epsilon_quantile=epsilon_quantile)

        if len(df_filtered) < 300:
            logger.warning("too_few_candles", symbol=symbol, rows=len(df_filtered))
            return False

        # Оставляем только доступные фичи
        available = [f for f in CRYPTO_FEATURES if f in df_filtered.columns]
        missing = set(CRYPTO_FEATURES) - set(available)
        if missing:
            logger.warning("missing_features", missing=list(missing))

        # Определяем vol-режим по P33/P67 vol_ratio
        vol_p33 = float(df_filtered["vol_ratio"].quantile(0.33))
        vol_p67 = float(df_filtered["vol_ratio"].quantile(0.67))

        logger.info(
            "vol_regime_tertiles",
            symbol=symbol,
            p33=round(vol_p33, 4),
            p67=round(vol_p67, 4),
        )

        now = datetime.now(timezone.utc)
        # Сохраняем tertile-границы для предсказателя
        for key, val in [(f"CRYPTO_VOL_P33_{symbol}", vol_p33), (f"CRYPTO_VOL_P67_{symbol}", vol_p67)]:
            row = (await self.db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
            if row:
                row.value = str(round(val, 4))
                row.updated_at = now
                row.updated_by = "crypto_train_job"
            else:
                self.db.add(RuntimeSettings(key=key, value=str(round(val, 4)), updated_at=now, updated_by="crypto_train_job"))

        df_low  = df_filtered[df_filtered["vol_ratio"] <= vol_p33]
        df_mid  = df_filtered[(df_filtered["vol_ratio"] > vol_p33) & (df_filtered["vol_ratio"] <= vol_p67)]
        df_high = df_filtered[df_filtered["vol_ratio"] > vol_p67]

        trained_any = False
        from polyflip.crypto.predictor import CryptoPredictor

        for regime, df_regime in [("low_vol", df_low), ("mid_vol", df_mid), ("high_vol", df_high)]:
            if len(df_regime) < 300:
                logger.warning("regime_too_small", regime=regime, rows=len(df_regime))
                continue

            try:
                X_r = df_regime[available].reset_index(drop=True)
                y_r = df_regime["target"].reset_index(drop=True)
                
                n_regime = len(df_regime)
                adaptive_params = lgbm_params.copy()
                if n_regime < 500:
                    adaptive_params["num_leaves"] = 15
                    adaptive_params["max_depth"] = 4
                    adaptive_params["min_child_samples"] = 30
                    adaptive_params["n_estimators"] = 200
                elif n_regime < 1000:
                    adaptive_params["num_leaves"] = 20
                    adaptive_params["max_depth"] = 5
                    adaptive_params["min_child_samples"] = 25
                
                logger.info("adaptive_lgbm_params", regime=regime, n_regime=n_regime, num_leaves=adaptive_params["num_leaves"])

                # CPU-bound в thread
                t0 = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            _fit_lgbm_and_serialize,
                            X_r, y_r, cv_n_splits,
                            min_precision,
                            min_valid_thr,
                            max_valid_thr,
                            thr_fallback,
                            **adaptive_params
                        ),
                        timeout=1800.0,   # 30 минут — hard limit
                    )
                except asyncio.TimeoutError:
                    logger.error("regime_train_timeout", symbol=symbol, regime=regime)
                    continue
                finally:
                    logger.info("regime_train_duration", symbol=symbol, regime=regime,
                                elapsed_sec=round(time.monotonic() - t0, 1))

                model_bytes, val_auc, baseline_auc, threshold, ece, fi = result

                logger.info(
                    "crypto_regime_model_trained",
                    symbol=symbol,
                    regime=regime,
                    val_auc=round(val_auc, 4),
                    baseline=round(baseline_auc, 4),
                    threshold=round(threshold, 4),
                    ece=round(ece, 4),
                )

                regime_asset = f"{symbol}_{regime}"

                # Деактивируем старые записи
                await self.db.execute(
                    update(ModelRegistry)
                    .where(ModelRegistry.asset == regime_asset)
                    .values(is_active=False)
                )

                # Версионирование
                v_res = await self.db.execute(
                    select(ModelRegistry.version)
                    .where(ModelRegistry.asset == regime_asset)
                    .order_by(ModelRegistry.version.desc())
                    .limit(1)
                )
                next_version = (v_res.scalar_one_or_none() or 0) + 1

                # Сохраняем порог в RuntimeSettings
                thr_key = f"CRYPTO_THRESHOLD_{regime_asset}"
                thr_row = (await self.db.execute(
                    select(RuntimeSettings).where(RuntimeSettings.key == thr_key)
                )).scalar_one_or_none()

                threshold_quality = "ok"
                if threshold < 0.40 or threshold > 0.65:
                    threshold_quality = "marginal"

                logger.info(
                    "threshold_saved",
                    key=thr_key,
                    value=round(threshold, 4),
                    quality=threshold_quality,
                )

                if thr_row:
                    thr_row.value = str(round(threshold, 4))
                    thr_row.updated_at = now
                    thr_row.updated_by = "crypto_train_job"
                else:
                    self.db.add(RuntimeSettings(
                        key=thr_key,
                        value=str(round(threshold, 4)),
                        updated_at=now,
                        updated_by="crypto_train_job",
                    ))

                # Сохраняем feature importance в RuntimeSettings
                fi_key = f"CRYPTO_FI_{regime_asset}"
                fi_row = (await self.db.execute(
                    select(RuntimeSettings).where(RuntimeSettings.key == fi_key)
                )).scalar_one_or_none()
                if fi_row:
                    fi_row.value = json.dumps(fi)
                    fi_row.updated_at = now
                    fi_row.updated_by = "crypto_train_job"
                else:
                    self.db.add(RuntimeSettings(
                        key=fi_key,
                        value=json.dumps(fi),
                        updated_at=now,
                        updated_by="crypto_train_job",
                    ))

                # Сохраняем модель
                self.db.add(ModelRegistry(
                    asset=regime_asset,
                    version=next_version,
                    model_blob=model_bytes,
                    accuracy=val_auc,
                    baseline=baseline_auc,
                    features=",".join(available),
                    ece=ece,
                    is_active=True,
                    interval=interval,
                    trained_at=now,
                ))
                await self.db.commit()
                trained_any = True
                logger.info("crypto_model_saved", asset=regime_asset, version=next_version)
            except Exception as e:
                await self.db.rollback()
                logger.exception("regime_train_failed", symbol=symbol, regime=regime, error=str(e))

        if trained_any:
            # Инвалидируем кэш у инстансов предсказателя после коммита в базу
            CryptoPredictor.invalidate_all(symbol)
            return True
        return False

