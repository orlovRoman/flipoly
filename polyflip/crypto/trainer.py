"""
LightGBM-тренер для крипто-модели Up/Down на OHLCV-свечах.

Таргет: Up (1) / Down (0) с фильтром |return_15m| >= ε (90-й перцентиль).
Фичи: из feature_builder.build_features().
Сериализация: pickle в ModelRegistry (та же схема, что и LogReg-модель).
"""
from __future__ import annotations

import asyncio
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.constants import (
    BACKTEST_TRAIN_RATIO,
    CANDLE_EPSILON_QUANTILE,
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    LGBM_COLSAMPLE_BYTREE,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_N_ESTIMATORS,
    LGBM_NUM_LEAVES,
    LGBM_REG_ALPHA,
    LGBM_REG_LAMBDA,
    LGBM_SUBSAMPLE,
    MAX_SUSPICIOUS_THRESHOLD,
    MIN_PRECISION_FOR_THRESHOLD,
)
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features
from polyflip.db.models import CryptoCandle, ModelRegistry, RuntimeSettings

logger = structlog.get_logger(__name__)

# Фичи, которые подаём в модель (должны совпадать с колонками из build_features())
CRYPTO_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "vol_6", "vol_24", "vol_48",
    "vol_ratio",          # vol_6 / vol_48 — режим волатильности
    "rsi_14",
    "ema_ratio_9_21",     # ema9 / ema21 — тренд
    "bb_width",           # ширина полос Боллинджера
    "bb_position",        # (close - lower) / (upper - lower)
    "taker_buy_ratio",    # taker_buy_volume / volume
    "hour_utc",           # час дня
    "consec_up",          # кол-во подряд идущих зелёных свечей
    "consec_down",
]


def _build_target(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """
    Вычисляет таргет Up(1)/Down(0) и фильтрует свечи с |return| < epsilon.
    epsilon вычисляется как CANDLE_EPSILON_QUANTILE-перцентиль |ret_1|.
    """
    df = df.copy()
    df["target"] = (df["ret_1"].shift(-1) > 0).astype(int)  # следующая свеча Up?
    df["abs_ret_next"] = df["ret_1"].shift(-1).abs()
    df = df.dropna(subset=["target", "abs_ret_next"])
    df = df[df["abs_ret_next"] >= epsilon].copy()
    return df


def _make_lgbm(**kwargs) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        num_leaves=LGBM_NUM_LEAVES,
        max_depth=LGBM_MAX_DEPTH,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        subsample=LGBM_SUBSAMPLE,
        colsample_bytree=LGBM_COLSAMPLE_BYTREE,
        reg_alpha=LGBM_REG_ALPHA,
        reg_lambda=LGBM_REG_LAMBDA,
        class_weight="balanced",
        random_state=CV_RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
        **kwargs,
    )


def _fit_lgbm_and_serialize(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_N_SPLITS,
    **lgbm_params,
) -> tuple[bytes, float, float, float, float, dict[str, int]]:
    """
    CPU-bound. Обучает LightGBM с TimeSeriesSplit.
    Возвращает: (model_bytes, val_auc, baseline_auc, optimal_threshold, ece, feature_importance)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_scores = np.zeros(len(y))
    aucs: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_lgbm = _make_lgbm(**lgbm_params)
        fold_lgbm.fit(X_train, y_train)

        # Platt-scaling поверх уже обученной модели
        # В sklearn >= 1.2 cv="prefit" удалён; передаём None + estimator уже обучен
        fold_cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(fold_lgbm),
            method="sigmoid",
            cv=None,
        )
        fold_cal.fit(X_val, y_val)

        y_proba = fold_cal.predict_proba(X_val)[:, 1]
        oof_scores[val_idx] = y_proba
        aucs.append(roc_auc_score(y_val, y_proba))

    val_auc = float(np.mean(aucs))
    baseline_auc = float(max(y.mean(), 1.0 - y.mean()))

    # ECE через OOF
    from sklearn.calibration import calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(y, oof_scores, n_bins=10, strategy="uniform")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    except ValueError:
        ece = 0.5  # недостаточно данных для расчёта

    logger.info("crypto_calibration", ece=round(ece, 4))

    # Финальная модель на всех данных
    final_lgbm = _make_lgbm(**lgbm_params)
    final_lgbm.fit(X, y)
    final_cal = CalibratedClassifierCV(
        estimator=FrozenEstimator(final_lgbm),
        method="sigmoid",
        cv=None,
    )
    final_cal.fit(X, y)

    # Оптимальный порог через OOF (без leakage)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y, oof_scores)
    if len(thr_arr) > 0:
        valid = prec_arr[:-1] >= MIN_PRECISION_FOR_THRESHOLD
        f1 = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
        if valid.any():
            optimal_threshold = float(thr_arr[np.argmax(np.where(valid, f1, 0.0))])
        else:
            optimal_threshold = float(thr_arr[np.argmax(f1)])
    else:
        optimal_threshold = 0.55

    # Защита от leakage
    if optimal_threshold >= MAX_SUSPICIOUS_THRESHOLD:
        logger.warning(
            "suspicious_threshold",
            threshold=optimal_threshold,
            message="Clipping to 0.80 to avoid suspected leakage",
        )
        optimal_threshold = 0.80

    # Feature importance для логирования и дашборда
    fi = {
        col: int(imp)
        for col, imp in zip(X.columns, final_lgbm.feature_importances_)
    }
    logger.info("crypto_feature_importance", top5=dict(sorted(fi.items(), key=lambda x: -x[1])[:5]))

    return pickle.dumps(final_cal), val_auc, baseline_auc, optimal_threshold, ece, fi


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
        async def _get_float_setting(key: str, default: float) -> float:
            row = (await self.db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
            return float(row.value) if row else default

        async def _get_int_setting(key: str, default: int) -> int:
            row = (await self.db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))).scalar_one_or_none()
            return int(row.value) if row else default

        n_estimators = await _get_int_setting("CRYPTO_LGBM_N_ESTIMATORS", LGBM_N_ESTIMATORS)
        learning_rate = await _get_float_setting("CRYPTO_LGBM_LEARNING_RATE", LGBM_LEARNING_RATE)
        num_leaves = await _get_int_setting("CRYPTO_LGBM_NUM_LEAVES", LGBM_NUM_LEAVES)
        max_depth = await _get_int_setting("CRYPTO_LGBM_MAX_DEPTH", LGBM_MAX_DEPTH)
        min_child_samples = await _get_int_setting("CRYPTO_LGBM_MIN_CHILD_SAMPLES", LGBM_MIN_CHILD_SAMPLES)
        subsample = await _get_float_setting("CRYPTO_LGBM_SUBSAMPLE", LGBM_SUBSAMPLE)
        colsample_bytree = await _get_float_setting("CRYPTO_LGBM_COLSAMPLE_BYTREE", LGBM_COLSAMPLE_BYTREE)
        reg_alpha = await _get_float_setting("CRYPTO_LGBM_REG_ALPHA", LGBM_REG_ALPHA)
        reg_lambda = await _get_float_setting("CRYPTO_LGBM_REG_LAMBDA", LGBM_REG_LAMBDA)
        eps_quantile = await _get_float_setting("CRYPTO_CANDLE_EPSILON_QUANTILE", CANDLE_EPSILON_QUANTILE)

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

        # Строим фичи
        df = build_features(candles)

        # Фильтруем по ε
        epsilon = float(df["ret_1"].abs().quantile(eps_quantile))
        logger.info("epsilon_filter", symbol=symbol, epsilon=round(epsilon, 5), quantile=eps_quantile)
        df_filtered = _build_target(df, epsilon)

        if len(df_filtered) < 300:
            logger.warning("too_few_after_epsilon_filter", symbol=symbol, rows=len(df_filtered))
            return False

        # Оставляем только доступные фичи
        available = [f for f in CRYPTO_FEATURES if f in df_filtered.columns]
        missing = set(CRYPTO_FEATURES) - set(available)
        if missing:
            logger.warning("missing_features", missing=list(missing))

        # Определяем vol-режим по медиане vol_ratio
        vol_median = float(df_filtered["vol_ratio"].median())
        logger.info("vol_regime_split", symbol=symbol, vol_median=round(vol_median, 4))

        # Сохраняем медиану в RuntimeSettings
        now = datetime.now(timezone.utc)
        median_key = f"CRYPTO_VOL_MEDIAN_{symbol}"
        median_row = (await self.db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == median_key)
        )).scalar_one_or_none()
        if median_row:
            median_row.value = str(round(vol_median, 4))
            median_row.updated_at = now
            median_row.updated_by = "crypto_train_job"
        else:
            self.db.add(RuntimeSettings(
                key=median_key,
                value=str(round(vol_median, 4)),
                updated_at=now,
                updated_by="crypto_train_job",
            ))

        df_low  = df_filtered[df_filtered["vol_ratio"] <= vol_median]
        df_high = df_filtered[df_filtered["vol_ratio"] >  vol_median]

        trained_any = False
        import json

        for regime, df_regime in [("low_vol", df_low), ("high_vol", df_high)]:
            if len(df_regime) < 150:
                logger.warning("regime_too_small", regime=regime, rows=len(df_regime))
                continue

            X_r = df_regime[available].reset_index(drop=True)
            y_r = df_regime["target"].reset_index(drop=True)

            # CPU-bound в thread
            result = await asyncio.to_thread(
                _fit_lgbm_and_serialize, X_r, y_r, CV_N_SPLITS, **lgbm_params
            )
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
                trained_at=now,
            ))
            trained_any = True
            logger.info("crypto_model_saved", asset=regime_asset, version=next_version)

        if trained_any:
            await self.db.commit()
            return True
        return False

