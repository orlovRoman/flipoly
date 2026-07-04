from __future__ import annotations
import pickle
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.db.models import ModelRegistry, RuntimeSettings
from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.edge import compute_crypto_edge
from polyflip.crypto.trainer import CRYPTO_FEATURES


logger = structlog.get_logger(__name__)

# Схема валидации входного вектора признаков перед инференсом
class CryptoFeaturesValidator(BaseModel):
    ret_1: float
    ret_3: float
    ret_6: float
    ret_12: float
    ret_24: float
    ret_48: float
    vol_6: float
    vol_24: float
    vol_48: float
    vol_ratio: float
    vol_z_6: float
    taker_buy_ratio: float
    rsi_14: float
    ema_ratio_9_21: float
    bb_width: float
    bb_position: float
    dist_to_high_24: float
    dist_to_low_24: float
    dist_to_high_96: float
    dist_to_low_96: float
    range_1: float
    range_avg_24: float
    consec_up: float
    consec_down: float
    hour_utc: float
    dow: float

    @field_validator("*", mode="before")
    @classmethod
    def check_nan_or_none(cls, v: Any) -> float:
        if v is None:
            raise ValueError("Feature value cannot be None")
        fval = float(v)
        if np.isnan(fval) or np.isinf(fval):
            raise ValueError("Feature value cannot be NaN or Inf")
        return fval

@dataclass(frozen=True)
class CryptoSignal:
    symbol: str              # "BTCUSDT" | "ETHUSDT"
    p_up: float              # Вероятность роста [0, 1]
    p_down: float            # = 1 - p_up
    direction: str           # "UP" | "DOWN" | "NONE"
    edge: float              # Сила сигнала относительно порога
    strike: float            # close-цена последней закрытой свечи Binance
    threshold_up: float      # Порог для лонга (BUY_UP)
    threshold_down: float    # Порог для шорта (BUY_DOWN)
    model_version: int       # Версия модели из реестра
    features_ok: bool        # False, если не прошли Pydantic-валидацию

class CryptoPredictor:
    """Кэширует загруженные модели в памяти во избежание частой десериализации."""
    _instances: list["CryptoPredictor"] = []

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        self._model_versions: dict[str, dict[str, int]] = {}
        self._thresholds: dict[str, dict[str, tuple[float, float]]] = {}
        self._vol_medians: dict[str, float] = {}
        self._loaded_symbols: set[str] = set()
        CryptoPredictor._instances.append(self)

    @classmethod
    def invalidate_all(cls, symbol: str) -> None:
        """
        Инвалидирует кэш для symbol во всех живых инстансах.
        Вызывается из trainer.py после успешного переобучения.
        """
        for inst in cls._instances:
            inst._loaded_symbols.discard(symbol)
            inst._models.pop(symbol, None)
            inst._model_versions.pop(symbol, None)
            inst._thresholds.pop(symbol, None)
            inst._vol_medians.pop(symbol, None)
        logger.info("predictor_cache_invalidated", symbol=symbol, instances=len(cls._instances))

    def invalidate(self, symbol: str) -> None:
        """Инвалидирует локальный кэш для указанного символа."""
        self._loaded_symbols.discard(symbol)
        self._models.pop(symbol, None)
        self._model_versions.pop(symbol, None)
        self._thresholds.pop(symbol, None)
        self._vol_medians.pop(symbol, None)

    async def load(self, db: AsyncSession, symbol: str) -> bool:
        """Ленивая загрузка моделей и порогов для low_vol и high_vol."""
        if symbol in self._loaded_symbols:
            return True

        try:
            # 1. Загружаем медиану волатильности из RuntimeSettings
            median_key = f"CRYPTO_VOL_MEDIAN_{symbol}"
            median_row = (await db.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == median_key)
            )).scalar_one_or_none()
            vol_median = float(median_row.value) if median_row else 1.0
            self._vol_medians[symbol] = vol_median

            self._models[symbol] = {}
            self._model_versions[symbol] = {}
            self._thresholds[symbol] = {}

            for regime in ["low_vol", "high_vol"]:
                regime_asset = f"{symbol}_{regime}"
                stmt = select(ModelRegistry).where(
                    ModelRegistry.asset == regime_asset,
                    ModelRegistry.is_active.is_(True)
                )
                row = (await db.execute(stmt)).scalars().first()
                
                # Обратная совместимость: если нет двухрежимной модели, ищем старую общую по "CRYPTO"
                if not row:
                    logger.warning("no_active_regime_model_found", asset=regime_asset)
                    fallback_stmt = select(ModelRegistry).where(
                        ModelRegistry.asset == "CRYPTO",
                        ModelRegistry.is_active.is_(True)
                    )
                    row = (await db.execute(fallback_stmt)).scalars().first()
                    
                if not row:
                    logger.error("no_fallback_model_found", symbol=symbol)
                    # ВАЖНО: при неудаче не добавляем в loaded_symbols и очищаем частично загруженное
                    self.invalidate(symbol)
                    return False

                self._models[symbol][regime] = pickle.loads(row.model_blob)
                self._model_versions[symbol][regime] = row.version

                # Пороги: берем CRYPTO_THRESHOLD_BTCUSDT_low_vol или общие CRYPTO_THRESHOLD_UP_BTC / DOWN_BTC
                thr_key = f"CRYPTO_THRESHOLD_{regime_asset}"
                thr_row = (await db.execute(
                    select(RuntimeSettings).where(RuntimeSettings.key == thr_key)
                )).scalar_one_or_none()
                
                if thr_row:
                    threshold = float(thr_row.value)
                    th_up = threshold
                    th_down = 1.0 - threshold
                else:
                    coin_prefix = symbol.replace("USDT", "")
                    up_key = f"CRYPTO_THRESHOLD_UP_{coin_prefix}"
                    down_key = f"CRYPTO_THRESHOLD_DOWN_{coin_prefix}"
                    rows = (await db.execute(
                        select(RuntimeSettings).where(RuntimeSettings.key.in_([up_key, down_key]))
                    )).scalars().all()
                    settings = {r.key: float(r.value) for r in rows}
                    th_up = settings.get(up_key, 0.55)
                    th_down = settings.get(down_key, 0.45)

                self._thresholds[symbol][regime] = (th_up, th_down)
                logger.info(
                    "crypto_regime_model_loaded",
                    symbol=symbol, regime=regime, version=row.version,
                    th_up=th_up, th_down=th_down, vol_median=vol_median
                )

            self._loaded_symbols.add(symbol)
            return True
        except Exception as e:
            logger.exception("failed_to_load_crypto_models", error=str(e))
            self.invalidate(symbol)
            return False

    def predict(self, candles: list[Any], symbol: str) -> CryptoSignal:
        """Синхронный инференс по нужной модели (в зависимости от vol_ratio)."""
        if symbol not in self._loaded_symbols:
            return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False)

        try:
            # 1. Сборка вектора признаков
            feature_vector = build_crypto_features(candles)
            if not feature_vector.valid:
                return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False)

            fv_dict = dict(zip(CRYPTO_FEATURE_COLUMNS, feature_vector.features[0]))
            
            # Определяем режим волатильности
            vol_ratio = fv_dict.get("vol_ratio", 1.0)
            vol_median = self._vol_medians.get(symbol, 1.0)
            regime = "low_vol" if vol_ratio <= vol_median else "high_vol"

            # 2. Pydantic-валидация признаков
            validated = CryptoFeaturesValidator(**fv_dict)
            
            # Порядок фичей для LightGBM
            fv_array = np.array([getattr(validated, f) for f in CRYPTO_FEATURES], dtype=np.float64)

            # Выбор модели с защитой от отсутствия конкретного режима (fallback)
            symbol_models = self._models.get(symbol, {})
            model = symbol_models.get(regime) or next(iter(symbol_models.values()), None)
            
            version = (self._model_versions.get(symbol, {}).get(regime)
                       or next(iter(self._model_versions.get(symbol, {}).values()), -1))
                       
            th_up, th_down = (self._thresholds.get(symbol, {}).get(regime)
                              or next(iter(self._thresholds.get(symbol, {}).values()), (0.55, 0.45)))

            if model is None:
                logger.warning("no_model_available_for_predict", symbol=symbol, regime=regime)
                return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False)

            # 3. Инференс
            p_up = float(model.predict_proba([fv_array])[0][1])
            p_down = 1.0 - p_up
            
            edge, direction = compute_crypto_edge(p_up, th_up, th_down)
            
            # Страйк (цена последней закрытой свечи)
            strike = float(candles[-1].close)

            return CryptoSignal(
                symbol=symbol,
                p_up=p_up,
                p_down=p_down,
                direction=direction,
                edge=edge,
                strike=strike,
                threshold_up=th_up,
                threshold_down=th_down,
                model_version=version,
                features_ok=True
            )
        except Exception as e:
            logger.exception("crypto_inference_failed", symbol=symbol, error=str(e))
            return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False)

