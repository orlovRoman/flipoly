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
    """Кэширует загруженную модель в памяти во избежание частой десериализации."""
    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._model_version: int = -1
        self._thresholds: dict[str, tuple[float, float]] = {}  # symbol -> (up, down)
        self._loaded_symbols: set[str] = set()

    async def load(self, db: AsyncSession, symbol: str) -> bool:
        """Ленивая загрузка модели и порогов для CRYPTO доменов."""
        if symbol in self._loaded_symbols and self._model is not None:
            return True  # Уже загружено — выходим без лишних запросов к БД

        try:
            # Загружаем активную LightGBM-модель из ModelRegistry (имя blob - model_blob)
            stmt = select(ModelRegistry).where(
                ModelRegistry.asset == "CRYPTO",
                ModelRegistry.is_active.is_(True)
            )
            row = (await db.execute(stmt)).scalars().first()
            if not row:
                logger.warning("no_active_crypto_model_found")
                return False

            self._model = pickle.loads(row.model_blob)
            self._model_version = row.version

            # Загружаем пороги калибровки из RuntimeSettings
            up_key = f"CRYPTO_THRESHOLD_UP_{symbol.replace('USDT', '')}"
            down_key = f"CRYPTO_THRESHOLD_DOWN_{symbol.replace('USDT', '')}"
            
            set_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_([up_key, down_key]))
            rows = (await db.execute(set_stmt)).scalars().all()
            settings = {r.key: float(r.value) for r in rows}
            
            threshold_up = settings.get(up_key, 0.55)
            threshold_down = settings.get(down_key, 0.45)
            self._thresholds[symbol] = (threshold_up, threshold_down)
            
            self._loaded_symbols.add(symbol)
            logger.info("crypto_model_loaded", version=row.version, symbol=symbol, th_up=threshold_up, th_down=threshold_down)
            return True
        except Exception as e:
            logger.exception("failed_to_load_crypto_model", error=str(e))
            return False

    def predict(self, candles: list[Any], symbol: str) -> CryptoSignal:
        """Синхронный инференс (IO операции выполнены в load())."""
        if not self._model or symbol not in self._thresholds:
            return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False)

        try:
            # 1. Сборка вектора признаков
            feature_vector = build_crypto_features(candles)
            if not feature_vector.valid:
                return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, self._model_version, False)

            fv_dict = dict(zip(CRYPTO_FEATURE_COLUMNS, feature_vector.features[0]))

            # 2. Pydantic-валидация признаков
            validated = CryptoFeaturesValidator(**fv_dict)
            
            # Порядок фичей для LightGBM
            fv_array = np.array([getattr(validated, f) for f in CRYPTO_FEATURES], dtype=np.float64)

            # 3. Инференс
            p_up = float(self._model.predict_proba([fv_array])[0][1])
            p_down = 1.0 - p_up
            
            # Пороги
            th_up, th_down = self._thresholds[symbol]
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
                model_version=self._model_version,
                features_ok=True
            )
        except Exception as e:
            logger.exception("crypto_inference_failed", symbol=symbol, error=str(e))
            return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, self._model_version, False)
