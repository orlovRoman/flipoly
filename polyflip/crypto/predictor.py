from __future__ import annotations
import pickle
import weakref
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

MIN_CANDLES_REQUIRED = 110   # запас +10% к min_candles=100

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
    vol_trend: float
    vol_z_1: float
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
    ece: float = 0.0         # BUG-AO

class CryptoPredictor:
    """Кэширует загруженные модели в памяти во избежание частой десериализации."""
    _instances: list[weakref.ref] = []

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        self._model_versions: dict[str, dict[str, int]] = {}
        self._model_intervals: dict[str, dict[str, str]] = {}
        self._thresholds: dict[str, dict[str, tuple[float, float]]] = {}
        self._model_eces: dict[str, dict[str, float]] = {} # BUG-AO
        self._vol_p33s: dict[str, float] = {}
        self._vol_p67s: dict[str, float] = {}
        self._loaded_symbols: set[str] = set()
        CryptoPredictor._instances.append(weakref.ref(self))

    @classmethod
    def invalidate_all(cls, symbol: str) -> None:
        """
        Инвалидирует кэш для symbol во всех живых инстансах.
        Вызывается из trainer.py после успешного переобучения.
        """
        alive = []
        for ref in cls._instances:
            inst = ref()
            if inst is not None:
                inst._loaded_symbols.discard(symbol)
                inst._models.pop(symbol, None)
                inst._model_versions.pop(symbol, None)
                inst._model_eces.pop(symbol, None) # BUG-AO
                inst._model_intervals.pop(symbol, None)
                inst._thresholds.pop(symbol, None)
                inst._vol_p33s.pop(symbol, None)
                inst._vol_p67s.pop(symbol, None)
                alive.append(ref)
        cls._instances = alive
        logger.info("predictor_cache_invalidated", symbol=symbol, instances=len(alive))

    def invalidate(self, symbol: str) -> None:
        """Инвалидирует локальный кэш для указанного символа."""
        self._loaded_symbols.discard(symbol)
        self._models.pop(symbol, None)
        self._model_versions.pop(symbol, None)
        self._model_intervals.pop(symbol, None)
        self._thresholds.pop(symbol, None)
        self._model_eces.pop(symbol, None) # BUG-AO
        self._vol_p33s.pop(symbol, None)
        self._vol_p67s.pop(symbol, None)

    def get_interval(self, symbol: str) -> str:
        """Возвращает интервал обучения для моделей указанного символа (по умолчанию '15m')."""
        if symbol in self._model_intervals:
            for val in self._model_intervals[symbol].values():
                return val
        return "15m"

    async def load(self, db: AsyncSession, symbol: str) -> bool:
        """Ленивая загрузка моделей и порогов для low_vol и high_vol с авто-обновлением по БД."""
        try:
            allowed_assets = [f"{symbol}_low_vol", f"{symbol}_mid_vol", f"{symbol}_high_vol"]
            stmt = select(ModelRegistry.asset, ModelRegistry.version).where(
                ModelRegistry.asset.in_(allowed_assets),
                ModelRegistry.is_active
            )
            db_versions = (await db.execute(stmt)).all()
            db_ver_dict = {row.asset: row.version for row in db_versions}
            
            # Если для какого-то режима модель еще не обучалась, делаем fallback на "CRYPTO"
            for regime in ["low_vol", "mid_vol", "high_vol"]:
                reg_asset = f"{symbol}_{regime}"
                if reg_asset not in db_ver_dict:
                    fallback_stmt = select(ModelRegistry.version).where(
                        ModelRegistry.asset == "CRYPTO",
                        ModelRegistry.is_active
                    )
                    f_ver = (await db.execute(fallback_stmt)).scalar()
                    if f_ver is not None:
                        db_ver_dict[reg_asset] = f_ver
            
            # Проверяем, совпадает ли то, что загружено в память, с актуальным в БД
            if symbol in self._loaded_symbols:
                cache_ok = True
                for asset, ver in db_ver_dict.items():
                    regime = asset.replace(f"{symbol}_", "")
                    if self._model_versions.get(symbol, {}).get(regime) != ver:
                        cache_ok = False
                        break
                if cache_ok:
                    return True
                else:
                    logger.info("new_models_detected_in_db", symbol=symbol, old_versions=self._model_versions.get(symbol), new_versions=db_ver_dict)
                    self.invalidate(symbol)
        except Exception as e:
            logger.warning("failed_to_check_db_model_versions", symbol=symbol, error=str(e))
            if symbol in self._loaded_symbols:
                return True

        try:
            # 1. Загружаем квантили волатильности из RuntimeSettings
            p33_key = f"CRYPTO_VOL_P33_{symbol}"
            p33_row = (await db.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == p33_key)
            )).scalar_one_or_none()
            self._vol_p33s[symbol] = float(p33_row.value) if p33_row else 0.5

            p67_key = f"CRYPTO_VOL_P67_{symbol}"
            p67_row = (await db.execute(
                select(RuntimeSettings).where(RuntimeSettings.key == p67_key)
            )).scalar_one_or_none()
            self._vol_p67s[symbol] = float(p67_row.value) if p67_row else 1.5

            self._models[symbol] = {}
            self._model_versions[symbol] = {}
            self._model_intervals[symbol] = {}
            self._thresholds[symbol] = {}

            for regime in ["low_vol", "mid_vol", "high_vol"]:
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
                self._model_intervals[symbol][regime] = getattr(row, 'interval', '15m')
                self._model_eces[symbol][regime] = row.ece or 0.0 # BUG-AO

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
                    th_up=th_up, th_down=th_down, vol_p33=self._vol_p33s[symbol], vol_p67=self._vol_p67s[symbol]
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
            return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False, 0.0)

        try:
            # 1. Сборка вектора признаков
            feature_vector = build_crypto_features(candles)
            if not feature_vector.valid:
                return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False, 0.0)

            fv_dict = dict(zip(CRYPTO_FEATURE_COLUMNS, feature_vector.features[0]))
            
            # Определяем режим волатильности
            vol_ratio = fv_dict.get("vol_ratio", 1.0)
            vol_p33 = self._vol_p33s.get(symbol, 0.5)
            vol_p67 = self._vol_p67s.get(symbol, 1.5)

            if vol_ratio <= vol_p33:
                regime = "low_vol"
            elif vol_ratio <= vol_p67:
                regime = "mid_vol"
            else:
                regime = "high_vol"

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
                return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False, 0.0)

            # 3. Инференс
            p_up = float(model.predict_proba([fv_array])[0][1])
            p_down = 1.0 - p_up
            
            edge, direction = compute_crypto_edge(p_up, th_up, th_down)
            
            # Страйк (цена последней закрытой свечи)
            strike = float(candles[-1].close)

            ece = (self._model_eces.get(symbol, {}).get(regime)
                   or next(iter(self._model_eces.get(symbol, {}).values()), 0.0))

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
                features_ok=True,
                ece=ece # BUG-AO
            )
        except Exception as e:
            logger.exception("crypto_inference_failed", symbol=symbol, error=str(e))
            return CryptoSignal(symbol, 0.5, 0.5, "NONE", 0.0, 0.0, 0.5, 0.5, -1, False, 0.0)

