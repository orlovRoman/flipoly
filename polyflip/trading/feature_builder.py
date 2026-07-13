"""
Построение feature-вектора для ML-модели.
Единственный источник правды для порядка и состава фичей.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ВАЖНО: порядок колонок должен совпадать с порядком при обучении модели.
# Если модель переобучается — обновить этот список.
from polyflip.models.feature_lags import LAG_FEATURE_NAMES

FEATURE_COLUMNS: list[str] = [
    # Базовые (существующие)
    "time_left_min",
    "mid_price",
    "spread",
    "volume_5min",
    "price_velocity",
    "hour_of_day",
    # Новые статические
    "day_of_week",
    "price_distance_from_max",
    "time_phase",
    "price_deviation",
    "deviation_x_time",
    "price_deviation_sq",
    "spread_pct",
    "log_time_left",
    # Лаговые (динамические)
    *LAG_FEATURE_NAMES,   # price_velocity_lag1, price_momentum, spread_trend, volume_trend
]

@dataclass(frozen=True)
class MarketSignal:
    """
    Снимок рыночного состояния в момент принятия решения.
    Не зависит от БД, API или других внешних систем.
    Используется и движком, и бэктестом.
    """
    asset: str
    mid_price: float          # вероятность YES по mid
    spread: float             # best_ask - best_bid
    volume_5min: float        # объём за последние 5 минут
    price_velocity: float     # скорость изменения mid_price
    hour_of_day: int          # час дня в UTC (0–23); намеренно UTC — зафиксировано как стандарт.
                              # Переход на ET (UTC-5/UTC-4) отложен до v2.x: потребует переобучения моделей.
    time_left_min: float      # минут до закрытия рынка

    # Симулированные цены (вычисляются из mid_price и spread)
    @property
    def yes_ask(self) -> float:
        return min(self.mid_price + self.spread / 2, 0.99)

    @property
    def yes_bid(self) -> float:
        return max(self.mid_price - self.spread / 2, 0.01)

    @property
    def no_ask(self) -> float:
        return min((1.0 - self.mid_price) + self.spread / 2, 0.99)

    @property
    def no_bid(self) -> float:
        return max((1.0 - self.mid_price) - self.spread / 2, 0.01)


def build_feature_vector(signal: MarketSignal) -> np.ndarray:
    """
    Возвращает numpy array shape (1, N) для model.predict_proba().
    Порядок колонок строго соответствует FEATURE_COLUMNS.
    """
    return np.array([[
        signal.time_left_min,
        signal.mid_price,
        signal.spread,
        signal.volume_5min,
        signal.price_velocity,
        signal.hour_of_day,
    ]], dtype=np.float64)


def signal_from_snapshot_row(row) -> MarketSignal:
    """
    Создаёт MarketSignal из ORM-объекта MarketSnapshot.
    Используется в BacktestRunner при загрузке исторических данных.
    """
    return MarketSignal(
        asset=row.asset,
        mid_price=float(row.mid_price),
        spread=float(row.spread) if row.spread else 0.01,
        volume_5min=float(row.volume_5min) if row.volume_5min else 0.0,
        price_velocity=float(row.price_velocity) if row.price_velocity else 0.0,
        hour_of_day=int(row.hour_of_day) if row.hour_of_day is not None else 0,
        time_left_min=float(row.time_left_min) if row.time_left_min else 0.0,
    )
