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
    "price_deviation",
    "spread_pct",
    "log_time_left",
    "is_final_phase",
    "high_price_final",
    # Лаговые (динамические)
    *LAG_FEATURE_NAMES,  # price_momentum, spread_trend, volume_trend
]


@dataclass(frozen=True)
class MarketSignal:
    """
    Снимок рыночного состояния в момент принятия решения.
    Не зависит от БД, API или других внешних систем.
    Используется и движком, и бэктестом.
    """

    asset: str
    mid_price: float  # вероятность YES по mid
    spread: float  # best_ask - best_bid
    volume_5min: float  # объём за последние 5 минут
    price_velocity: float  # скорость изменения mid_price
    hour_of_day: (
        int  # час дня в UTC (0–23); намеренно UTC — зафиксировано как стандарт.
    )
    # Переход на ET (UTC-5/UTC-4) отложен до v2.x: потребует переобучения моделей.
    time_left_min: float  # минут до закрытия рынка
    market_duration_min: float = (
        60.0  # полная длительность рынка в минутах (default 60.0)
    )

    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None

    def get_yes_ask(self) -> float:
        if self.yes_ask is not None and self.yes_ask > 0:
            return self.yes_ask
        return min(self.mid_price + self.spread / 2, 0.99)

    def get_yes_bid(self) -> float:
        if self.yes_bid is not None and self.yes_bid > 0:
            return self.yes_bid
        return max(self.mid_price - self.spread / 2, 0.01)

    def get_no_ask(self) -> float:
        if self.no_ask is not None and self.no_ask > 0:
            return self.no_ask
        return min((1.0 - self.mid_price) + self.spread / 2, 0.99)

    def get_no_bid(self) -> float:
        if self.no_bid is not None and self.no_bid > 0:
            return self.no_bid
        return max((1.0 - self.mid_price) - self.spread / 2, 0.01)


def build_feature_vector(
    signal: MarketSignal,
    lag_history: list[dict] | None = None,
    price_max_observed: float | None = None,
) -> np.ndarray:
    """
    Возвращает numpy array shape (1, N) для model.predict_proba().
    Порядок колонок строго соответствует FEATURE_COLUMNS.
    """
    import pandas as pd
    from datetime import datetime, timezone
    from polyflip.models.trainer import add_derived_features
    from polyflip.models.feature_lags import add_lag_features

    current_duration = getattr(signal, "market_duration_min", 60.0) or 60.0

    current_row = {
        "market_id": signal.asset,
        "recorded_at": datetime.now(timezone.utc),
        "time_left_min": signal.time_left_min,
        "mid_price": signal.mid_price,
        "spread": signal.spread,
        "volume_5min": signal.volume_5min,
        "price_velocity": signal.price_velocity,
        "hour_of_day": signal.hour_of_day,
        "market_duration_min": current_duration,
    }

    if lag_history:
        rows = lag_history[-6:] + [current_row]
        df = pd.DataFrame(rows)
        df = add_derived_features(df)
        if price_max_observed is not None:
            df["price_distance_from_max"] = (price_max_observed - df["mid_price"]).clip(
                lower=0.0
            )
        else:
            df["price_distance_from_max"] = 0.02
        df = add_lag_features(df)
        df = df.tail(1).reset_index(drop=True)
    else:
        df = pd.DataFrame([current_row])
        df = add_derived_features(df)
        if price_max_observed is not None:
            df["price_distance_from_max"] = max(
                price_max_observed - signal.mid_price, 0.0
            )
        else:
            df["price_distance_from_max"] = 0.02
        df = add_lag_features(df)

    # Заполняем NaN в лагах или других колонках нулями для совместимости с формой
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    return df[FEATURE_COLUMNS].to_numpy()


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
        market_duration_min=float(getattr(row, "market_duration_min", 60.0) or 60.0),
    )
