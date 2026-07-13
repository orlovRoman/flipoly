"""
Вычисление лаговых (динамических) фич из DataFrame MarketSnapshot.

Требования к входному DataFrame:
  - колонки: market_id, recorded_at, mid_price, spread, volume_5min, price_velocity
  - строки могут принадлежать разным market_id

Лаги вычисляются через pandas groupby + shift, без изменения схемы БД.
Строки с NaN в лаговых фичах заполняются медианой по колонке (imputation).
"""
import numpy as np
import pandas as pd

# Количество снапшотов назад для каждого лага
# Если коллектор пишет снапшот каждые ~5 минут:
#   LAG_1 ≈ 5 мин, LAG_3 ≈ 15 мин, LAG_6 ≈ 30 мин
LAG_1 = 1   # ~5 мин
LAG_3 = 3   # ~15 мин
LAG_6 = 6   # ~30 мин

LAG_FEATURE_NAMES = [
    "price_velocity_lag1",
    "price_momentum",
    "spread_trend",
    "volume_trend",
]

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет лаговые фичи к DataFrame.
    
    ВАЖНО: входной df ДОЛЖЕН содержать колонки market_id и recorded_at.
    Функция сортирует df внутри себя, но НЕ изменяет порядок строк (reset_index).
    """
    df = df.copy()

    if df.empty:
        for col in LAG_FEATURE_NAMES:
            df[col] = 0.0
        return df

    # Сортируем для корректного shift() внутри каждого рынка
    df = df.sort_values(["market_id", "recorded_at"]).reset_index(drop=True)

    grp = df.groupby("market_id", sort=False)

    # ── price_velocity_lag1: скорость цены 5 минут назад ──────────────────
    df["price_velocity_lag1"] = grp["price_velocity"].shift(LAG_1)

    # ── price_momentum: импульс mid_price за LAG_3 снапшота ───────────────
    df["price_momentum"] = df["mid_price"] - grp["mid_price"].shift(LAG_3)

    # ── spread_trend: спред растёт или сжимается ──────────────────────────
    spread_lag = grp["spread"].shift(LAG_6)
    df["spread_trend"] = df["spread"] / (spread_lag + 1e-8)
    df["spread_trend"] = df["spread_trend"].clip(upper=10.0)

    # ── volume_trend: всплеск объёма ──────────────────────────────────────
    vol_lag = grp["volume_5min"].shift(LAG_3)
    df["volume_trend"] = df["volume_5min"] / (vol_lag + 1e-8)
    df["volume_trend"] = df["volume_trend"].clip(upper=10.0)

    # ── Imputation: NaN (первые LAG_N строк каждого рынка) → медиана ─────
    lag_cols = ["price_velocity_lag1", "price_momentum", "spread_trend", "volume_trend"]
    for col in lag_cols:
        median_val = df[col].median()
        fill_val = 0.0 if pd.isna(median_val) else float(median_val)
        df[col] = df[col].fillna(fill_val)

    return df
