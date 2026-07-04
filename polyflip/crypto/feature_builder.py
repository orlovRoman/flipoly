"""
Построение вектора фичей для Up/Down модели по 15m-свечам.

Входные данные : последние 100–200 свечей CryptoCandle (ASC по open_time).
Выходные данные: numpy array shape (1, N) + CRYPTO_FEATURE_COLUMNS.

Принципы:
  - Только pandas + numpy. Никаких внешних зависимостей.
  - Все вычисления детерминированы.
  - Нет lookahead: используем только свечи до текущей включительно.
  - NaN/Inf → 0.0 через np.nan_to_num (safety net).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence

CRYPTO_FEATURE_COLUMNS: list[str] = [
    # --- Returns (log) ---
    "ret_1",        # log-return последней свечи
    "ret_3",        # log-return за 3 свечи  (≈45 мин)
    "ret_6",        # log-return за 6 свечей (≈90 мин)
    "ret_12",       # log-return за 12 свечей (≈3 ч)
    "ret_24",       # log-return за 24 свечи  (≈6 ч)
    "ret_48",       # log-return за 48 свечей (≈12 ч)
    # --- Volatility ---
    "vol_6",        # std log-returns за 6 свечей
    "vol_24",       # std log-returns за 24 свечи
    "vol_48",       # std log-returns за 48 свечей
    "vol_ratio",    # vol_6 / vol_48 — высокая/низкая волатильность
    # --- Volume anomaly ---
    "vol_z_6",      # z-score объёма текущей свечи относительно 24-свечного окна
    "tbv_ratio",    # taker_buy_volume / volume — давление покупателей (0..1)
    # --- Position relative to extremes ---
    "dist_to_high_24",   # (close - max_high_24) / close  ≤ 0
    "dist_to_low_24",    # (close - min_low_24)  / close  ≥ 0
    "dist_to_high_96",   # то же за 96 свечей (≈1 сутки)
    "dist_to_low_96",
    # --- Range ---
    "range_1",           # (high - low) / close текущей свечи
    "range_avg_24",      # средний range за 24 свечи
    # --- Consecutive candles ---
    "consec_up",         # число подряд идущих up-свечей перед текущей
    "consec_dn",         # число подряд идущих down-свечей перед текущей
    # --- Time ---
    "hour_utc",          # час открытия (0–23)
    "dow",               # день недели (0=Mon, 6=Sun)
]


@dataclass(frozen=True)
class CryptoFeatureVector:
    symbol:    str
    open_time: object           # datetime UTC
    features:  np.ndarray       # shape (1, len(CRYPTO_FEATURE_COLUMNS))
    valid:     bool             # False если меньше min_candles истории


def build_crypto_features(
    candles: Sequence,          # Sequence[CryptoCandle] или pd.DataFrame
    min_candles: int = 50,
) -> CryptoFeatureVector:
    """
    Принимает свечи отсортированные ASC.
    Строит фичи для ПОСЛЕДНЕЙ свечи.
    """
    # ── 1. Нормализация → DataFrame ─────────────────────────────
    if isinstance(candles, pd.DataFrame):
        df = candles.copy()
    else:
        df = pd.DataFrame([{
            "open_time":        c.open_time,
            "open":             c.open,
            "high":             c.high,
            "low":              c.low,
            "close":            c.close,
            "volume":           c.volume,
            "taker_buy_volume": c.taker_buy_volume,
        } for c in candles])

    df = df.sort_values("open_time").reset_index(drop=True)

    if len(df) < min_candles:
        return CryptoFeatureVector(
            symbol="", open_time=None,
            features=np.zeros((1, len(CRYPTO_FEATURE_COLUMNS))),
            valid=False,
        )

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"].fillna(0.0)
    tbv    = df["taker_buy_volume"].fillna(0.0)

    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # ── 2. Returns ───────────────────────────────────────────────
    def safe_ret(n: int) -> float:
        if len(close) <= n:
            return 0.0
        return float(np.log(close.iloc[-1] / close.iloc[-(n + 1)]))

    ret_1  = safe_ret(1)
    ret_3  = safe_ret(3)
    ret_6  = safe_ret(6)
    ret_12 = safe_ret(12)
    ret_24 = safe_ret(24)
    ret_48 = safe_ret(48) if len(close) > 48 else 0.0

    # ── 3. Volatility ────────────────────────────────────────────
    vol_6  = float(log_ret.iloc[-6:].std())  if len(log_ret) >= 6  else 0.0
    vol_24 = float(log_ret.iloc[-24:].std()) if len(log_ret) >= 24 else 0.0
    vol_48 = float(log_ret.iloc[-48:].std()) if len(log_ret) >= 48 else 0.0
    vol_ratio = (vol_6 / vol_48) if vol_48 > 1e-10 else 1.0

    # ── 4. Volume anomaly ────────────────────────────────────────
    vol_window = volume.iloc[-24:]
    vol_mean = float(vol_window.mean())
    vol_std  = float(vol_window.std())
    vol_z_6  = float((volume.iloc[-1] - vol_mean) / (vol_std + 1e-10))

    last_vol = float(volume.iloc[-1])
    last_tbv = float(tbv.iloc[-1])
    tbv_ratio = (last_tbv / last_vol) if last_vol > 1e-10 else 0.5

    # ── 5. Position relative to extremes ────────────────────────
    c_last = float(close.iloc[-1])

    def dist_high(n: int) -> float:
        mx = float(high.iloc[-n:].max()) if len(high) >= n else c_last
        return (c_last - mx) / (c_last + 1e-10)   # ≤ 0

    def dist_low(n: int) -> float:
        mn = float(low.iloc[-n:].min()) if len(low) >= n else c_last
        return (c_last - mn) / (c_last + 1e-10)    # ≥ 0

    dist_h24 = dist_high(24)
    dist_l24 = dist_low(24)
    dist_h96 = dist_high(96) if len(high) >= 96 else dist_h24
    dist_l96 = dist_low(96)  if len(low)  >= 96 else dist_l24

    # ── 6. Range ─────────────────────────────────────────────────
    range_1   = float((high.iloc[-1] - low.iloc[-1]) / (c_last + 1e-10))
    range_avg = float(((high - low) / (close + 1e-10)).iloc[-24:].mean()) \
                if len(df) >= 24 else range_1

    # ── 7. Consecutive candles ───────────────────────────────────
    direction = (df["close"] >= df["open"]).values  # True = up
    consec_up, consec_dn = 0, 0
    for i in range(len(direction) - 2, -1, -1):     # идём назад от предпоследней
        if direction[i]:
            consec_up += 1
        else:
            break
    if consec_up == 0:
        for i in range(len(direction) - 2, -1, -1):
            if not direction[i]:
                consec_dn += 1
            else:
                break

    # ── 8. Time ──────────────────────────────────────────────────
    last_dt  = df["open_time"].iloc[-1]
    hour_utc = int(last_dt.hour)
    dow      = int(last_dt.weekday())   # 0=Mon

    # ── 9. Сборка ────────────────────────────────────────────────
    vec = np.array([[
        ret_1, ret_3, ret_6, ret_12, ret_24, ret_48,
        vol_6, vol_24, vol_48, vol_ratio,
        vol_z_6, tbv_ratio,
        dist_h24, dist_l24, dist_h96, dist_l96,
        range_1, range_avg,
        float(consec_up), float(consec_dn),
        float(hour_utc), float(dow),
    ]], dtype=np.float64)

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    assert vec.shape == (1, len(CRYPTO_FEATURE_COLUMNS)), \
        f"shape {vec.shape} ≠ (1, {len(CRYPTO_FEATURE_COLUMNS)})"

    return CryptoFeatureVector(
        symbol=str(df["symbol"].iloc[0]) if "symbol" in df.columns else "",
        open_time=last_dt,
        features=vec,
        valid=True,
    )
