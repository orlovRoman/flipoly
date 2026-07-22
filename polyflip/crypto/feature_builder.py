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
    # --- Volatility ---
    "vol_6",        # std log-returns за 6 свечей
    "vol_24",       # std log-returns за 24 свечи
    "vol_trend",    # vol_6 / vol_24 — нарастание краткосрочной волатильности
    # --- Volume anomaly & CVD ---
    "vol_z_1",          # z-score объёма текущей свечи относительно 24-свечного окна
    "taker_buy_ratio",  # taker_buy_volume / volume — давление покупателей (0..1)
    "cvd_1",            # нормализованная дельта за 1 свечу
    "cvd_6",            # накопленный дисбаланс за 6 свечей
    # --- Technical Indicators ---
    "rsi_14",
    "ema_ratio_9_21",
    "bb_width",
    "bb_position",
    # --- Position relative to extremes ---
    "dist_to_high_24",   # (close - max_high_24) / close  ≤ 0
    "dist_to_low_24",    # (close - min_low_24)  / close  ≥ 0
    # --- Range ---
    "range_1",           # (high - low) / close текущей свечи
    "range_avg_24",      # средний range за 24 свечи
    # --- Consecutive candles ---
    "consec_balance",    # баланс серий свечей (consec_up - consec_down)
    # --- Time (Cyclic) ---
    "hour_sin",          # sin(2*pi*hour/24)
    "hour_cos",          # cos(2*pi*hour/24)
    "dow_sin",           # sin(2*pi*dow/7)
    "dow_cos",           # cos(2*pi*dow/7)
]


@dataclass(frozen=True)
class CryptoFeatureVector:
    symbol:    str
    open_time: object           # datetime UTC
    features:  np.ndarray       # shape (1, len(CRYPTO_FEATURE_COLUMNS))
    valid:     bool             # False если меньше min_candles истории


def build_crypto_features(
    candles: Sequence | pd.DataFrame,
    min_candles: int = 100,
) -> CryptoFeatureVector:
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
            "taker_buy_volume": c.taker_buy_volume if c.taker_buy_volume is not None else 0.0,
        } for c in candles])

    if len(df) < min_candles:
        return CryptoFeatureVector("", None, np.zeros((1, len(CRYPTO_FEATURE_COLUMNS))), False)

    df = df.sort_values("open_time").reset_index(drop=True)

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].fillna(0.0).values
    tbv    = df["taker_buy_volume"].fillna(0.0).values

    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.insert(log_ret, 0, 0.0)

    # ── 1. Returns ───────────────────────────────────────────────
    ret_1  = float(log_ret[-1])
    ret_3  = float(np.log(close[-1] / close[-4]))  if len(close) > 3  else 0.0
    ret_6  = float(np.log(close[-1] / close[-7]))  if len(close) > 6  else 0.0

    # ── 2. Volatility ────────────────────────────────────────────
    vol_6   = float(np.std(log_ret[-6:]))   if len(log_ret) >= 6  else 0.0
    vol_24  = float(np.std(log_ret[-24:]))  if len(log_ret) >= 24 else 0.0
    vol_trend = float(vol_6 / (vol_24 + 1e-10))

    # ── 3. Volume anomaly & CVD ──────────────────────────────────
    v24      = volume[-24:] if len(volume) >= 24 else volume
    vol_mean = float(np.mean(v24))
    vol_std  = float(np.std(v24))
    vol_z_1  = float((volume[-1] - vol_mean) / (vol_std + 1e-10))

    taker_buy_ratio = float(tbv[-1] / (volume[-1] + 1e-10))

    taker_sell = volume - tbv
    cvd_raw    = tbv - taker_sell
    cvd_1      = float(cvd_raw[-1] / (volume[-1] + 1e-10))

    c6_sum    = float(np.sum(cvd_raw[-6:])) if len(cvd_raw) >= 6 else float(np.sum(cvd_raw))
    v6_sum    = float(np.sum(volume[-6:]))  if len(volume)  >= 6 else float(np.sum(volume))
    cvd_6     = float(c6_sum / (v6_sum + 1e-10))

    # ── 4. RSI(14) ───────────────────────────────────────────────
    diffs = np.diff(close[-15:]) if len(close) >= 15 else np.diff(close)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = float(np.mean(gains))  if len(gains)  > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    rs       = avg_gain / (avg_loss + 1e-10)
    rsi_14   = float(100.0 - (100.0 / (1.0 + rs)))

    # ── 5. EMA ratio 9/21 ────────────────────────────────────────
    s = pd.Series(close)
    ema9  = float(s.ewm(span=9,  adjust=False).mean().iloc[-1])
    ema21 = float(s.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_ratio_9_21 = float(ema9 / (ema21 + 1e-10))

    # ── 6. Bollinger Bands (20, 2σ) ──────────────────────────────
    c20      = close[-20:] if len(close) >= 20 else close
    bb_mean  = float(np.mean(c20))
    bb_std   = float(np.std(c20))
    bb_upper = bb_mean + 2.0 * bb_std
    bb_lower = bb_mean - 2.0 * bb_std
    bb_width = float((bb_upper - bb_lower) / (bb_mean + 1e-10))
    bb_position = float((close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10))

    # ── 7. Distance to extremes (24h) ────────────────────────────
    h24 = high[-24:] if len(high) >= 24 else high
    l24 = low[-24:]  if len(low)  >= 24 else low
    max_h24 = float(np.max(h24))
    min_l24 = float(np.min(l24))
    dist_h24 = float((close[-1] - max_h24) / (close[-1] + 1e-10))
    dist_l24 = float((close[-1] - min_l24) / (close[-1] + 1e-10))

    # ── 8. Range ─────────────────────────────────────────────────
    range_1   = float((high[-1] - low[-1]) / (close[-1] + 1e-10))
    h24_arr   = high[-24:] if len(high) >= 24 else high
    l24_arr   = low[-24:]  if len(low)  >= 24 else low
    c24_arr   = close[-24:] if len(close) >= 24 else close
    r24       = (h24_arr - l24_arr) / (c24_arr + 1e-10)
    range_avg = float(np.mean(r24))

    # ── 9. Consecutive candles ───────────────────────────────────
    opens = df["open"].values
    dirs  = (close[:-1] >= opens[:-1])
    dirs_list = list(reversed(dirs))

    consec_up = 0
    for d in dirs_list:
        if d:
            consec_up += 1
        else:
            break

    consec_down = 0
    for d in dirs_list:
        if not d:
            consec_down += 1
        else:
            break
    consec_balance = float(consec_up - consec_down)

    # ── 10. Time (Cyclic) ────────────────────────────────────────
    last_dt  = df["open_time"].iloc[-1]
    hour_sin = float(np.sin(2 * np.pi * last_dt.hour / 24))
    hour_cos = float(np.cos(2 * np.pi * last_dt.hour / 24))
    dow_sin  = float(np.sin(2 * np.pi * last_dt.weekday() / 7))
    dow_cos  = float(np.cos(2 * np.pi * last_dt.weekday() / 7))

    # ── 11. Сборка ───────────────────────────────────────────────
    vec = np.array([[
        ret_1, ret_3, ret_6,
        vol_6, vol_24, vol_trend,
        vol_z_1, taker_buy_ratio, cvd_1, cvd_6,
        rsi_14, ema_ratio_9_21, bb_width, bb_position,
        dist_h24, dist_l24,
        range_1, range_avg,
        consec_balance,
        hour_sin, hour_cos, dow_sin, dow_cos,
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


def build_features(
    candles: Sequence | pd.DataFrame,
    funding_rate: float = 0.0,
    funding_rate_ma3: float = 0.0,
) -> pd.DataFrame:
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
            "taker_buy_volume": c.taker_buy_volume if c.taker_buy_volume is not None else 0.0,
        } for c in candles])

    df = df.sort_values("open_time").reset_index(drop=True)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"].fillna(0.0)
    tbv    = df["taker_buy_volume"].fillna(0.0)

    log_ret = np.log(close / close.shift(1))

    # ── Returns ──────────────────────────────────────────────────
    out = pd.DataFrame(index=df.index)
    out["open_time"] = df["open_time"]
    out["ret_1"]  = log_ret
    out["ret_3"]  = np.log(close / close.shift(3))
    out["ret_6"]  = np.log(close / close.shift(6))

    # ── Volatility ───────────────────────────────────────────────
    out["vol_6"]   = log_ret.rolling(6,  min_periods=2).std()
    out["vol_24"]  = log_ret.rolling(24, min_periods=6).std()
    out["vol_trend"] = out["vol_6"] / (out["vol_24"] + 1e-10)

    # ── Volume anomaly & CVD ─────────────────────────────────────
    vol_mean = volume.rolling(24, min_periods=6).mean()
    vol_std  = volume.rolling(24, min_periods=6).std()
    out["vol_z_1"]         = (volume - vol_mean) / (vol_std + 1e-10)
    out["taker_buy_ratio"] = tbv / (volume + 1e-10)

    taker_sell = volume - tbv
    cvd = (tbv - taker_sell)
    out["cvd_1"] = cvd / (volume + 1e-10)
    out["cvd_6"] = cvd.rolling(6, min_periods=2).sum() / (
        volume.rolling(6, min_periods=2).sum() + 1e-10
    )

    # ── RSI(14) ──────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / (loss + 1e-10)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── EMA ratio 9/21 ───────────────────────────────────────────
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["ema_ratio_9_21"] = ema9 / (ema21 + 1e-10)

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────
    bb_mean  = close.rolling(20, min_periods=10).mean()
    bb_std   = close.rolling(20, min_periods=10).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_mean + 1e-10)
    bb_pos   = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    out["bb_width"]    = bb_width
    out["bb_position"] = bb_pos

    # ── Distance to extremes ─────────────────────────────────────
    c = close
    out["dist_to_high_24"] = (c - high.rolling(24, min_periods=1).max()) / (c + 1e-10)
    out["dist_to_low_24"]  = (c - low.rolling(24,  min_periods=1).min()) / (c + 1e-10)

    # ── Range ────────────────────────────────────────────────────
    out["range_1"]       = (high - low) / (close + 1e-10)
    out["range_avg_24"]  = out["range_1"].rolling(24, min_periods=6).mean()

    # ── Consecutive candles ──────────────────────────────────────
    direction = (close >= df["open"]).astype(int)
    direction_shifted = direction.shift(1).fillna(0).astype(int)
    consec_up = []
    consec_dn = []
    cu = 0
    cd = 0
    for d in direction_shifted:
        if d == 1:
            cu += 1
            cd = 0
        else:
            cd += 1
            cu = 0
        consec_up.append(cu)
        consec_dn.append(cd)
    out["consec_balance"] = [float(u - d) for u, d in zip(consec_up, consec_dn)]

    # ── Time (Cyclic) ────────────────────────────────────────────
    dt = pd.to_datetime(df["open_time"])
    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["dow_sin"]  = np.sin(2 * np.pi * dt.dt.weekday / 7)
    out["dow_cos"]  = np.cos(2 * np.pi * dt.dt.weekday / 7)

    # ── NaN → 0 (safety net) ────────────────────────────────────
    out = out.fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0)

    return out
