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
    "vol_trend",    # vol_6 / vol_24 — нарастание краткосрочной волатильности
    # --- Volume anomaly & CVD ---
    "vol_z_1",          # z-score объёма текущей свечи относительно 24-свечного окна
    "taker_buy_ratio",  # taker_buy_volume / volume — давление покупателей (0..1)
    "cvd_1",            # нормализованная дельта за 1 свечу
    "cvd_6",            # накопленный дисбаланс за 6 свечей
    "cvd_trend",        # нарастание/убывание давления (cvd_6 - cvd_1)
    # --- Technical Indicators ---
    "rsi_14",
    "ema_ratio_9_21",
    "bb_width",
    "bb_position",
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
    "consec_down",       # число подряд идущих down-свечей перед текущей
    # --- Time ---
    "hour_utc",          # час открытия (0–23)
    "dow",               # день недели (0=Mon, 6=Sun)
    # --- Funding Rate ---
    "funding_rate",      # текущий 8h funding rate
    "funding_rate_ma3",  # MA(3) последних 3 значений
    "funding_extreme",   # abs(funding_rate) > 0.01 → бинарный маркер перегрева
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
    funding_rate: float = 0.0,
    funding_rate_ma3: float = 0.0,
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
    ret_48 = safe_ret(48)

    # ── 3. Volatility ────────────────────────────────────────────
    vol_6  = float(log_ret.iloc[-6:].std())  if len(log_ret) >= 6  else 0.0
    vol_24 = float(log_ret.iloc[-24:].std()) if len(log_ret) >= 24 else 0.0
    vol_48 = float(log_ret.iloc[-48:].std()) if len(log_ret) >= 48 else 0.0
    vol_ratio = (vol_6 / vol_48) if vol_48 > 1e-10 else 1.0
    vol_trend = (vol_6 / vol_24) if vol_24 > 1e-10 else 1.0

    # ── 4. Volume anomaly ────────────────────────────────────────
    vol_window = volume.iloc[-24:]
    vol_mean = float(vol_window.mean())
    vol_std  = float(vol_window.std())
    vol_z_1  = float((volume.iloc[-1] - vol_mean) / (vol_std + 1e-10))

    last_vol = float(volume.iloc[-1])
    last_tbv = float(tbv.iloc[-1])
    taker_buy_ratio = (last_tbv / last_vol) if last_vol > 1e-10 else 0.5

    all_vol = volume.fillna(0.0)
    all_tbv = tbv.fillna(0.0)
    all_tsell = all_vol - all_tbv
    all_cvd = all_tbv - all_tsell
    cvd_1 = float((all_cvd / (all_vol + 1e-10)).iloc[-1])
    cvd_6_sum = float(all_cvd.iloc[-6:].sum()) if len(all_cvd) >= 6 else float(all_cvd.sum())
    vol_6_sum = float(all_vol.iloc[-6:].sum()) if len(all_vol) >= 6 else float(all_vol.sum())
    cvd_6 = cvd_6_sum / (vol_6_sum + 1e-10)
    cvd_trend = cvd_6 - cvd_1

    # ── 4a. Technical Indicators ─────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / (loss + 1e-10)
    rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1])

    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_ratio_9_21 = float((ema9 / (ema21 + 1e-10)).iloc[-1])

    bb_mean  = close.rolling(20, min_periods=10).mean()
    bb_std   = close.rolling(20, min_periods=10).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    bb_width = float(((bb_upper - bb_lower) / (bb_mean + 1e-10)).iloc[-1])
    bb_position = float(((close - bb_lower) / (bb_upper - bb_lower + 1e-10)).iloc[-1])

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
    dirs = (df["close"] >= df["open"]).values[:-1]  # исключаем текущую (последнюю) свечу
    consec_up = 0
    for d in reversed(dirs):
        if d:
            consec_up += 1
        else:
            break

    consec_down = 0
    for d in reversed(dirs):
        if not d:
            consec_down += 1
        else:
            break

    # ── 8. Time ──────────────────────────────────────────────────
    last_dt  = df["open_time"].iloc[-1]
    hour_utc = int(last_dt.hour)
    dow      = int(last_dt.weekday())   # 0=Mon

    # ── 8a. Funding Rate ─────────────────────────────────────────
    funding_extreme = float(abs(funding_rate) > 0.001)


    # ── 9. Сборка ────────────────────────────────────────────────
    vec = np.array([[
        ret_1, ret_3, ret_6, ret_12, ret_24, ret_48,
        vol_6, vol_24, vol_48, vol_ratio, vol_trend,
        vol_z_1, taker_buy_ratio, cvd_1, cvd_6, cvd_trend,
        rsi_14, ema_ratio_9_21, bb_width, bb_position,
        dist_h24, dist_l24, dist_h96, dist_l96,
        range_1, range_avg,
        float(consec_up), float(consec_down),
        float(hour_utc), float(dow),
        float(funding_rate), float(funding_rate_ma3), float(funding_extreme),
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
    """
    Строит DataFrame с фичами для ВСЕХ свечей (используется при обучении модели).
    Порядок строк: ASC по open_time.
    Колонки: все поля из CRYPTO_FEATURES (тренер) + ret_1 (для таргета/ε-фильтра).

    В отличие от build_crypto_features(), здесь vectorized-вычисления по всему ряду.
    """
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
    out["ret_12"] = np.log(close / close.shift(12))
    out["ret_24"] = np.log(close / close.shift(24))
    out["ret_48"] = np.log(close / close.shift(48))

    # ── Volatility ───────────────────────────────────────────────
    out["vol_6"]   = log_ret.rolling(6,  min_periods=2).std()
    out["vol_24"]  = log_ret.rolling(24, min_periods=6).std()
    out["vol_48"]  = log_ret.rolling(48, min_periods=12).std()
    out["vol_ratio"] = out["vol_6"] / (out["vol_48"] + 1e-10)
    out["vol_trend"] = out["vol_6"] / (out["vol_24"] + 1e-10)

    # ── Volume anomaly & CVD ─────────────────────────────────────
    vol_mean = volume.rolling(24, min_periods=6).mean()
    vol_std  = volume.rolling(24, min_periods=6).std()
    out["vol_z_1"]         = (volume - vol_mean) / (vol_std + 1e-10)
    out["taker_buy_ratio"] = tbv / (volume + 1e-10)

    taker_sell = volume - tbv
    cvd = (tbv - taker_sell)                         # delta за 1 свечу
    out["cvd_1"] = cvd / (volume + 1e-10)            # нормализованная дельта
    out["cvd_6"] = cvd.rolling(6, min_periods=2).sum() / (
        volume.rolling(6, min_periods=2).sum() + 1e-10
    )  # накопленный дисбаланс за 6 свечей
    out["cvd_trend"] = out["cvd_6"] - out["cvd_1"]   # нарастание/убывание давления

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
    out["dist_to_high_96"] = (c - high.rolling(96, min_periods=1).max()) / (c + 1e-10)
    out["dist_to_low_96"]  = (c - low.rolling(96,  min_periods=1).min()) / (c + 1e-10)

    # ── Range ────────────────────────────────────────────────────
    out["range_1"]       = (high - low) / (close + 1e-10)
    out["range_avg_24"]  = out["range_1"].rolling(24, min_periods=6).mean()

    # ── Consecutive candles ──────────────────────────────────────
    direction = (close >= df["open"]).astype(int)  # 1=up, 0=down
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
    out["consec_up"]   = consec_up
    out["consec_down"] = consec_dn

    # ── Time ─────────────────────────────────────────────────────
    dt = pd.to_datetime(df["open_time"])
    out["hour_utc"] = dt.dt.hour.astype(float)
    out["dow"]      = dt.dt.weekday.astype(float)

    # ── Funding Rate ─────────────────────────────────────────────
    out["funding_rate"]     = float(funding_rate)
    out["funding_rate_ma3"] = float(funding_rate_ma3)
    out["funding_extreme"]  = (out["funding_rate"].abs() > 0.001).astype(float)


    # ── NaN → 0 (safety net) ────────────────────────────────────
    out = out.fillna(0.0)
    # Inf → 0
    out = out.replace([np.inf, -np.inf], 0.0)

    return out
