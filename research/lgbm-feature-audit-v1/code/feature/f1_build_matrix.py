"""f1_build_matrix.py — Build 79-feature matrix for F-series audit.

Stages:
  1. Load decision rows (shared_ds with contract_target)
  2. Per-symbol: 15m candle → build_features + sequence + extended → parquet
  3. All snapsdec → derived + lags → parquet
  4. As-of merge: candle (per asset) + snapshot (per market) → decision rows
  5. Finalize: PM context, time, strike overrides, funding=NaN → save
"""
import os, sys
import warnings
import numpy as np
import pandas as pd
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "exp")
OUT = os.path.join(ROOT, "out", "feature")
CANDLES_PATH = os.path.join(DATA, "full", "candles.csv.gz")
SNAPS_DIR = os.path.join(DATA, "snapsdec")
SHARED_DS = os.path.join(ROOT, "out", "shared_ds", "shared_dataset.parquet")
MATRIX_PATH = os.path.join(OUT, "FEATURE_MATRIX.parquet")

ASSET2SYM = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
             "XRP": "XRPUSDT", "DOGE": "DOGEUSDT"}
SYM2ASSET = {v: k for k, v in ASSET2SYM.items()}

# ── Sequence feature formulas (inline, no imports from polyflip) ──────────────
NEUTRAL_RETURN_THRESHOLD = 0.0002
SEQ_CAP = 8

def _run_length_series(direction: pd.Series, expected: int, cap: int = SEQ_CAP) -> pd.Series:
    matches = direction.fillna(0).astype(int).eq(expected)
    reset_groups = (~matches).cumsum()
    return (matches.astype(int).groupby(reset_groups, sort=False)
            .cumsum().clip(upper=cap).astype(float))

def _alternation_rate(values):
    if len(values) < 2:
        return np.nan
    transitions = []
    for prev, cur in zip(values[:-1], values[1:]):
        if prev == 0 or cur == 0:
            transitions.append(0.0)
        else:
            transitions.append(float(prev != cur))
    return float(np.mean(transitions)) if transitions else np.nan

def _signed_efficiency(values):
    denom = float(np.abs(values).sum())
    return float(values.sum() / denom) if denom > 1e-12 else 0.0


# ── Stage 1: Load decisions ───────────────────────────────────────────────────
def load_decisions():
    ds = pd.read_parquet(SHARED_DS, columns=[
        "decision_event_id", "market_id", "asset", "fold", "decision_at",
        "contract_target", "flip_native", "synthetic_no",
    ])
    ds = ds.dropna(subset=["decision_at", "market_id"]).reset_index(drop=True)
    ds["decision_at"] = pd.to_datetime(ds["decision_at"], utc=True, errors="coerce")
    ds["market_id"] = ds["market_id"].astype(str)
    print(f"[stage1] decisions: {len(ds)}  markets: {ds['market_id'].nunique()}")
    return ds


# ── Stage 2: Per-symbol candle features ───────────────────────────────────────
def build_candle_features(symbol: str) -> pd.DataFrame:
    """Compute all candle-derived features for one symbol at closed 15m bars."""
    df = pd.read_csv(CANDLES_PATH, usecols=[
        "symbol", "interval", "open_time", "open", "high", "low", "close",
        "volume", "taker_buy_volume", "close_time", "is_closed",
    ])
    df = df[(df["symbol"] == symbol) & (df["interval"] == "15m") & (df["is_closed"] == "t")]
    for c in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["open_time", "close_time"]).sort_values("close_time").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    tbv = df["taker_buy_volume"]
    open_ = df["open"]

    out = pd.DataFrame(index=df.index)
    out["close_time"] = df["close_time"]

    # ── Returns ──
    log_ret = np.log(close / close.shift(1))
    out["ret_1"] = log_ret
    out["ret_3"] = np.log(close / close.shift(3))
    out["ret_6"] = np.log(close / close.shift(6))
    out["ret_12"] = np.log(close / close.shift(12))
    out["ret_24"] = np.log(close / close.shift(24))
    out["ret_48"] = np.log(close / close.shift(48))

    # ── Volatility ──
    out["vol_6"] = log_ret.rolling(6, min_periods=2).std(ddof=1)
    out["vol_24"] = log_ret.rolling(24, min_periods=6).std(ddof=1)
    out["vol_48"] = log_ret.rolling(48, min_periods=6).std(ddof=1)
    out["vol_trend"] = out["vol_6"] / (out["vol_24"] + 1e-10)

    # ── Volume anomaly ──
    v_mean24 = volume.rolling(24, min_periods=6).mean()
    v_std24 = volume.rolling(24, min_periods=6).std(ddof=1)
    out["vol_z_1"] = (volume - v_mean24) / (v_std24 + 1e-10)
    v_mean6 = volume.rolling(6, min_periods=2).mean()
    v_std6 = volume.rolling(6, min_periods=2).std(ddof=1)
    out["vol_z_6"] = (volume - v_mean6) / (v_std6 + 1e-10)
    out["vol_ratio"] = out["vol_6"] / (out["vol_48"] + 1e-10)

    # ── CVD ──
    taker_sell = volume - tbv
    cvd = tbv - taker_sell
    out["cvd_1"] = cvd / (volume + 1e-10)
    out["cvd_6"] = cvd.rolling(6, min_periods=2).sum() / (volume.rolling(6, min_periods=2).sum() + 1e-10)
    out["cvd_trend"] = out["cvd_6"] / (out["cvd_6"].shift(3) + 1e-10)

    # ── Taker ──
    out["taker_buy_ratio"] = tbv / (volume + 1e-10)

    # ── RSI ──
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── EMA ──
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["ema_ratio_9_21"] = ema9 / (ema21 + 1e-10)

    # ── Bollinger ──
    bb_mean = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std(ddof=1)
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    out["bb_width"] = (bb_upper - bb_lower) / (bb_mean + 1e-10)
    out["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # ── Extremes ──
    out["dist_to_high_24"] = (close - high.rolling(24, min_periods=1).max()) / (close + 1e-10)
    out["dist_to_low_24"] = (close - low.rolling(24, min_periods=1).min()) / (close + 1e-10)
    out["dist_to_high_96"] = (close - high.rolling(96, min_periods=1).max()) / (close + 1e-10)
    out["dist_to_low_96"] = (close - low.rolling(96, min_periods=1).min()) / (close + 1e-10)

    # ── Range ──
    out["range_1"] = (high - low) / (close + 1e-10)
    r24 = (high - low) / (close + 1e-10)
    out["range_avg_24"] = r24.rolling(24, min_periods=6).mean()

    # ── Consecutive ──
    up = close >= open_
    previous = up.shift(1)
    run_groups = previous.ne(previous.shift()).cumsum()
    run_lengths = previous.groupby(run_groups).cumcount().add(1)
    consec = np.where(previous.eq(1), run_lengths,
                      np.where(previous.eq(0), -run_lengths, 0))
    out["consec_balance"] = consec.astype(float)
    # run lengths for consec_up / consec_down
    is_up = up.astype(int)
    up_groups = (~is_up.astype(bool)).cumsum()
    consec_up_counts = is_up.groupby(up_groups).cumsum().astype(float)
    is_down = (~up).astype(int)
    down_groups = up.astype(bool).cumsum()
    consec_down_counts = is_down.groupby(down_groups).cumsum().astype(float)
    out["consec_up"] = consec_up_counts
    out["consec_down"] = consec_down_counts

    # ── Sequence features (on DataFrame) ──
    candle_return = close / open_ - 1.0
    direction = pd.Series(
        np.where(candle_return.abs() < NEUTRAL_RETURN_THRESHOLD, 0.0,
                 np.sign(candle_return)),
        index=df.index, dtype=float
    ).fillna(0.0)
    out["direction_lag_1"] = direction
    out["direction_lag_2"] = direction.shift(1)
    out["direction_lag_3"] = direction.shift(2)
    out["consecutive_up"] = _run_length_series(direction, 1)
    out["consecutive_down"] = _run_length_series(direction, -1)
    out["up_ratio_4"] = direction.eq(1).astype(float).rolling(4, min_periods=4).mean()
    out["alternation_rate_6"] = direction.rolling(6, min_periods=6).apply(_alternation_rate, raw=True)
    out["signed_trend_efficiency_6"] = candle_return.rolling(6, min_periods=6).apply(_signed_efficiency, raw=True)
    out["signed_body_pct"] = candle_return
    candle_range = (high - low).abs()
    out["body_to_range"] = ((close - open_).abs() / candle_range.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    # ── Time from candle close_time ──
    dt = df["close_time"]
    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.weekday / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.weekday / 7)

    # ── Strike features (placeholder, overridden in finalize) ──
    out["strike_gap_pct"] = 0.0
    out["log_moneyness"] = 0.0

    # ── PM features (placeholder, overridden in finalize) ──
    for c in ["pm_momentum_5m", "pm_volume_5m", "pm_spread_pct", "pm_quote_pressure",
              "pm_best_bid", "pm_best_ask"]:
        out[c] = 0.0

    # ── Safety ──
    out = out.replace([np.inf, -np.inf], 0.0)
    out = out.fillna(0.0)
    return out


# ── Stage 3: Snapshot features ────────────────────────────────────────────────
def build_snapshot_features():
    """Compute all snapshot-derived features from snapsdec files."""
    chunks = []
    usecols = ["market_id", "recorded_at", "mid_price", "spread", "volume_5min",
               "price_velocity", "best_bid", "best_ask", "strike_value",
               "time_left_seconds"]
    for fn in sorted(os.listdir(SNAPS_DIR)):
        if not fn.endswith(".csv.gz"):
            continue
        df = pd.read_csv(os.path.join(SNAPS_DIR, fn), usecols=usecols)
        chunks.append(df)
    sn = pd.concat(chunks, ignore_index=True)
    sn["market_id"] = sn["market_id"].astype(str)
    sn["recorded_at"] = pd.to_datetime(sn["recorded_at"], utc=True, errors="coerce")
    for c in ["mid_price", "spread", "volume_5min", "price_velocity",
              "best_bid", "best_ask", "strike_value", "time_left_seconds"]:
        sn[c] = pd.to_numeric(sn[c], errors="coerce")
    sn = sn.dropna(subset=["recorded_at", "market_id"]).sort_values(
        ["market_id", "recorded_at"]).reset_index(drop=True)
    print(f"[stage3] snapshots: {len(sn)}  markets: {sn['market_id'].nunique()}")

    sn["time_left_min"] = sn["time_left_seconds"].fillna(0.0) / 60.0

    # ── add_derived_features (inline) ──
    sn["price_deviation"] = (sn["mid_price"] - 0.5).abs()
    sn["spread_pct"] = (sn["spread"] / (sn["mid_price"] + 1e-6)).clip(upper=10.0)
    sn["log_time_left"] = np.log1p(sn["time_left_min"].clip(lower=0.0))
    sn["deviation_x_time"] = sn["price_deviation"] * sn["time_left_min"]
    sn["price_deviation_sq"] = sn["price_deviation"] ** 2
    sn["day_of_week"] = sn["recorded_at"].dt.weekday.astype(float)

    # price_distance_from_max (expanding per market)
    grp = sn.groupby("market_id", sort=False)["mid_price"]
    sn["price_distance_from_max"] = (
        grp.transform(lambda x: x.expanding().max()) - sn["mid_price"]
    ).clip(lower=0.0)

    # time_phase
    max_tl = sn.groupby("market_id", sort=False)["time_left_min"].transform("max").clip(lower=15.0)
    sn["time_phase"] = (sn["time_left_min"] / max_tl).clip(0, 1)

    # interactions
    velocity = sn["price_velocity"].fillna(0.0)
    sn["velocity_x_phase"] = velocity * (1.0 - sn["time_phase"])
    sn["dev_sq_x_phase"] = sn["price_deviation_sq"] * (1.0 - sn["time_phase"])
    sn["is_final_phase"] = (sn["time_phase"] <= 0.20).astype(float)
    sn["high_price_final"] = sn["price_deviation"] * (1.0 - sn["time_phase"])

    # ── add_lag_features (inline) ──
    g = sn.groupby("market_id", sort=False)
    sn["price_velocity_lag1"] = g["price_velocity"].shift(1)
    sn["price_velocity_lag1"] = sn.groupby("market_id", sort=False)["price_velocity_lag1"].ffill().fillna(0.0)
    sn["price_momentum"] = sn["mid_price"] - g["mid_price"].shift(3)
    spread_lag = g["spread"].shift(6)
    sn["spread_trend"] = (sn["spread"] / (spread_lag + 1e-8)).clip(upper=10.0)
    vol_lag = g["volume_5min"].shift(3)
    sn["volume_trend"] = (sn["volume_5min"] / (vol_lag + 1e-8)).clip(upper=10.0)

    # PM context features
    sn["pm_momentum_5m"] = sn["price_velocity"].fillna(0.0)
    sn["pm_volume_5m"] = sn["volume_5min"].fillna(0.0).clip(lower=0.0)
    sn["pm_spread_pct"] = (sn["spread"] / (sn["mid_price"] + 1e-9)).clip(lower=0.0)
    sn["pm_quote_pressure"] = sn["mid_price"] - 0.5
    sn["pm_best_bid"] = sn["best_bid"].fillna(0.0).clip(lower=0.0)
    sn["pm_best_ask"] = sn["best_ask"].fillna(0.0).clip(lower=0.0)

    keep = ["market_id", "recorded_at", "mid_price", "spread", "volume_5min",
            "price_velocity", "time_left_min",
            "price_deviation", "spread_pct", "log_time_left",
            "deviation_x_time", "price_deviation_sq",
            "price_distance_from_max", "time_phase",
            "velocity_x_phase", "dev_sq_x_phase",
            "is_final_phase", "high_price_final",
            "day_of_week",
            "price_momentum", "spread_trend", "volume_trend", "price_velocity_lag1",
            "pm_momentum_5m", "pm_volume_5m", "pm_spread_pct", "pm_quote_pressure",
            "pm_best_bid", "pm_best_ask",
            "strike_value"]
    return sn[keep]


# ── Stage 4: As-of merge ─────────────────────────────────────────────────────
CANDLE_FEAT_COLS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    "vol_6", "vol_24", "vol_48", "vol_trend", "vol_ratio", "vol_z_1", "vol_z_6",
    "cvd_1", "cvd_6", "cvd_trend", "taker_buy_ratio",
    "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position",
    "dist_to_high_24", "dist_to_low_24", "dist_to_high_96", "dist_to_low_96",
    "range_1", "range_avg_24",
    "consec_balance", "consec_up", "consec_down",
    "direction_lag_1", "direction_lag_2", "direction_lag_3",
    "consecutive_up", "consecutive_down",
    "up_ratio_4", "alternation_rate_6",
    "signed_trend_efficiency_6", "signed_body_pct", "body_to_range",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "strike_gap_pct", "log_moneyness",
    "pm_momentum_5m", "pm_volume_5m", "pm_spread_pct", "pm_quote_pressure",
    "pm_best_bid", "pm_best_ask",
]
SNAP_FEAT_COLS = [
    "mid_price", "spread", "volume_5min", "price_velocity", "time_left_min",
    "price_deviation", "spread_pct", "log_time_left",
    "deviation_x_time", "price_deviation_sq",
    "price_distance_from_max", "time_phase",
    "velocity_x_phase", "dev_sq_x_phase",
    "is_final_phase", "high_price_final", "day_of_week",
    "price_momentum", "spread_trend", "volume_trend", "price_velocity_lag1",
    "pm_momentum_5m", "pm_volume_5m", "pm_spread_pct", "pm_quote_pressure",
    "pm_best_bid", "pm_best_ask",
    "strike_value",
]


def asof_merge_candle(ds: pd.DataFrame, sym_frames: dict) -> pd.DataFrame:
    """Merge candle features to decisions via as-of (nearest closed candle <= decision_at)."""
    results = []
    for sym, frame in sym_frames.items():
        asset = SYM2ASSET.get(sym, sym)
        sub = ds[ds["asset"] == asset].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("decision_at").reset_index(drop=True)
        frame_s = frame.sort_values("close_time").reset_index(drop=True)
        merged = pd.merge_asof(
            sub, frame_s[CANDLE_FEAT_COLS + ["close_time"]],
            left_on="decision_at", right_on="close_time",
            direction="backward", allow_exact_matches=True,
        )
        results.append(merged)
    out = pd.concat(results, ignore_index=True)
    # check coverage
    candle_covered = out["close_time"].notna().sum()
    print(f"[stage4a] candle coverage: {candle_covered}/{len(out)} = {100*candle_covered/len(out):.2f}%")
    return out


def asof_merge_snapshot(ds: pd.DataFrame, snap_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge snapshot features to decisions via per-market as-of (nearest snapshot <= decision_at).

    Vectorized per-market searchsorted (f0-validated); avoids the full-size
    merge_asof that timed out. Rows in markets absent from snapsdec stay NaN.
    """
    feat_cols = [c for c in SNAP_FEAT_COLS if c in snap_frame.columns]

    # drop candle placeholder pm_* cols so real snapshot pm values are not
    # suffix-collided away (pm_* exist both in CANDLE_FEAT_COLS and SNAP_FEAT_COLS)
    pm_placeholders = ["pm_momentum_5m", "pm_volume_5m", "pm_spread_pct",
                       "pm_quote_pressure", "pm_best_bid", "pm_best_ask"]
    for c in pm_placeholders:
        if c in ds.columns:
            ds = ds.drop(columns=[c])

    inter = set(ds["market_id"].unique()) & set(snap_frame["market_id"].unique())
    dsi = ds[ds["market_id"].isin(inter)]
    sni = snap_frame[snap_frame["market_id"].isin(inter)].sort_values(["market_id", "recorded_at"])

    # carry forward per market: one groupby-dict, one vectorized numpy write per market
    out_cols = ["recorded_at"] + feat_cols
    sn_by_market = {mk: g for mk, g in sni.groupby("market_id", sort=False)}
    arr_out = np.full((len(ds), len(out_cols)), np.nan)
    col_idx = {c: i for i, c in enumerate(out_cols)}
    lag_sec = []
    for mk, g in dsi.groupby("market_id", sort=True):
        sg = sn_by_market[mk]
        sg_vals = sg[out_cols].to_numpy(dtype=np.float64)
        rt = sg["recorded_at"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").values.astype("int64")
        lt = g["decision_at"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").values.astype("int64")
        idx = np.searchsorted(rt, lt) - 1
        ok = idx >= 0
        idxc = np.clip(idx, 0, None)
        rows = g.index.to_numpy(dtype=np.int64)
        arr_out[rows[ok], :] = sg_vals[idxc[ok], :]
        if ok.any():
            lag_sec.extend((lt[ok] - rt[idxc[ok]]).astype(float) / 1e9)
    merged = pd.DataFrame(arr_out, index=ds.index, columns=out_cols)
    # arr_out stores datetime cols as float epoch µs (to_numpy); convert to ns
    merged["recorded_at"] = pd.to_datetime(merged["recorded_at"] * 1e3, unit="ns", utc=True)
    for c in out_cols:
        ds[c] = merged[c].values
    lag_sec = np.asarray(lag_sec)
    snap_covered = ds["recorded_at"].notna().sum()
    print(f"[stage4b] snapshot coverage: {snap_covered}/{len(ds)} = {100*snap_covered/len(ds):.2f}%"
          f"  lag median={np.median(lag_sec):.0f}s  pct<300={float((lag_sec <= 300).mean()):.3f}")
    return ds


# ── Stage 5: Finalize ─────────────────────────────────────────────────────────
CATALOG_79 = [
    "alternation_rate_6", "bb_position", "bb_width", "body_to_range",
    "consec_balance", "consec_down", "consec_up", "consecutive_down", "consecutive_up",
    "cvd_1", "cvd_6", "cvd_trend", "day_of_week",
    "dev_sq_x_phase", "deviation_x_time",
    "direction_lag_1", "direction_lag_2", "direction_lag_3",
    "dist_to_high_24", "dist_to_high_96", "dist_to_low_24", "dist_to_low_96",
    "dow", "dow_cos", "dow_sin",
    "ema_ratio_9_21",
    "funding_extreme", "funding_rate", "funding_rate_ma3",
    "high_price_final", "hour_cos", "hour_of_day", "hour_sin", "hour_utc",
    "is_final_phase", "log_moneyness", "log_time_left",
    "mid_price",
    "pm_best_ask", "pm_best_bid", "pm_momentum_5m", "pm_quote_pressure",
    "pm_spread_pct", "pm_volume_5m",
    "price_deviation", "price_deviation_sq", "price_distance_from_max",
    "price_momentum", "price_velocity", "price_velocity_lag1",
    "range_1", "range_avg_24",
    "ret_1", "ret_12", "ret_24", "ret_3", "ret_48", "ret_6",
    "rsi_14",
    "signed_body_pct", "signed_trend_efficiency_6",
    "spread", "spread_pct", "spread_trend",
    "strike_gap_pct", "taker_buy_ratio",
    "time_left_min", "time_phase",
    "up_ratio_4", "velocity_x_phase",
    "vol_24", "vol_48", "vol_6", "vol_ratio", "vol_trend", "vol_z_1", "vol_z_6",
    "volume_5min", "volume_trend",
]

TARGET_COLS = ["decision_event_id", "market_id", "asset", "fold", "decision_at",
               "contract_target", "flip_native", "synthetic_no"]


def finalize(merged: pd.DataFrame) -> pd.DataFrame:
    """Add time features, override strike, ensure all 79 columns exist."""
    out = merged

    # ── Time features from decision_at ──
    dt = out["decision_at"]
    out["hour_utc"] = dt.dt.hour.astype(float)
    out["hour_of_day"] = dt.dt.hour.astype(float)
    out["dow"] = dt.dt.weekday.astype(float)
    # day_of_week already from snapshot, but override with decision time for consistency
    out["day_of_week"] = dt.dt.weekday.astype(float)

    # ── Override strike features from snapshot strike_value + candle context ──
    # strike_value from snapshot as-of; binaries: underlying ≈ pm mid_price.
    # Rows without snapshot strike (or strike<=0) stay NaN (no fabricated 0.0).
    strike = out["strike_value"]
    mid = out["mid_price"]
    valid = strike.notna() & (strike > 0.0) & mid.notna()
    out["strike_gap_pct"] = np.nan
    out["log_moneyness"] = np.nan
    out.loc[valid, "strike_gap_pct"] = (mid[valid] - strike[valid]) / strike[valid]
    out.loc[valid, "log_moneyness"] = np.log(mid[valid] / strike[valid])
    out.drop(columns=["strike_value"], inplace=True, errors="ignore")

    # ── Funding: no data source → NaN ──
    out["funding_rate"] = np.nan
    out["funding_rate_ma3"] = np.nan
    out["funding_extreme"] = np.nan

    # ── Ensure all 79 columns exist ──
    for col in CATALOG_79:
        if col not in out.columns:
            out[col] = np.nan
            print(f"  [warn] missing column added as NaN: {col}")

    # ── Reorder ──
    final_cols = TARGET_COLS + CATALOG_79
    out = out[final_cols]

    # ── Final safety: in-place inf→nan (memory-lean) ──
    vals = out[CATALOG_79].values
    isf = np.isinf(vals)
    if isf.any():
        vals[isf] = np.nan
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)

    # Stage 1
    ds = load_decisions()

    # Stage 2: per-symbol candle features
    sym_frames = {}
    for sym in ASSET2SYM.values():
        cache = os.path.join(OUT, f"_candle_features_{sym}.parquet")
        if os.path.exists(cache):
            sym_frames[sym] = pd.read_parquet(cache)
            print(f"[stage2] {sym}: loaded cache ({len(sym_frames[sym])})")
        else:
            sym_frames[sym] = build_candle_features(sym)
            sym_frames[sym].to_parquet(cache, index=False)
            print(f"[stage2] {sym}: built ({len(sym_frames[sym])})")

    # Stage 3: snapshot features
    snap_cache = os.path.join(OUT, "_snapshot_features.parquet")
    if os.path.exists(snap_cache):
        snap_frame = pd.read_parquet(snap_cache)
        print(f"[stage3] loaded cache ({len(snap_frame)})")
    else:
        snap_frame = build_snapshot_features()
        snap_frame.to_parquet(snap_cache, index=False)
        print(f"[stage3] built ({len(snap_frame)})")

    # Stage 4a: candle as-of
    ds = asof_merge_candle(ds, sym_frames)

    # Stage 4b: snapshot as-of
    ds = asof_merge_snapshot(ds, snap_frame)

    # Stage 5: finalize
    matrix = finalize(ds)
    matrix.to_parquet(MATRIX_PATH, index=False)
    print(f"[done] saved {MATRIX_PATH}  shape={matrix.shape}")
    print(f"  catalog cols present: {sum(c in matrix.columns for c in CATALOG_79)}/79")
    print(f"  nan rates top-10:")
    nan_rates = matrix[CATALOG_79].isna().mean().sort_values(ascending=False)
    for feat, rate in nan_rates.head(10).items():
        print(f"    {feat}: {100*rate:.1f}%")


if __name__ == "__main__":
    main()
