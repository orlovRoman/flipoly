"""
Rule-based market regime features — pure, deterministic, no DB reads.

Computes per-asset and global basket features from closed 15m candles
for regime classification (T04 of MRF plan).

Horizons: 4h (16 candles), 12h (48 candles), 24h (96 candles).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd

# ── Horizon constants (15m candles) ───────────────────────────
CANDLES_PER_HOUR = 4
HORIZON_4H = 16   # 4 * 4
HORIZON_12H = 48  # 12 * 4
HORIZON_24H = 96  # 24 * 4

# Minimum candles required for a valid feature set
# Volatility needs HORIZON_24H log-returns = HORIZON_24H+1 closes
MIN_HISTORY_CANDLES = HORIZON_24H + 1  # 97

# Maximum expected time span for MIN_HISTORY_CANDLES closed 15m candles
# Allow 25h to tolerate minor gaps without failing
_MAX_CANDLE_SPAN = timedelta(hours=25)


def validate_candle_continuity(
    open_times: Sequence[datetime],
    expected_count: int,
) -> tuple[bool, str]:
    """
    Check that candles cover the expected time range without large gaps.

    Args:
        open_times: sorted open_time of each candle (oldest first)
        expected_count: how many candles we expect for the features

    Returns:
        (is_valid, reason). is_valid=True means no problematic gaps.
    """
    if len(open_times) < 2:
        return True, "insufficient_candles_for_validation"

    # Check for duplicates
    unique_times = set(open_times)
    if len(unique_times) != len(open_times):
        return False, f"duplicates:{len(open_times) - len(unique_times)}"

    # Check count matches
    if expected_count > 0 and len(open_times) < expected_count:
        return False, f"count_mismatch:{len(open_times)}/{expected_count}"

    # Check monotonic ordering
    for i in range(1, len(open_times)):
        if open_times[i] <= open_times[i - 1]:
            return False, f"not_sorted_at_{i}"

    span = open_times[-1] - open_times[0]
    if span > _MAX_CANDLE_SPAN:
        return False, f"span_exceeded:{span.total_seconds()/3600:.1f}h"

    # Check individual gaps > 30 minutes (allow 1 missed candle)
    for i in range(1, len(open_times)):
        gap = open_times[i] - open_times[i - 1]
        if gap > timedelta(minutes=30):
            return False, f"gap_at_index_{i}:{gap.total_seconds()/60:.0f}min"

    return True, "ok"


@dataclass(frozen=True)
class AssetRegimeFeatures:
    """Per-asset features for regime classification."""
    symbol: str
    # Returns at horizons (log-returns)
    ret_4h: float = 0.0
    ret_12h: float = 0.0
    ret_24h: float = 0.0
    # Volatility (annualized from 15m std)
    vol_4h: float = 0.0
    vol_24h: float = 0.0
    vol_ratio: float = 0.0  # vol_4h / vol_24h
    # Trend efficiency ratio (directional / total movement)
    efficiency_24h: float = 0.0
    # Range ratio (high-low / close)
    range_ratio_24h: float = 0.0
    # Candle direction stats (last 24h)
    up_ratio_24h: float = 0.0
    # Number of candles available
    candle_count: int = 0
    # Whether this asset has enough history
    history_ready: bool = False
    # Computed strength (0.0-1.0)
    strength_score: float = 0.0


@dataclass(frozen=True)
class BasketRegimeFeatures:
    """Cross-asset basket features for global regime."""
    # Median returns at horizons
    median_ret_4h: float = 0.0
    median_ret_12h: float = 0.0
    median_ret_24h: float = 0.0
    # Breadth: fraction of assets with positive return
    breadth_up_4h: float = 0.0
    breadth_up_12h: float = 0.0
    breadth_up_24h: float = 0.0
    # Cross-asset dispersion (std of returns)
    dispersion_4h: float = 0.0
    dispersion_24h: float = 0.0
    # Market efficiency (abs(median_ret) / median_vol)
    market_efficiency_24h: float = 0.0
    # Number of assets with ready history
    ready_count: int = 0
    total_count: int = 0
    # Overall history readiness
    history_ready: bool = False
    # Computed global strength
    strength: float = 0.0


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    """Complete regime feature snapshot at a point in time."""
    as_of: datetime  # UTC
    assets: dict[str, AssetRegimeFeatures] = field(default_factory=dict)
    basket: BasketRegimeFeatures = field(default_factory=BasketRegimeFeatures)
    reason_codes: list[str] = field(default_factory=list)


def _log_returns(closes: np.ndarray, periods: int) -> float:
    """Log return over `periods` intervals (requires periods+1 closes)."""
    if len(closes) < periods + 1:
        return 0.0
    base = closes[-periods - 1]
    r = math.log(closes[-1] / base) if base > 0 else 0.0
    return r if math.isfinite(r) else 0.0


def _volatility(returns: np.ndarray) -> float:
    """Standard deviation of returns. Returns 0.0 if insufficient data."""
    if len(returns) < 2:
        return 0.0
    v = float(np.std(returns, ddof=1))
    return v if math.isfinite(v) else 0.0


def _efficiency_ratio(closes: np.ndarray) -> float:
    """
    Trend efficiency ratio: |net_direction| / total_movement.
    Ranges from 0 (pure noise) to 1 (straight line).
    """
    if len(closes) < 2:
        return 0.0
    net = abs(closes[-1] - closes[0])
    total = np.sum(np.abs(np.diff(closes)))
    if total < 1e-10:
        return 0.0
    r = net / total
    return r if math.isfinite(r) else 0.0


def compute_asset_strength(
    ret_4h: float,
    ret_12h: float,
    ret_24h: float,
    efficiency_24h: float,
    ret_norm_cap: float = 0.10,
) -> float:
    """Compute strength score for an asset (0.0-1.0)."""
    def _norm(r, cap):
        if cap <= 0:
            return 0.0
        return math.tanh(abs(r) / cap)
    magnitude = 0.50 * _norm(ret_4h, ret_norm_cap) + 0.30 * _norm(ret_12h, ret_norm_cap) + 0.20 * _norm(ret_24h, ret_norm_cap)
    rets = [ret_4h, ret_12h, ret_24h]
    n_pos = sum(1 for r in rets if r > 0)
    n_neg = sum(1 for r in rets if r < 0)
    consistency = max(n_pos, n_neg) / len(rets)
    return magnitude * (0.5 + 0.5 * efficiency_24h) * consistency


def compute_asset_features(
    symbol: str,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    candle_count: int,
) -> AssetRegimeFeatures:
    """
    Compute per-asset regime features from OHLC arrays (oldest → newest).

    All inputs are raw numpy arrays. The caller must ensure:
    - Arrays are sorted by open_time ascending
    - Only closed candles are included (no forming bars)
    - Arrays have the same length
    - No lookahead: the last candle must be the most recent CLOSED bar,
      i.e. open_time <= decision_time - 15 minutes. The caller should
      filter at the SQL level: CryptoCandle.open_time <= market_start_time
    """
    if len(closes) < MIN_HISTORY_CANDLES or candle_count < MIN_HISTORY_CANDLES:
        return AssetRegimeFeatures(
            symbol=symbol,
            candle_count=min(len(closes), candle_count),
            history_ready=False,
        )

    n = len(closes)

    # Returns at horizons
    ret_4h = _log_returns(closes, HORIZON_4H)
    ret_12h = _log_returns(closes, HORIZON_12H)
    ret_24h = _log_returns(closes, HORIZON_24H)

    # Volatility (std of 1-period log returns)
    log_ret_1 = np.log(closes[1:] / closes[:-1])
    log_ret_1 = log_ret_1[np.isfinite(log_ret_1)]

    vol_4h = _volatility(log_ret_1[-HORIZON_4H:]) if len(log_ret_1) >= HORIZON_4H else 0.0
    vol_24h = _volatility(log_ret_1[-HORIZON_24H:]) if len(log_ret_1) >= HORIZON_24H else 0.0
    vol_ratio = vol_4h / vol_24h if vol_24h > 1e-10 else 0.0

    # Trend efficiency (24h)
    eff_closes = closes[-HORIZON_24H:] if n >= HORIZON_24H else closes
    efficiency_24h = _efficiency_ratio(eff_closes)

    # Range ratio (24h average)
    ranges = (highs - lows) / np.where(closes > 0, closes, 1.0)
    range_ratio_24h = float(np.mean(ranges[-HORIZON_24H:])) if n >= HORIZON_24H else 0.0

    # Up ratio (24h)
    directions = (closes >= opens).astype(float)
    up_ratio_24h = float(np.mean(directions[-HORIZON_24H:])) if n >= HORIZON_24H else 0.5

    strength = compute_asset_strength(ret_4h, ret_12h, ret_24h, efficiency_24h)

    return AssetRegimeFeatures(
        symbol=symbol,
        ret_4h=ret_4h,
        ret_12h=ret_12h,
        ret_24h=ret_24h,
        vol_4h=vol_4h,
        vol_24h=vol_24h,
        vol_ratio=vol_ratio,
        efficiency_24h=efficiency_24h,
        range_ratio_24h=range_ratio_24h,
        up_ratio_24h=up_ratio_24h,
        candle_count=candle_count,
        history_ready=True,
        strength_score=strength,
    )


def compute_basket_features(
    assets: dict[str, AssetRegimeFeatures],
) -> BasketRegimeFeatures:
    """
    Compute cross-asset basket features from per-asset features.
    Only includes assets with history_ready=True.
    """
    ready = {k: v for k, v in assets.items() if v.history_ready}
    ready_count = len(ready)
    total_count = len(assets)

    if ready_count == 0:
        return BasketRegimeFeatures(
            ready_count=0,
            total_count=total_count,
            history_ready=False,
        )

    rets_4h = np.array([a.ret_4h for a in ready.values()])
    rets_12h = np.array([a.ret_12h for a in ready.values()])
    rets_24h = np.array([a.ret_24h for a in ready.values()])
    vols_24h = np.array([a.vol_24h for a in ready.values()])

    # Breadth: fraction with positive return
    breadth_up_4h = float(np.mean(rets_4h > 0))
    breadth_up_12h = float(np.mean(rets_12h > 0))
    breadth_up_24h = float(np.mean(rets_24h > 0))

    # Dispersion (std of returns across assets)
    dispersion_4h = float(np.std(rets_4h, ddof=1)) if ready_count > 1 else 0.0
    dispersion_24h = float(np.std(rets_24h, ddof=1)) if ready_count > 1 else 0.0

    # Market efficiency: |median_ret| / median_vol
    median_ret_24h = float(np.median(rets_24h))
    median_vol_24h = float(np.median(vols_24h))
    market_efficiency_24h = abs(median_ret_24h) / median_vol_24h if median_vol_24h > 1e-10 else 0.0

    # Compute global strength from basket medians
    b_strength = compute_asset_strength(
        float(np.median(rets_4h)), float(np.median(rets_12h)),
        float(np.median(rets_24h)), market_efficiency_24h,
    )

    return BasketRegimeFeatures(
        median_ret_4h=float(np.median(rets_4h)),
        median_ret_12h=float(np.median(rets_12h)),
        median_ret_24h=median_ret_24h,
        breadth_up_4h=breadth_up_4h,
        breadth_up_12h=breadth_up_12h,
        breadth_up_24h=breadth_up_24h,
        dispersion_4h=dispersion_4h,
        dispersion_24h=dispersion_24h,
        market_efficiency_24h=market_efficiency_24h,
        ready_count=ready_count,
        total_count=total_count,
        history_ready=True,
        strength=b_strength,
    )


def build_regime_snapshot(
    candle_data: dict[str, dict[str, np.ndarray]],
    as_of: datetime | None = None,
    max_open_time: datetime | None = None,
) -> MarketRegimeSnapshot:
    """
    Build a complete regime snapshot from candle data.

    Args:
        candle_data: {symbol: {"closes": array, "highs": array, "lows": array,
                               "opens": array, "count": int}}
        as_of: Decision timestamp (UTC). Defaults to now.

    Returns:
        MarketRegimeSnapshot with per-asset features, basket features, and reason codes.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    # max_open_time guard: reject candles opened after decision time
    cutoff = max_open_time or as_of

    reason_codes: list[str] = []
    assets: dict[str, AssetRegimeFeatures] = {}

    for symbol, data in candle_data.items():
        closes = np.asarray(data["closes"], dtype=np.float64)
        highs = np.asarray(data["highs"], dtype=np.float64)
        lows = np.asarray(data["lows"], dtype=np.float64)
        opens = np.asarray(data["opens"], dtype=np.float64)
        open_times = data.get("open_times")
        count = int(data.get("count", len(closes)))

        # Trim future candles if open_times provided
        if open_times is not None and len(open_times) == len(closes):
            mask = np.array([t <= cutoff for t in open_times])
            if not mask.all():
                closes = closes[mask]
                highs = highs[mask]
                lows = lows[mask]
                opens = opens[mask]
                count = int(len(closes))

        asset_feat = compute_asset_features(symbol, closes, highs, lows, opens, count)
        assets[symbol] = asset_feat

        if not asset_feat.history_ready:
            reason_codes.append(f"insufficient_history:{symbol}:{count}/{MIN_HISTORY_CANDLES}")

    basket = compute_basket_features(assets)

    if not basket.history_ready:
        reason_codes.append("no_assets_ready")
    elif basket.ready_count < basket.total_count:
        reason_codes.append(f"partial_coverage:{basket.ready_count}/{basket.total_count}")

    return MarketRegimeSnapshot(
        as_of=as_of,
        assets=assets,
        basket=basket,
        reason_codes=reason_codes,
    )
