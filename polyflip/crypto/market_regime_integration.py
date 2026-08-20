"""
Market regime integration at decision boundary (T08 of MRF plan).

Builds regime snapshot from candles up to a cutoff time, avoiding lookahead.
Pure function — no decision logic, just snapshot construction.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from polyflip.crypto.market_regime import (
    build_regime_snapshot,
    MarketRegimeSnapshot,
    validate_candle_continuity,
    MIN_HISTORY_CANDLES,
)
from polyflip.crypto.market_regime_classifier import classify_global_regime, Regime
from polyflip.crypto.market_regime_policy import (
    FilterMode,
    PolicyResult,
    StrategyType,
    evaluate_policy,
)


def build_snapshot_from_candles(
    candles: Sequence[Any],
    symbol: str,
    as_of: datetime,
) -> MarketRegimeSnapshot:
    """
    Build a regime snapshot from ORM candle objects.

    The caller MUST ensure:
    - All candles have open_time <= as_of (no lookahead)
    - Candles are sorted by open_time ascending
    - Only CLOSED candles are included

    Args:
        candles: list of CryptoCandle ORM objects (or similar with .open/.high/.low/.close/.open_time)
        symbol: e.g. "BTC"
        as_of: decision timestamp (UTC)

    Returns:
        MarketRegimeSnapshot
    """
    if not candles:
        return build_regime_snapshot({}, as_of=as_of)

    closes = np.array([float(c.close) for c in candles], dtype=np.float64)
    highs = np.array([float(c.high) for c in candles], dtype=np.float64)
    lows = np.array([float(c.low) for c in candles], dtype=np.float64)
    opens = np.array([float(c.open) for c in candles], dtype=np.float64)

    # Validate continuity
    open_times = []
    for c in candles:
        ot = getattr(c, "open_time", None)
        if ot is not None:
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)
            open_times.append(ot)

    candle_data = {
        symbol: {
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "opens": opens,
            "count": len(candles),
        }
    }

    snapshot = build_regime_snapshot(candle_data, as_of=as_of)

    # Add continuity validation reason codes
    if open_times:
        is_valid, reason = validate_candle_continuity(open_times, len(candles))
        if not is_valid:
            snapshot.reason_codes.append(f"candle_continuity:{reason}")

    return snapshot
