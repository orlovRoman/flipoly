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

    Automatically filters out candles with open_time > as_of to prevent lookahead.
    Candles should be sorted by open_time ascending and only CLOSED candles included.

    Args:
        candles: list of CryptoCandle ORM objects (or similar with .open/.high/.low/.close/.open_time)
        symbol: e.g. "BTC"
        as_of: decision timestamp (UTC)

    Returns:
        MarketRegimeSnapshot
    """
    if not candles:
        return build_regime_snapshot({}, as_of=as_of)

    # Collect open_times and filter future candles
    open_times = []
    filtered = []
    for c in candles:
        ot = getattr(c, "open_time", None)
        if ot is not None:
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)
            if ot > as_of:
                continue
            open_times.append(ot)
        filtered.append(c)

    if not filtered:
        return build_regime_snapshot({}, as_of=as_of)

    closes = np.array([float(c.close) for c in filtered], dtype=np.float64)
    highs = np.array([float(c.high) for c in filtered], dtype=np.float64)
    lows = np.array([float(c.low) for c in filtered], dtype=np.float64)
    opens = np.array([float(c.open) for c in filtered], dtype=np.float64)

    candle_data = {
        symbol: {
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "opens": opens,
            "open_times": open_times if open_times else None,
            "count": len(filtered),
        }
    }

    snapshot = build_regime_snapshot(
        candle_data, as_of=as_of, max_open_time=as_of,
    )

    # Add continuity validation reason codes
    if open_times:
        is_valid, reason = validate_candle_continuity(open_times, len(filtered))
        if not is_valid:
            snapshot.reason_codes.append(f"candle_continuity:{reason}")

    return snapshot
