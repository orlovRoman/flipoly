"""
Market regime integration v2 — multi-asset snapshot builder.

Fixes:
- Loads candles for ALL assets (not just one)
- Computes global phase from full basket
- Extracts per-asset phase for the decision asset
- Fail-open: if data insufficient, returns UNKNOWN without blocking
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
from polyflip.crypto.market_regime_classifier import classify_global_regime, classify_asset_regime, MarketPhase


def build_snapshot_from_candles(
    candles: Sequence[Any],
    symbol: str,
    as_of: datetime,
) -> MarketRegimeSnapshot:
    """
    Build a regime snapshot from ORM candle objects for a single asset.
    Kept for backward compat.
    """
    if not candles:
        return build_regime_snapshot({}, as_of=as_of)

    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    open_times = []
    filtered = []
    for c in candles:
        # Only use closed candles
        if getattr(c, "is_closed", None) is not True:
            continue
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

    if open_times:
        is_valid, reason = validate_candle_continuity(open_times, len(filtered))
        if not is_valid:
            snapshot.reason_codes.append(f"candle_continuity:{reason}")

    return snapshot


def build_snapshot_from_multi_asset_candles(
    candles_by_asset: dict[str, Sequence[Any]],
    as_of: datetime,
) -> MarketRegimeSnapshot:
    """
    Build a multi-asset regime snapshot from candles for ALL configured assets.

    Args:
        candles_by_asset: {symbol: [CryptoCandle ORM objects]}
        as_of: decision timestamp (UTC)

    Returns:
        MarketRegimeSnapshot with full basket + per-asset features.
        If data insufficient, returns UNKNOWN with fail-open behavior.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    candle_data = {}
    reason_codes = []

    for symbol, candles in candles_by_asset.items():
        if not candles:
            reason_codes.append(f"no_candles:{symbol}")
            continue

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
            reason_codes.append(f"no_valid_candles:{symbol}")
            continue

        closes = np.array([float(c.close) for c in filtered], dtype=np.float64)
        highs = np.array([float(c.high) for c in filtered], dtype=np.float64)
        lows = np.array([float(c.low) for c in filtered], dtype=np.float64)
        opens = np.array([float(c.open) for c in filtered], dtype=np.float64)

        # Validate continuity
        if open_times:
            is_valid, reason = validate_candle_continuity(open_times, len(filtered))
            if not is_valid:
                reason_codes.append(f"candle_continuity:{symbol}:{reason}")
                # Mark asset as invalid — continuity failures block MRF for this asset
                continue

        candle_data[symbol] = {
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "opens": opens,
            "open_times": open_times if open_times else None,
            "count": len(filtered),
        }

    snapshot = build_regime_snapshot(
        candle_data, as_of=as_of, max_open_time=as_of,
    )

    # Merge reason codes
    snapshot.reason_codes.extend(reason_codes)

    return snapshot


def extract_asset_phase(
    snapshot: MarketRegimeSnapshot,
    asset_symbol: str,
) -> tuple[MarketPhase, float, float]:
    """
    Extract the phase, strength, and confidence for a specific asset
    from the snapshot. Returns UNKNOWN if asset not found or not ready.
    """
    feat = snapshot.assets.get(asset_symbol)
    if feat is None or not feat.history_ready:
        return MarketPhase.UNKNOWN, 0.0, 0.0

    cls = classify_asset_regime(feat)
    return cls.phase, cls.strength, cls.confidence
