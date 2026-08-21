"""
Market regime integration v2 — multi-asset snapshot builder.

Fixes:
- Loads candles for ALL assets (not just one)
- Computes global phase from full basket
- Extracts per-asset phase for the decision asset
- Fail-open: if data insufficient, returns UNKNOWN without blocking
- Step 4: tracks failure reasons per asset, sets history_ready=false for incomplete baskets
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from polyflip.crypto.market_regime import (
    build_regime_snapshot,
    MarketRegimeSnapshot,
    validate_candle_continuity,
    MIN_HISTORY_CANDLES,
)
from polyflip.crypto.market_regime_classifier import classify_global_regime, classify_asset_regime, MarketPhase, RegimeConfig
from polyflip.crypto.market_regime import AssetRegimeFeatures


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
    expected_assets: list[str] | None = None,
) -> MarketRegimeSnapshot:
    """
    Build a multi-asset regime snapshot from candles for ALL configured assets.

    Args:
        candles_by_asset: {symbol: [CryptoCandle ORM objects]}
        as_of: decision timestamp (UTC)
        expected_assets: full list of assets that should be present.
            If provided and any are missing, basket.history_ready=False
            and per-asset failure reasons are tracked.

    Returns:
        MarketRegimeSnapshot with full basket + per-asset features.
        If any expected asset is missing or has errors, basket.history_ready=False
        and the caller should skip MRF classification (fail-open: return action unchanged).
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    candle_data = {}
    reason_codes = []
    asset_failure_reasons: dict[str, str] = {}

    for symbol, candles in candles_by_asset.items():
        if not candles:
            reason_codes.append(f"no_candles:{symbol}")
            asset_failure_reasons[symbol] = "no_candles"
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
            asset_failure_reasons[symbol] = "no_valid_candles"
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
                asset_failure_reasons[symbol] = f"candle_continuity:{reason}"
                continue

        candle_data[symbol] = {
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "opens": opens,
            "open_times": open_times if open_times else None,
            "count": len(filtered),
        }

    # Check for missing expected assets
    if expected_assets:
        for exp in expected_assets:
            if exp not in candle_data and exp not in asset_failure_reasons:
                reason_codes.append(f"asset_missing:{exp}")
                asset_failure_reasons[exp] = "asset_missing"

    snapshot = build_regime_snapshot(
        candle_data, as_of=as_of, max_open_time=as_of,
    )

    # Merge reason codes
    snapshot.reason_codes.extend(reason_codes)

    # Step 4: If any expected asset is missing/broken, mark basket as not ready.
    # This prevents global classification on an incomplete basket.
    if expected_assets and asset_failure_reasons:
        missing_count = len(asset_failure_reasons)
        total_expected = len(expected_assets)
        if missing_count >= total_expected:
            # ALL assets failed — basket is completely empty
            snapshot = dataclasses.replace(
                snapshot,
                basket=dataclasses.replace(
                    snapshot.basket,
                    history_ready=False,
                    ready_count=0,
                    total_count=total_expected,
                ),
            )
        elif missing_count >= 1:
            # Some assets failed — mark as not ready to avoid misleading classification
            snapshot = dataclasses.replace(
                snapshot,
                basket=dataclasses.replace(
                    snapshot.basket,
                    history_ready=False,
                    ready_count=len(candle_data),
                    total_count=total_expected,
                ),
            )

    # Store failure reasons as snapshot metadata for audit
    if asset_failure_reasons:
        snapshot.reason_codes.append(
            f"asset_failures:{','.join(f'{k}={v}' for k, v in asset_failure_reasons.items())}"
        )

    return snapshot


def extract_asset_phase(
    snapshot: MarketRegimeSnapshot,
    asset_symbol: str,
    regime_config: RegimeConfig | None = None,
) -> tuple[MarketPhase, float, float]:
    """
    Extract the phase, strength, and confidence for a specific asset
    from the snapshot. Returns UNKNOWN if asset not found or not ready.
    """
    feat = snapshot.assets.get(asset_symbol)
    if feat is None or not feat.history_ready:
        return MarketPhase.UNKNOWN, 0.0, 0.0

    cls = classify_asset_regime(feat, config=regime_config)
    return cls.phase, cls.strength, cls.confidence
