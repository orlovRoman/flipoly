"""
Market regime audit / telemetry serialization (T10 of MRF plan).

Produces a compact JSON-safe dict for the decision funnel metadata namespace.
Does not touch lgbm_metadata — writes to a separate "mrf" key.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from polyflip.crypto.market_regime import MarketRegimeSnapshot, MIN_HISTORY_CANDLES
from polyflip.crypto.market_regime_classifier import Regime, classify_global_regime
from polyflip.crypto.market_regime_policy import FilterMode, PolicyResult


def serialize_regime_audit(
    snapshot: MarketRegimeSnapshot,
    policy_result: PolicyResult | None,
    mode: FilterMode,
    mrf_version: int,
    strategy_type: str = "",
    applied: bool = False,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """
    Build a compact audit dict for the decision funnel.

    Returns a JSON-serializable dict suitable for storage in
    DecisionFunnelLog.market_regime_audit or a metadata namespace.

    Keys:
        mode, version, as_of, global_regime, global_confidence,
        assets (per-asset summary), basket (cross-asset summary),
        policy (allow/multiplier/reason), applied, failure_reason, reason_codes
    """
    global_regime, global_confidence = _extract_global_regime(snapshot)

    assets_summary = {}
    for sym, feat in snapshot.assets.items():
        assets_summary[sym] = {
            "ret_24h": round(feat.ret_24h, 6),
            "efficiency": round(feat.efficiency_24h, 4),
            "vol_ratio": round(feat.vol_ratio, 4),
            "up_ratio": round(feat.up_ratio_24h, 4),
            "ready": feat.history_ready,
            "candles": feat.candle_count,
        }

    basket_summary = {}
    if snapshot.basket.history_ready:
        basket_summary = {
            "median_ret_24h": round(snapshot.basket.median_ret_24h, 6),
            "efficiency": round(snapshot.basket.market_efficiency_24h, 4),
            "breadth_up_24h": round(snapshot.basket.breadth_up_24h, 4),
            "dispersion_24h": round(snapshot.basket.dispersion_24h, 6),
            "ready_count": snapshot.basket.ready_count,
            "total_count": snapshot.basket.total_count,
        }

    policy_summary = {}
    if policy_result is not None:
        policy_summary = {
            "allow": policy_result.allow,
            "multiplier": round(policy_result.stake_multiplier, 4),
            "reason": policy_result.reason,
            "regime": policy_result.regime.value,
        }

    audit = {
        "mode": mode.value,
        "version": mrf_version,
        "as_of": snapshot.as_of.isoformat(),
        "global_regime": global_regime.value,
        "global_confidence": round(global_confidence, 4),
        "strategy_type": strategy_type,
        "assets": assets_summary,
        "basket": basket_summary,
        "policy": policy_summary,
        "applied": applied,
        "failure_reason": failure_reason,
        "reason_codes": snapshot.reason_codes,
    }

    return audit


def serialize_regime_audit_json(
    snapshot: MarketRegimeSnapshot,
    policy_result: PolicyResult | None,
    mode: FilterMode,
    mrf_version: int,
    strategy_type: str = "",
    applied: bool = False,
    failure_reason: str | None = None,
) -> str:
    """Serialize audit to JSON string for direct storage."""
    audit = serialize_regime_audit(
        snapshot, policy_result, mode, mrf_version,
        strategy_type, applied, failure_reason,
    )
    return json.dumps(audit, ensure_ascii=False, separators=(",", ":"))


def _extract_global_regime(snapshot: MarketRegimeSnapshot) -> tuple[Regime, float]:
    """Extract global regime and confidence from snapshot."""
    if snapshot.basket.history_ready:
        cls = classify_global_regime(snapshot)
        return cls.regime, cls.confidence
    return Regime.UNKNOWN, 0.0
