"""
Market regime audit / telemetry serialization v2.

Produces compact JSON-safe dict for the decision funnel metadata.
Uses MarketPhase (global_phase + per-asset phases).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from polyflip.crypto.market_regime import MarketRegimeSnapshot, MIN_HISTORY_CANDLES
from polyflip.crypto.market_regime_classifier import MarketPhase, classify_global_regime, classify_asset_regime
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

    Includes both global phase and per-asset phases.
    """
    global_phase, global_confidence = _extract_global_phase(snapshot)
    global_strength = snapshot.basket.strength if snapshot.basket.history_ready else 0.0

    # Per-asset phases
    assets_phases = {}
    for sym, feat in snapshot.assets.items():
        if feat.history_ready:
            cls = classify_asset_regime(feat)
            assets_phases[sym] = {
                "phase": cls.phase.value,
                "strength": round(cls.strength, 4),
                "confidence": round(cls.confidence, 4),
                "direction": round(cls.direction, 4),
            }
        else:
            assets_phases[sym] = {
                "phase": "UNKNOWN",
                "strength": 0.0,
                "confidence": 0.0,
                "direction": 0.0,
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
            "phase": policy_result.phase.value,
        }

    audit = {
        "mode": mode.value,
        "version": mrf_version,
        "as_of": snapshot.as_of.isoformat(),
        "global_phase": global_phase.value,
        "global_regime": global_phase.value,  # backward compat alias
        "global_confidence": round(global_confidence, 4),
        "global_strength": round(global_strength, 4),
        "strategy_type": strategy_type,
        "assets": assets_phases,
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
    """Serialize audit to JSON string."""
    audit = serialize_regime_audit(
        snapshot, policy_result, mode, mrf_version,
        strategy_type, applied, failure_reason,
    )
    return json.dumps(audit, ensure_ascii=False, separators=(",", ":"))


def _extract_global_phase(snapshot: MarketRegimeSnapshot) -> tuple[MarketPhase, float]:
    """Extract global phase and confidence from snapshot."""
    if snapshot.basket.history_ready:
        cls = classify_global_regime(snapshot)
        return cls.phase, cls.confidence
    return MarketPhase.UNKNOWN, 0.0
