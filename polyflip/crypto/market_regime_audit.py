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
from polyflip.crypto.market_regime_classifier import MarketPhase, classify_global_regime, classify_asset_regime, RegimeConfig
from polyflip.crypto.market_regime_policy import (
    FilterMode,
    PolicyResult,
    RegimeGateResult,
)


def serialize_regime_audit(
    snapshot: MarketRegimeSnapshot,
    policy_result: PolicyResult | None,
    mode: FilterMode,
    mrf_version: int,
    strategy_type: str = "",
    applied: bool = False,
    failure_reason: str | None = None,
    regime_config: RegimeConfig | None = None,
    gate_result: RegimeGateResult | None = None,
    effective_block: bool = False,
    candidate_role: str = "",
) -> dict[str, Any]:
    """
    Build a compact audit dict for the decision funnel.

    Includes both global phase and per-asset phases.
    Uses the same RegimeConfig as the classifier for consistency.
    """
    global_phase, global_confidence = _extract_global_phase(
        snapshot, regime_config=regime_config,
    )
    global_strength = snapshot.basket.strength if snapshot.basket.history_ready else 0.0
    if gate_result is not None:
        global_phase = gate_result.global_phase
        global_confidence = gate_result.global_confidence
        global_strength = gate_result.global_strength
    elif policy_result is not None:
        # Legacy policy already computed the exact classifier values used for
        # the decision.  Prefer them over the snapshot's cached basket score
        # so v1/v2 telemetry remains internally consistent.
        global_strength = policy_result.global_strength

    # Per-asset phases (using same RegimeConfig)
    assets_phases = {}
    for sym, feat in snapshot.assets.items():
        if feat.history_ready:
            cls = classify_asset_regime(feat, config=regime_config)
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

    gate_summary = {}
    if gate_result is not None:
        gate_summary = {
            "would_block": gate_result.would_block,
            "effective_block": effective_block,
            "reason": gate_result.reason,
            "candidate_direction": round(gate_result.candidate_direction, 4),
            "candidate_role": candidate_role,
            "asset_phase": gate_result.asset_phase.value,
            "global_phase": gate_result.global_phase.value,
            "asset_strength": round(gate_result.asset_strength, 4),
            "asset_confidence": round(gate_result.asset_confidence, 4),
            "global_strength": round(gate_result.global_strength, 4),
            "global_confidence": round(gate_result.global_confidence, 4),
            "asset_evidence": round(gate_result.asset_evidence, 6),
            "global_evidence": round(gate_result.global_evidence, 6),
            "regime_evidence": round(gate_result.regime_evidence, 6),
            "net_edge": round(gate_result.net_edge, 6),
            "min_edge_used": round(gate_result.min_edge_used, 6),
            "edge_margin": round(gate_result.edge_margin, 6),
            "veto_threshold": round(gate_result.veto_threshold, 6),
            "edge_override_margin": round(gate_result.edge_override_margin, 6),
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
        "gate": gate_summary,
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
    regime_config: RegimeConfig | None = None,
    gate_result: RegimeGateResult | None = None,
    effective_block: bool = False,
    candidate_role: str = "",
) -> str:
    """Serialize audit to JSON string."""
    audit = serialize_regime_audit(
        snapshot=snapshot,
        policy_result=policy_result,
        mode=mode,
        mrf_version=mrf_version,
        strategy_type=strategy_type,
        applied=applied,
        failure_reason=failure_reason,
        regime_config=regime_config,
        gate_result=gate_result,
        effective_block=effective_block,
        candidate_role=candidate_role,
    )
    return json.dumps(audit, ensure_ascii=False, separators=(",", ":"))


def _extract_global_phase(
    snapshot: MarketRegimeSnapshot,
    regime_config: RegimeConfig | None = None,
) -> tuple[MarketPhase, float]:
    """Extract global phase and confidence from snapshot."""
    if snapshot.basket.history_ready:
        cls = classify_global_regime(snapshot, config=regime_config)
        return cls.phase, cls.confidence
    return MarketPhase.UNKNOWN, 0.0
