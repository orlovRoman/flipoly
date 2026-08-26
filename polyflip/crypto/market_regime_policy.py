"""
Strategic policy function v2 — uses MarketPhase (STRONG_UP..UNKNOWN).

Takes a snapshot, strategy type, direction, and filter mode.
Returns allow/deny, stake multiplier, and reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Optional

from polyflip.crypto.market_regime import MarketRegimeSnapshot
from polyflip.crypto.market_regime_classifier import (
    MarketPhase,
    RegimeClassification,
    RegimeConfig,
    classify_asset_regime,
    classify_global_regime,
)


class FilterMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"


class StrategyType(str, Enum):
    OUTSIDER = "OUTSIDER"
    ML_TREND_FOLLOW = "ML_TREND_FOLLOW"
    ML_TREND_FADE = "ML_TREND_FADE"
    ML_FAVORITE = "ML_FAVORITE"
    OTHER = "OTHER"


@dataclass(frozen=True)
class PolicyResult:
    """Result of regime policy evaluation."""
    allow: bool
    stake_multiplier: float  # 0.0 = blocked, 1.0 = normal, <1 = reduced
    reason: str
    phase: MarketPhase
    global_confidence: float
    global_strength: float = 0.0

    @property
    def regime(self) -> MarketPhase:
        """Deprecated: use .phase instead."""
        return self.phase

    @property
    def global_regime(self) -> MarketPhase:
        """Deprecated: use .phase instead."""
        return self.phase


@dataclass(frozen=True)
class PolicyConfig:
    """Configurable thresholds for policy decisions."""
    outsider_trend_multiplier: float = 0.0
    outsider_sideways_multiplier: float = 1.0
    outsider_high_vol_multiplier: float = 0.5
    outsider_mixed_multiplier: float = 0.5
    outsider_unknown_multiplier: float = 0.8

    mltrend_favor_multiplier: float = 1.0
    mltrend_fade_multiplier: float = 0.8
    mltrend_against_multiplier: float = 0.0
    mltrend_high_vol_multiplier: float = 0.5
    mltrend_mixed_multiplier: float = 0.5
    mltrend_unknown_multiplier: float = 0.8
    mltrend_sideways_multiplier: float = 0.8

    favorite_follow_multiplier: float = 1.0
    favorite_against_multiplier: float = 0.0
    favorite_unknown_multiplier: float = 0.8

    unknown_multiplier: float = 0.8


@dataclass(frozen=True)
class VetoGateConfig:
    """Configuration for MRF v3's binary veto gate.

    The gate is deliberately separate from the legacy multiplier policy:
    v3 either preserves the candidate bet or vetoes it, and never resizes it.
    """

    asset_weight: float = 0.70
    global_weight: float = 0.30
    veto_threshold: float = 0.15
    edge_override_margin: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.asset_weight,
            self.global_weight,
            self.veto_threshold,
            self.edge_override_margin,
        )
        if not all(_is_finite_number(value) for value in values):
            raise ValueError("MRF gate config contains non-finite values")
        if self.asset_weight < 0 or self.global_weight < 0:
            raise ValueError("MRF gate weights must be non-negative")
        if self.asset_weight + self.global_weight <= 0:
            raise ValueError("MRF gate weight sum must be positive")
        if not 0 <= self.veto_threshold <= 1:
            raise ValueError("MRF veto threshold must be in [0, 1]")
        if self.edge_override_margin < 0:
            raise ValueError("MRF edge override margin must be non-negative")


@dataclass(frozen=True)
class RegimeGateResult:
    """Auditable result of one MRF v3 evaluation."""

    would_block: bool
    reason: str
    candidate_direction: float
    asset_phase: MarketPhase
    global_phase: MarketPhase
    asset_strength: float
    asset_confidence: float
    global_strength: float
    global_confidence: float
    asset_evidence: float
    global_evidence: float
    regime_evidence: float
    net_edge: float
    min_edge_used: float
    edge_margin: float
    veto_threshold: float
    edge_override_margin: float


DEFAULT_POLICY_CONFIG = PolicyConfig()


_DIRECTION_BY_PHASE = {
    MarketPhase.STRONG_UP: 1.0,
    MarketPhase.WEAK_UP: 1.0,
    MarketPhase.STRONG_DOWN: -1.0,
    MarketPhase.WEAK_DOWN: -1.0,
}


def _is_finite_number(value: object) -> bool:
    """Return whether a runtime value can safely participate in gate math."""
    try:
        return isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _classification_evidence(
    classification: RegimeClassification,
    candidate_direction: float,
) -> float:
    """Return signed evidence in [-1, 1] for a candidate side."""
    if not all(
        _is_finite_number(value)
        for value in (classification.strength, classification.confidence)
    ):
        raise ValueError("MRF classification contains non-finite values")
    regime_direction = _DIRECTION_BY_PHASE.get(classification.phase, 0.0)
    strength = float(classification.strength)
    confidence = float(classification.confidence)
    evidence = (
        candidate_direction
        * regime_direction
        * strength
        * confidence
    )
    return max(-1.0, min(1.0, evidence))


def compute_veto_gate(
    *,
    asset_classification: RegimeClassification,
    global_classification: RegimeClassification,
    candidate_direction: float,
    net_edge: float,
    min_edge_used: float,
    config: VetoGateConfig,
) -> RegimeGateResult:
    """Compute MRF v3 without I/O or trading-side effects."""
    if candidate_direction not in (-1.0, 1.0):
        raise ValueError("candidate_direction must be -1.0 or 1.0")
    if not all(_is_finite_number(value) for value in (net_edge, min_edge_used)):
        raise ValueError("MRF gate received non-finite edge input")

    net_edge = float(net_edge)
    min_edge_used = float(min_edge_used)

    asset_evidence = _classification_evidence(
        asset_classification, candidate_direction,
    )
    global_evidence = _classification_evidence(
        global_classification, candidate_direction,
    )
    weight_sum = config.asset_weight + config.global_weight
    regime_evidence = round(
        (
            config.asset_weight * asset_evidence
            + config.global_weight * global_evidence
        ) / weight_sum,
        6,
    )
    edge_margin = round(net_edge - min_edge_used, 6)
    # Neutral phases have zero direction and must never veto merely because
    # an operator configured a zero threshold.  Keep the threshold boundary
    # inclusive for genuinely opposing (negative) evidence.
    negative_regime = (
        regime_evidence < 0
        and regime_evidence <= -config.veto_threshold
    )
    strong_edge_override = edge_margin >= config.edge_override_margin
    would_block = negative_regime and not strong_edge_override

    if would_block:
        reason = "regime_veto"
    elif negative_regime and strong_edge_override:
        reason = "strong_edge_override"
    elif regime_evidence > 0:
        reason = "regime_supports_candidate"
    else:
        reason = "no_negative_regime_evidence"

    return RegimeGateResult(
        would_block=would_block,
        reason=reason,
        candidate_direction=candidate_direction,
        asset_phase=asset_classification.phase,
        global_phase=global_classification.phase,
        asset_strength=asset_classification.strength,
        asset_confidence=asset_classification.confidence,
        global_strength=global_classification.strength,
        global_confidence=global_classification.confidence,
        asset_evidence=round(asset_evidence, 6),
        global_evidence=round(global_evidence, 6),
        regime_evidence=regime_evidence,
        net_edge=net_edge,
        min_edge_used=min_edge_used,
        edge_margin=edge_margin,
        veto_threshold=config.veto_threshold,
        edge_override_margin=config.edge_override_margin,
    )


def evaluate_veto_gate(
    *,
    snapshot: MarketRegimeSnapshot,
    asset_symbol: str,
    candidate_direction: float,
    net_edge: float,
    min_edge_used: float,
    config: VetoGateConfig,
    regime_config: RegimeConfig | None = None,
) -> RegimeGateResult:
    """Evaluate v3 using the immutable snapshot produced by the collector."""
    asset_features = snapshot.assets.get(asset_symbol)
    if asset_features is None or not asset_features.history_ready:
        raise ValueError(f"MRF asset regime unavailable: {asset_symbol}")
    asset_cls = classify_asset_regime(asset_features, config=regime_config)
    global_cls = classify_global_regime(snapshot, config=regime_config)
    return compute_veto_gate(
        asset_classification=asset_cls,
        global_classification=global_cls,
        candidate_direction=candidate_direction,
        net_edge=net_edge,
        min_edge_used=min_edge_used,
        config=config,
    )


def _get_global_regime(
    snapshot: MarketRegimeSnapshot,
    regime_config: RegimeConfig | None = None,
) -> tuple[MarketPhase, float, float]:
    """Extract global phase, confidence, and strength from snapshot."""
    if snapshot.basket.history_ready:
        from polyflip.crypto.market_regime_classifier import classify_global_regime
        cls = classify_global_regime(snapshot, config=regime_config)
        return cls.phase, cls.confidence, cls.strength
    return MarketPhase.UNKNOWN, 0.0, 0.0


def _is_trending(phase: MarketPhase) -> bool:
    """Check if phase is a directional trend."""
    return phase in (
        MarketPhase.STRONG_UP, MarketPhase.WEAK_UP,
        MarketPhase.STRONG_DOWN, MarketPhase.WEAK_DOWN,
    )


def _phase_direction(phase: MarketPhase) -> float:
    """Get direction from phase: +1 up, -1 down, 0 neutral."""
    if phase in (MarketPhase.STRONG_UP, MarketPhase.WEAK_UP):
        return 1.0
    if phase in (MarketPhase.STRONG_DOWN, MarketPhase.WEAK_DOWN):
        return -1.0
    return 0.0


def evaluate_policy(
    snapshot: MarketRegimeSnapshot,
    strategy: StrategyType,
    direction: float,
    mode: FilterMode = FilterMode.SHADOW,
    config: PolicyConfig | None = None,
    regime_config: RegimeConfig | None = None,
) -> PolicyResult:
    """
    Evaluate regime-based policy for a given strategy and direction.
    regime_config is passed through to the global classifier.
    """
    cfg = config or DEFAULT_POLICY_CONFIG
    phase, confidence, strength = _get_global_regime(snapshot, regime_config=regime_config)

    if mode == FilterMode.OFF:
        return PolicyResult(
            allow=True,
            stake_multiplier=1.0,
            reason="filter_off",
            phase=phase,
            global_confidence=confidence,
            global_strength=strength,
        )

    regime_direction = _phase_direction(phase)

    # ── OUTSIDER strategy ──
    if strategy == StrategyType.OUTSIDER:
        if phase == MarketPhase.SIDEWAYS:
            return PolicyResult(
                allow=True,
                stake_multiplier=cfg.outsider_sideways_multiplier,
                reason="outsider_in_sideways",
                phase=phase, global_confidence=confidence, global_strength=strength,
            )
        if _is_trending(phase):
            return PolicyResult(
                allow=cfg.outsider_trend_multiplier > 0,
                stake_multiplier=cfg.outsider_trend_multiplier,
                reason="outsider_blocked_in_trend",
                phase=phase, global_confidence=confidence, global_strength=strength,
            )
        if phase == MarketPhase.HIGH_VOL_CHOP:
            return PolicyResult(
                allow=True,
                stake_multiplier=cfg.outsider_high_vol_multiplier,
                reason="outsider_reduced_in_high_vol",
                phase=phase, global_confidence=confidence, global_strength=strength,
            )
        if phase == MarketPhase.MIXED:
            return PolicyResult(
                allow=True,
                stake_multiplier=cfg.outsider_mixed_multiplier,
                reason="outsider_reduced_in_mixed",
                phase=phase, global_confidence=confidence, global_strength=strength,
            )
        return PolicyResult(
            allow=True,
            stake_multiplier=cfg.outsider_unknown_multiplier,
            reason="outsider_reduced_in_unknown",
            phase=phase, global_confidence=confidence, global_strength=strength,
        )

    # ── ML_TREND strategies ──
    if strategy in (StrategyType.ML_TREND_FOLLOW, StrategyType.ML_TREND_FADE):
        aligns = (direction * regime_direction > 0) if regime_direction != 0 else False

        if phase == MarketPhase.HIGH_VOL_CHOP:
            mult = cfg.mltrend_high_vol_multiplier
            reason = "mltrend_reduced_high_vol"
        elif phase == MarketPhase.MIXED:
            mult = cfg.mltrend_mixed_multiplier
            reason = "mltrend_reduced_mixed"
        elif phase == MarketPhase.UNKNOWN:
            mult = cfg.mltrend_unknown_multiplier
            reason = "mltrend_reduced_unknown"
        elif phase == MarketPhase.SIDEWAYS:
            mult = cfg.mltrend_sideways_multiplier
            reason = "mltrend_reduced_sideways"
        elif aligns:
            mult = cfg.mltrend_favor_multiplier
            reason = "mltrend_aligned_with_regime"
        else:
            mult = cfg.mltrend_against_multiplier
            reason = "mltrend_against_regime"

        return PolicyResult(
            allow=mult > 0,
            stake_multiplier=mult,
            reason=reason,
            phase=phase, global_confidence=confidence, global_strength=strength,
        )

    # ── ML_FAVORITE strategy ──
    if strategy == StrategyType.ML_FAVORITE:
        aligns = (direction * regime_direction > 0) if regime_direction != 0 else False

        if phase == MarketPhase.UNKNOWN:
            mult = cfg.favorite_unknown_multiplier
            reason = "favorite_reduced_unknown"
        elif aligns:
            mult = cfg.favorite_follow_multiplier
            reason = "favorite_aligned_with_regime"
        else:
            mult = cfg.favorite_against_multiplier
            reason = "favorite_against_regime"

        return PolicyResult(
            allow=mult > 0,
            stake_multiplier=mult,
            reason=reason,
            phase=phase, global_confidence=confidence, global_strength=strength,
        )

    # ── OTHER ──
    return PolicyResult(
        allow=True,
        stake_multiplier=1.0,
        reason="unknown_strategy_no_policy",
        phase=phase, global_confidence=confidence, global_strength=strength,
    )
