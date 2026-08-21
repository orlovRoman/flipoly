"""
Strategic policy function v2 — uses MarketPhase (STRONG_UP..UNKNOWN).

Takes a snapshot, strategy type, direction, and filter mode.
Returns allow/deny, stake multiplier, and reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from polyflip.crypto.market_regime import MarketRegimeSnapshot
from polyflip.crypto.market_regime_classifier import (
    MarketPhase,
    RegimeClassification,
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


DEFAULT_POLICY_CONFIG = PolicyConfig()


def _get_global_regime(snapshot: MarketRegimeSnapshot) -> tuple[MarketPhase, float, float]:
    """Extract global phase, confidence, and strength from snapshot."""
    if snapshot.basket.history_ready:
        from polyflip.crypto.market_regime_classifier import classify_global_regime
        cls = classify_global_regime(snapshot)
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
) -> PolicyResult:
    """
    Evaluate regime-based policy for a given strategy and direction.
    """
    cfg = config or DEFAULT_POLICY_CONFIG
    phase, confidence, strength = _get_global_regime(snapshot)

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
