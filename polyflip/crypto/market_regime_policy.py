"""
Strategic policy function for market regime filter (T07 of MRF plan).

Takes a snapshot, strategy type, and filter mode. Returns allow/deny,
stake multiplier, and reason. Pure, no DB reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from polyflip.crypto.market_regime import MarketRegimeSnapshot
from polyflip.crypto.market_regime_classifier import (
    Regime,
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
    regime: Regime
    global_confidence: float


@dataclass(frozen=True)
class PolicyConfig:
    """Configurable thresholds for policy decisions."""
    outsider_trend_multiplier: float = 0.0  # blocked in trend
    outsider_sideways_multiplier: float = 1.0
    outsider_high_vol_multiplier: float = 0.5
    outsider_mixed_multiplier: float = 0.5
    outsider_unknown_multiplier: float = 0.8

    mltrend_favor_multiplier: float = 1.0   # align with regime
    mltrend_fade_multiplier: float = 0.8     # fade: allow but reduce
    mltrend_against_multiplier: float = 0.0  # blocked
    mltrend_high_vol_multiplier: float = 0.5
    mltrend_mixed_multiplier: float = 0.5
    mltrend_unknown_multiplier: float = 0.8

    favorite_follow_multiplier: float = 1.0
    favorite_against_multiplier: float = 0.0
    favorite_unknown_multiplier: float = 0.8

    unknown_multiplier: float = 0.8  # default for unknown regime


DEFAULT_POLICY_CONFIG = PolicyConfig()


def _get_global_regime(snapshot: MarketRegimeSnapshot) -> tuple[Regime, float]:
    """Extract global regime and confidence from snapshot."""
    # Use basket global regime if available
    if snapshot.basket.history_ready:
        from polyflip.crypto.market_regime_classifier import classify_global_regime
        cls = classify_global_regime(snapshot)
        return cls.regime, cls.confidence
    return Regime.UNKNOWN, 0.0


def evaluate_policy(
    snapshot: MarketRegimeSnapshot,
    strategy: StrategyType,
    direction: float,  # +1 = bullish, -1 = bearish (predicted by ML)
    mode: FilterMode = FilterMode.SHADOW,
    config: PolicyConfig | None = None,
) -> PolicyResult:
    """
    Evaluate regime-based policy for a given strategy and direction.

    Args:
        snapshot: market regime snapshot
        strategy: which strategy is being evaluated
        direction: ML-predicted direction (+1/-1)
        mode: filter mode (OFF/SHADOW/ACTIVE)
        config: policy thresholds

    Returns:
        PolicyResult with allow, multiplier, reason
    """
    cfg = config or DEFAULT_POLICY_CONFIG
    global_regime, confidence = _get_global_regime(snapshot)

    if mode == FilterMode.OFF:
        return PolicyResult(
            allow=True,
            stake_multiplier=1.0,
            reason="filter_off",
            regime=global_regime,
            global_confidence=confidence,
        )

    # ── OUTSIDER strategy ─────────────────────────────────────
    if strategy == StrategyType.OUTSIDER:
        if global_regime == Regime.SIDEWAYS:
            return PolicyResult(
                allow=True,
                stake_multiplier=cfg.outsider_sideways_multiplier,
                reason="outsider_in_sideways",
                regime=global_regime,
                global_confidence=confidence,
            )
        if global_regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            return PolicyResult(
                allow=cfg.outsider_trend_multiplier > 0,
                stake_multiplier=cfg.outsider_trend_multiplier,
                reason="outsider_blocked_in_trend",
                regime=global_regime,
                global_confidence=confidence,
            )
        if global_regime == Regime.HIGH_VOL_CHOP:
            return PolicyResult(
                allow=True,
                stake_multiplier=cfg.outsider_high_vol_multiplier,
                reason="outsider_reduced_in_high_vol",
                regime=global_regime,
                global_confidence=confidence,
            )
        if global_regime == Regime.MIXED:
            return PolicyResult(
                allow=True,
                stake_multiplier=cfg.outsider_mixed_multiplier,
                reason="outsider_reduced_in_mixed",
                regime=global_regime,
                global_confidence=confidence,
            )
        # UNKNOWN
        return PolicyResult(
            allow=True,
            stake_multiplier=cfg.outsider_unknown_multiplier,
            reason="outsider_reduced_in_unknown",
            regime=global_regime,
            global_confidence=confidence,
        )

    # ── ML_TREND strategies ───────────────────────────────────
    if strategy in (StrategyType.ML_TREND_FOLLOW, StrategyType.ML_TREND_FADE):
        # Check if ML direction aligns with regime
        regime_direction = 1.0 if global_regime == Regime.TREND_UP else (
            -1.0 if global_regime == Regime.TREND_DOWN else 0.0
        )

        aligns = (direction * regime_direction > 0) if regime_direction != 0 else False

        if global_regime == Regime.HIGH_VOL_CHOP:
            mult = cfg.mltrend_high_vol_multiplier
            reason = "mltrend_reduced_high_vol"
        elif global_regime == Regime.MIXED:
            mult = cfg.mltrend_mixed_multiplier
            reason = "mltrend_reduced_mixed"
        elif global_regime == Regime.UNKNOWN:
            mult = cfg.mltrend_unknown_multiplier
            reason = "mltrend_reduced_unknown"
        elif global_regime == Regime.SIDEWAYS:
            # In sideways, ML trend is less reliable
            mult = cfg.mltrend_mixed_multiplier
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
            regime=global_regime,
            global_confidence=confidence,
        )

    # ── ML_FAVORITE strategy ──────────────────────────────────
    if strategy == StrategyType.ML_FAVORITE:
        regime_direction = 1.0 if global_regime == Regime.TREND_UP else (
            -1.0 if global_regime == Regime.TREND_DOWN else 0.0
        )
        aligns = (direction * regime_direction > 0) if regime_direction != 0 else False

        if global_regime == Regime.UNKNOWN:
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
            regime=global_regime,
            global_confidence=confidence,
        )

    # ── OTHER: no regime policy ───────────────────────────────
    return PolicyResult(
        allow=True,
        stake_multiplier=1.0,
        reason="unknown_strategy_no_policy",
        regime=global_regime,
        global_confidence=confidence,
    )
