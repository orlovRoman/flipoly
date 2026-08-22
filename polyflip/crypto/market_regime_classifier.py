"""
Market regime classifier v2 — per-asset + global phases with strength scoring.

Replaces old Regime enum with MarketPhase (STRONG_UP..UNKNOWN).
Uses magnitude/consistency/strength scoring for directional classification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from polyflip.crypto.market_regime import (
    AssetRegimeFeatures,
    BasketRegimeFeatures,
    MarketRegimeSnapshot,
)


class MarketPhase(str, Enum):
    STRONG_UP = "STRONG_UP"
    WEAK_UP = "WEAK_UP"
    SIDEWAYS = "SIDEWAYS"
    WEAK_DOWN = "WEAK_DOWN"
    STRONG_DOWN = "STRONG_DOWN"
    HIGH_VOL_CHOP = "HIGH_VOL_CHOP"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


# Backward compat aliases
Regime = MarketPhase
Regime.TREND_UP = MarketPhase.STRONG_UP  # type: ignore[attr-defined]
Regime.TREND_DOWN = MarketPhase.STRONG_DOWN  # type: ignore[attr-defined]
# Mapping from old regime names to new MarketPhase values
REGIME_COMPAT_MAP = {
    "TREND_UP": "STRONG_UP",
    "TREND_DOWN": "STRONG_DOWN",
    "TREND_UP_STRONG": "STRONG_UP",
    "TREND_UP_WEAK": "WEAK_UP",
    "TREND_DOWN_STRONG": "STRONG_DOWN",
    "TREND_DOWN_WEAK": "WEAK_DOWN",
}


@dataclass(frozen=True)
class RegimeClassification:
    """Result of regime classification."""
    phase: MarketPhase
    strength: float   # 0.0 .. 1.0
    confidence: float  # 0.0 .. 1.0
    direction: float   # -1.0 .. +1.0
    reason_codes: list[str]

    @property
    def regime(self) -> MarketPhase:
        """Deprecated: use .phase instead."""
        return self.phase

    @property
    def regime_value(self) -> str:
        """Deprecated: use .phase.value instead."""
        return self.phase.value


@dataclass(frozen=True)
class RegimeConfig:
    """Configurable thresholds for regime classification."""
    trend_ret_threshold: float = 0.02
    sideways_ret_max: float = 0.005
    trend_efficiency_min: float = 0.4
    sideways_efficiency_max: float = 0.3
    high_vol_efficiency_max: float = 0.25
    high_vol_ratio_threshold: float = 1.5
    breadth_strong_threshold: float = 0.65
    breadth_weak_threshold: float = 0.35
    dispersion_high_threshold: float = 0.02
    # Strength thresholds
    strong_score_threshold: float = 0.6
    weak_score_threshold: float = 0.2
    # Normalization caps for returns
    ret_norm_cap: float = 0.10  # 10%


DEFAULT_CONFIG = RegimeConfig()


def _normalize_ret(ret: float, cap: float) -> float:
    """Normalize abs return to [0, 1] using tanh-like scaling."""
    if cap <= 0:
        return 0.0
    x = abs(ret) / cap
    return math.tanh(x)


def classify_strength(
    ret_4h: float,
    ret_12h: float,
    ret_24h: float,
    efficiency_24h: float,
    config: RegimeConfig | None = None,
) -> tuple[float, float, float]:
    """
    Compute magnitude, consistency, and strength scores.

    Returns (magnitude_score, consistency_score, strength_score).
    """
    cfg = config or DEFAULT_CONFIG
    cap = cfg.ret_norm_cap

    magnitude_score = (
        0.50 * _normalize_ret(ret_4h, cap)
        + 0.30 * _normalize_ret(ret_12h, cap)
        + 0.20 * _normalize_ret(ret_24h, cap)
    )

    # Consistency: fraction of horizons with same sign
    rets = [ret_4h, ret_12h, ret_24h]
    n_positive = sum(1 for r in rets if r > 0)
    n_negative = sum(1 for r in rets if r < 0)
    consistency_score = max(n_positive, n_negative) / len(rets)

    strength_score = (
        magnitude_score
        * (0.5 + 0.5 * efficiency_24h)
        * consistency_score
    )

    return magnitude_score, consistency_score, strength_score


def classify_asset_regime(
    features: AssetRegimeFeatures,
    config: RegimeConfig | None = None,
) -> RegimeClassification:
    """
    Classify regime for a single asset based on features.
    Returns phase, strength, confidence, direction, reason codes.
    """
    cfg = config or DEFAULT_CONFIG
    reasons: list[str] = []

    if not features.history_ready:
        return RegimeClassification(
            phase=MarketPhase.UNKNOWN,
            strength=0.0,
            confidence=0.0,
            direction=0.0,
            reason_codes=["insufficient_history"],
        )

    ret_24h = features.ret_24h
    eff = features.efficiency_24h
    vol_ratio = features.vol_ratio

    direction = max(-1.0, min(1.0, ret_24h / cfg.trend_ret_threshold)) if cfg.trend_ret_threshold else 0.0

    _, _, strength = classify_strength(
        features.ret_4h, features.ret_12h, features.ret_24h, eff, cfg,
    )

    # ── HIGH_VOL_CHOP ──
    if vol_ratio > cfg.high_vol_ratio_threshold and eff < cfg.high_vol_efficiency_max:
        reasons.append(f"high_vol_ratio:{vol_ratio:.2f}")
        reasons.append(f"low_efficiency:{eff:.2f}")
        return RegimeClassification(
            phase=MarketPhase.HIGH_VOL_CHOP,
            strength=strength,
            confidence=min(1.0, vol_ratio / 2.0),
            direction=direction,
            reason_codes=reasons,
        )

    # ── TREND classification ──
    rets = [features.ret_4h, features.ret_12h, features.ret_24h]
    n_pos = sum(1 for r in rets if r > 0)
    n_neg = sum(1 for r in rets if r < 0)
    consistency = max(n_pos, n_neg) / len(rets)

    if n_pos == 3 and consistency >= 2 / 3 and strength >= cfg.strong_score_threshold and eff >= cfg.trend_efficiency_min:
        reasons.append(f"ret_24h:{ret_24h:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        reasons.append(f"consistency:{consistency:.2f}")
        reasons.append(f"efficiency:{eff:.2f}")
        return RegimeClassification(
            phase=MarketPhase.STRONG_UP,
            strength=strength,
            confidence=min(1.0, eff * consistency),
            direction=1.0,
            reason_codes=reasons,
        )

    if n_neg == 3 and consistency >= 2 / 3 and strength >= cfg.strong_score_threshold and eff >= cfg.trend_efficiency_min:
        reasons.append(f"ret_24h:{ret_24h:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        reasons.append(f"consistency:{consistency:.2f}")
        return RegimeClassification(
            phase=MarketPhase.STRONG_DOWN,
            strength=strength,
            confidence=min(1.0, eff * consistency),
            direction=-1.0,
            reason_codes=reasons,
        )

    if n_pos > n_neg and cfg.weak_score_threshold <= strength < cfg.strong_score_threshold:
        reasons.append(f"ret_24h:{ret_24h:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        return RegimeClassification(
            phase=MarketPhase.WEAK_UP,
            strength=strength,
            confidence=min(1.0, eff * consistency),
            direction=direction,
            reason_codes=reasons,
        )

    if n_neg > n_pos and cfg.weak_score_threshold <= strength < cfg.strong_score_threshold:
        reasons.append(f"ret_24h:{ret_24h:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        return RegimeClassification(
            phase=MarketPhase.WEAK_DOWN,
            strength=strength,
            confidence=min(1.0, eff * consistency),
            direction=direction,
            reason_codes=reasons,
        )

    # ── SIDEWAYS ──
    if abs(ret_24h) < cfg.sideways_ret_max and eff < cfg.sideways_efficiency_max:
        reasons.append(f"ret_24h:{ret_24h:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        return RegimeClassification(
            phase=MarketPhase.SIDEWAYS,
            strength=strength,
            confidence=min(1.0, (cfg.sideways_ret_max - abs(ret_24h)) / cfg.sideways_ret_max),
            direction=0.0,
            reason_codes=reasons,
        )

    # ── WEAK_UP/DOWN with low strength ──
    if n_pos > n_neg:
        reasons.append(f"weak_up_low_strength:{strength:.2f}")
        return RegimeClassification(
            phase=MarketPhase.WEAK_UP,
            strength=strength,
            confidence=0.3,
            direction=direction,
            reason_codes=reasons,
        )
    if n_neg > n_pos:
        reasons.append(f"weak_down_low_strength:{strength:.2f}")
        return RegimeClassification(
            phase=MarketPhase.WEAK_DOWN,
            strength=strength,
            confidence=0.3,
            direction=direction,
            reason_codes=reasons,
        )

    # ── MIXED / fallback ──
    reasons.append(f"ret_24h:{ret_24h:.4f}")
    reasons.append(f"efficiency:{eff:.2f}")
    reasons.append("contradictory_signals")
    return RegimeClassification(
        phase=MarketPhase.MIXED,
        strength=strength,
        confidence=0.3,
        direction=direction,
        reason_codes=reasons,
    )


def classify_global_regime(
    snapshot: MarketRegimeSnapshot,
    config: RegimeConfig | None = None,
) -> RegimeClassification:
    """
    Classify global regime from basket features.
    Uses cross-asset signals rather than single-asset view.
    """
    cfg = config or DEFAULT_CONFIG
    reasons: list[str] = []
    basket = snapshot.basket

    if not basket.history_ready:
        return RegimeClassification(
            phase=MarketPhase.UNKNOWN,
            strength=0.0,
            confidence=0.0,
            direction=0.0,
            reason_codes=["no_assets_ready"],
        )

    ret = basket.median_ret_24h
    eff = basket.market_efficiency_24h
    breadth = basket.breadth_up_24h
    disp = basket.dispersion_24h

    direction = max(-1.0, min(1.0, ret / cfg.trend_ret_threshold)) if cfg.trend_ret_threshold else 0.0

    # Strength for global
    _, _, strength = classify_strength(
        basket.median_ret_4h, basket.median_ret_12h, basket.median_ret_24h, eff, cfg,
    )

    # ── High dispersion = MIXED ──
    if disp > cfg.dispersion_high_threshold:
        reasons.append(f"high_dispersion:{disp:.4f}")
        return RegimeClassification(
            phase=MarketPhase.MIXED,
            strength=strength,
            confidence=min(1.0, disp / cfg.dispersion_high_threshold),
            direction=direction,
            reason_codes=reasons,
        )

    # ── HIGH_VOL_CHOP (global) ──
    # Use per-asset vol_ratio median if available
    ready_assets = [a for a in snapshot.assets.values() if a.history_ready]
    if ready_assets:
        median_vol_ratio = sorted([a.vol_ratio for a in ready_assets])[len(ready_assets) // 2]
    else:
        median_vol_ratio = 0.0

    if median_vol_ratio > cfg.high_vol_ratio_threshold and eff < cfg.high_vol_efficiency_max:
        reasons.append(f"global_high_vol:{median_vol_ratio:.2f}")
        return RegimeClassification(
            phase=MarketPhase.HIGH_VOL_CHOP,
            strength=strength,
            confidence=min(1.0, median_vol_ratio / 2.0),
            direction=direction,
            reason_codes=reasons,
        )

    # ── STRONG_UP ──
    rets = [basket.median_ret_4h, basket.median_ret_12h, basket.median_ret_24h]
    n_pos = sum(1 for r in rets if r > 0)
    n_neg = sum(1 for r in rets if r < 0)
    consistency = max(n_pos, n_neg) / len(rets) if rets else 0.0

    if (n_pos == 3 and consistency >= 2 / 3 and strength >= cfg.strong_score_threshold
            and breadth > cfg.breadth_strong_threshold and eff >= cfg.trend_efficiency_min):
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        reasons.append(f"breadth:{breadth:.2f}")
        reasons.append(f"efficiency:{eff:.2f}")
        return RegimeClassification(
            phase=MarketPhase.STRONG_UP,
            strength=strength,
            confidence=min(1.0, eff * breadth),
            direction=1.0,
            reason_codes=reasons,
        )

    # ── STRONG_DOWN ──
    if (n_neg == 3 and consistency >= 2 / 3 and strength >= cfg.strong_score_threshold
            and breadth < cfg.breadth_weak_threshold and eff >= cfg.trend_efficiency_min):
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        reasons.append(f"breadth:{breadth:.2f}")
        return RegimeClassification(
            phase=MarketPhase.STRONG_DOWN,
            strength=strength,
            confidence=min(1.0, eff * (1 - breadth)),
            direction=-1.0,
            reason_codes=reasons,
        )

    # ── WEAK_UP ──
    if n_pos > n_neg and cfg.weak_score_threshold <= strength < cfg.strong_score_threshold:
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        return RegimeClassification(
            phase=MarketPhase.WEAK_UP,
            strength=strength,
            confidence=min(1.0, eff * consistency),
            direction=direction,
            reason_codes=reasons,
        )

    # ── WEAK_DOWN ──
    if n_neg > n_pos and cfg.weak_score_threshold <= strength < cfg.strong_score_threshold:
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"strength:{strength:.2f}")
        return RegimeClassification(
            phase=MarketPhase.WEAK_DOWN,
            strength=strength,
            confidence=min(1.0, eff * consistency),
            direction=direction,
            reason_codes=reasons,
        )

    # ── SIDEWAYS ──
    if abs(ret) < cfg.sideways_ret_max and eff < cfg.sideways_efficiency_max:
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        return RegimeClassification(
            phase=MarketPhase.SIDEWAYS,
            strength=strength,
            confidence=min(1.0, (cfg.sideways_ret_max - abs(ret)) / cfg.sideways_ret_max),
            direction=0.0,
            reason_codes=reasons,
        )

    # ── MIXED / fallback ──
    reasons.append(f"median_ret:{ret:.4f}")
    reasons.append(f"efficiency:{eff:.2f}")
    reasons.append("no_clear_regime")
    return RegimeClassification(
        phase=MarketPhase.MIXED,
        strength=strength,
        confidence=0.3,
        direction=direction,
        reason_codes=reasons,
    )
