"""
Rule-based market regime classifier (T06 of MRF plan).

Classifies global and per-asset regime from MarketRegimeSnapshot features.
Pure, deterministic, no DB reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from polyflip.crypto.market_regime import (
    AssetRegimeFeatures,
    BasketRegimeFeatures,
    MarketRegimeSnapshot,
)


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL_CHOP = "HIGH_VOL_CHOP"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RegimeClassification:
    """Result of regime classification."""
    regime: Regime
    confidence: float  # 0.0 .. 1.0
    direction: float   # -1.0 .. +1.0 (negative = bearish)
    reason_codes: list[str]


@dataclass(frozen=True)
class RegimeConfig:
    """Configurable thresholds for regime classification."""
    # Return thresholds (log-return over 24h)
    trend_ret_threshold: float = 0.02      # ~2% move
    sideways_ret_max: float = 0.005        # <0.5% = sideways
    # Efficiency thresholds
    trend_efficiency_min: float = 0.4      # need directional movement
    sideways_efficiency_max: float = 0.3   # low efficiency = noise
    high_vol_efficiency_max: float = 0.25  # very low efficiency with high vol
    # Volatility
    high_vol_ratio_threshold: float = 1.5  # vol_4h / vol_24h
    # Breadth
    breadth_strong_threshold: float = 0.65 # >65% same direction
    breadth_weak_threshold: float = 0.35   # <35% = opposite direction
    # Cross-asset
    dispersion_high_threshold: float = 0.02  # high disagreement


DEFAULT_CONFIG = RegimeConfig()


def classify_asset_regime(
    features: AssetRegimeFeatures,
    config: RegimeConfig | None = None,
) -> RegimeClassification:
    """
    Classify regime for a single asset based on its features.

    Returns regime, confidence (0-1), direction (-1..+1), and reason codes.
    """
    cfg = config or DEFAULT_CONFIG
    reasons: list[str] = []

    if not features.history_ready:
        return RegimeClassification(
            regime=Regime.UNKNOWN,
            confidence=0.0,
            direction=0.0,
            reason_codes=["insufficient_history"],
        )

    ret = features.ret_24h
    eff = features.efficiency_24h
    vol_ratio = features.vol_ratio
    up_ratio = features.up_ratio_24h

    # Direction: sign of 24h return, scaled by magnitude
    direction = max(-1.0, min(1.0, ret / cfg.trend_ret_threshold)) if cfg.trend_ret_threshold else 0.0

    # ── HIGH_VOL_CHOP: high volatility + low efficiency ──────
    if vol_ratio > cfg.high_vol_ratio_threshold and eff < cfg.high_vol_efficiency_max:
        reasons.append(f"high_vol_ratio:{vol_ratio:.2f}")
        reasons.append(f"low_efficiency:{eff:.2f}")
        return RegimeClassification(
            regime=Regime.HIGH_VOL_CHOP,
            confidence=min(1.0, vol_ratio / 2.0),
            direction=direction,
            reason_codes=reasons,
        )

    # ── TREND_UP: positive return + high efficiency + bullish breadth ──
    if (ret > cfg.trend_ret_threshold
            and eff > cfg.trend_efficiency_min
            and up_ratio > cfg.breadth_strong_threshold):
        reasons.append(f"ret_24h:{ret:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        reasons.append(f"up_ratio:{up_ratio:.2f}")
        return RegimeClassification(
            regime=Regime.TREND_UP,
            confidence=min(1.0, eff * up_ratio),
            direction=1.0,
            reason_codes=reasons,
        )

    # ── TREND_DOWN: negative return + high efficiency + bearish breadth ──
    if (ret < -cfg.trend_ret_threshold
            and eff > cfg.trend_efficiency_min
            and up_ratio < cfg.breadth_weak_threshold):
        reasons.append(f"ret_24h:{ret:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        reasons.append(f"up_ratio:{up_ratio:.2f}")
        return RegimeClassification(
            regime=Regime.TREND_DOWN,
            confidence=min(1.0, eff * (1 - up_ratio)),
            direction=-1.0,
            reason_codes=reasons,
        )

    # ── SIDEWAYS: small return + low efficiency + mixed breadth ──
    if (abs(ret) < cfg.sideways_ret_max
            and eff < cfg.sideways_efficiency_max):
        reasons.append(f"ret_24h:{ret:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        return RegimeClassification(
            regime=Regime.SIDEWAYS,
            confidence=min(1.0, (cfg.sideways_ret_max - abs(ret)) / cfg.sideways_ret_max),
            direction=0.0,
            reason_codes=reasons,
        )

    # ── MIXED: contradictory signals ──
    reasons.append(f"ret_24h:{ret:.4f}")
    reasons.append(f"efficiency:{eff:.2f}")
    reasons.append(f"up_ratio:{up_ratio:.2f}")
    reasons.append("contradictory_signals")
    return RegimeClassification(
        regime=Regime.MIXED,
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
            regime=Regime.UNKNOWN,
            confidence=0.0,
            direction=0.0,
            reason_codes=["no_assets_ready"],
        )

    ret = basket.median_ret_24h
    eff = basket.market_efficiency_24h
    breadth = basket.breadth_up_24h
    disp = basket.dispersion_24h

    direction = max(-1.0, min(1.0, ret / cfg.trend_ret_threshold)) if cfg.trend_ret_threshold else 0.0

    # ── High dispersion = MIXED across assets ──
    if disp > cfg.dispersion_high_threshold:
        reasons.append(f"high_dispersion:{disp:.4f}")
        return RegimeClassification(
            regime=Regime.MIXED,
            confidence=min(1.0, disp / cfg.dispersion_high_threshold),
            direction=direction,
            reason_codes=reasons,
        )

    # ── TREND_UP: broad positive + efficient ──
    if (ret > cfg.trend_ret_threshold
            and eff > cfg.trend_efficiency_min
            and breadth > cfg.breadth_strong_threshold):
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        reasons.append(f"breadth:{breadth:.2f}")
        return RegimeClassification(
            regime=Regime.TREND_UP,
            confidence=min(1.0, eff * breadth),
            direction=1.0,
            reason_codes=reasons,
        )

    # ── TREND_DOWN ──
    if (ret < -cfg.trend_ret_threshold
            and eff > cfg.trend_efficiency_min
            and breadth < cfg.breadth_weak_threshold):
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        reasons.append(f"breadth:{breadth:.2f}")
        return RegimeClassification(
            regime=Regime.TREND_DOWN,
            confidence=min(1.0, eff * (1 - breadth)),
            direction=-1.0,
            reason_codes=reasons,
        )

    # ── SIDEWAYS ──
    if abs(ret) < cfg.sideways_ret_max and eff < cfg.sideways_efficiency_max:
        reasons.append(f"median_ret:{ret:.4f}")
        reasons.append(f"efficiency:{eff:.2f}")
        return RegimeClassification(
            regime=Regime.SIDEWAYS,
            confidence=min(1.0, (cfg.sideways_ret_max - abs(ret)) / cfg.sideways_ret_max),
            direction=0.0,
            reason_codes=reasons,
        )

    # ── MIXED / fallback ──
    reasons.append(f"median_ret:{ret:.4f}")
    reasons.append(f"efficiency:{eff:.2f}")
    reasons.append("no_clear_regime")
    return RegimeClassification(
        regime=Regime.MIXED,
        confidence=0.3,
        direction=direction,
        reason_codes=reasons,
    )
