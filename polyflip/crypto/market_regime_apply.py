"""
Apply regime policy v2 — bridges snapshot + policy into decision pipeline.

Fixes from review:
- BUY_NO strategy uses 1 - fresh_yes_price for price classification
- lgbm_applied passed through from caller
- UI thresholds from TradingConfig passed into RegimeConfig
- Audit contains both global_phase and asset_phase
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from polyflip.crypto.market_regime import MarketRegimeSnapshot, MIN_HISTORY_CANDLES
from polyflip.crypto.market_regime_classifier import MarketPhase, classify_global_regime, classify_asset_regime, RegimeConfig
from polyflip.crypto.market_regime_policy import (
    FilterMode,
    PolicyConfig,
    PolicyResult,
    RegimeGateResult,
    StrategyType,
    VetoGateConfig,
    evaluate_veto_gate,
    evaluate_policy,
)
from polyflip.crypto.market_regime_audit import serialize_regime_audit
from polyflip.trading.trading_config import TradingConfig

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RegimeDecisionOutcome:
    """Result of regime policy application."""
    regime_snapshot: MarketRegimeSnapshot | None
    policy_result: PolicyResult | None
    audit_dict: dict[str, Any] | None
    applied: bool
    original_bet_size: float
    adjusted_bet_size: float
    original_action: str
    adjusted_action: str
    skip_reason: str | None = None
    global_phase: str | None = None
    asset_phase: str | None = None
    gate_result: RegimeGateResult | None = None
    policy_version: int = 2


def _determine_strategy_type(
    candidate_side: str | None,
    fresh_yes_price: float,
    lgbm_applied: bool,
) -> StrategyType:
    """
    Map decision context to strategy type.

    FIX: BUY_NO uses 1 - fresh_yes_price to determine favorite/outsider
    for the actual side being purchased.
    """
    if candidate_side is None:
        return StrategyType.OTHER

    if lgbm_applied:
        # With LightGBM: follow = same direction as model, fade = opposite
        if candidate_side == "BUY_YES" and fresh_yes_price < 0.50:
            return StrategyType.ML_TREND_FOLLOW
        if candidate_side == "BUY_NO" and (1 - fresh_yes_price) < 0.50:
            return StrategyType.ML_TREND_FOLLOW
        return StrategyType.ML_TREND_FADE

    # Without LightGBM: classify by price of the side being bought
    if candidate_side == "BUY_YES":
        candidate_price = fresh_yes_price
    else:
        # BUY_NO: the price of NO = 1 - YES price
        candidate_price = 1.0 - fresh_yes_price

    if candidate_price < 0.50:
        return StrategyType.OUTSIDER
    return StrategyType.ML_FAVORITE


def _build_regime_config(cfg: TradingConfig) -> RegimeConfig:
    """
    Build RegimeConfig from TradingConfig, passing through UI thresholds.
    MRF-FIX-06 + Step 3: wire thresholds, but keep ret_norm_cap separate from
    trend_efficiency_min (they serve different purposes).
    """
    return RegimeConfig(
        trend_efficiency_min=cfg.mrf_efficiency_threshold,
        breadth_strong_threshold=cfg.mrf_breadth_threshold,
        breadth_weak_threshold=1.0 - cfg.mrf_breadth_threshold,
    )


def apply_regime_policy(
    cfg: TradingConfig,
    snapshot: MarketRegimeSnapshot,
    candidate_side: str | None,
    fresh_yes_price: float,
    lgbm_applied: bool,
    bet_size_usdc: float,
    action: str,
    decision_run_id: str = "",
    asset_symbol: str = "",
) -> RegimeDecisionOutcome:
    """
    Apply regime policy to a decision.

    SHADOW: compute and log only.
    ACTIVE: apply multiplier to bet_size_usdc.
    OFF: return unmodified sizes.
    """
    mrf_mode_str = cfg.mrf_mode
    mode = FilterMode(mrf_mode_str) if mrf_mode_str in ("OFF", "SHADOW", "ACTIVE") else FilterMode.OFF

    if mode == FilterMode.OFF or not snapshot.basket.history_ready:
        return RegimeDecisionOutcome(
            regime_snapshot=snapshot,
            policy_result=None,
            audit_dict=None,
            applied=False,
            original_bet_size=bet_size_usdc,
            adjusted_bet_size=bet_size_usdc,
            original_action=action,
            adjusted_action=action,
            policy_version=cfg.mrf_version,
        )

    # Build regime config from UI thresholds
    regime_cfg = _build_regime_config(cfg)

    strategy_type = _determine_strategy_type(candidate_side, fresh_yes_price, lgbm_applied)
    direction = 1.0 if candidate_side == "BUY_YES" else -1.0

    # Build policy config from TradingConfig
    policy_cfg = PolicyConfig(
        outsider_trend_multiplier=cfg.mrf_outsider_trend_multiplier,
        unknown_multiplier=cfg.mrf_unknown_multiplier,
    )

    policy_result = evaluate_policy(
        snapshot, strategy_type, direction, mode, config=policy_cfg,
        regime_config=regime_cfg,
    )

    # Get asset phase for the current decision (MRF-FIX-07: use explicit asset_symbol)
    asset_phase_str = "UNKNOWN"
    if asset_symbol and snapshot.assets:
        asset_feat = snapshot.assets.get(asset_symbol)
        if asset_feat and asset_feat.history_ready:
            asset_cls = classify_asset_regime(asset_feat, regime_cfg)
            asset_phase_str = asset_cls.phase.value

    # Compute adjusted bet size
    adjusted_bet_size = bet_size_usdc * policy_result.stake_multiplier
    adjusted_action = action
    skip_reason = None

    if not policy_result.allow and action != "SKIP":
        adjusted_action = "SKIP"
        adjusted_bet_size = 0.0
        skip_reason = f"MRF:{policy_result.phase.value}:{policy_result.reason}"
        logger.info(
            "mrf_policy_blocked",
            decision_run_id=decision_run_id,
            phase=policy_result.phase.value,
            strategy=strategy_type.value,
            reason=policy_result.reason,
            original_action=action,
        )

    # Audit (always, regardless of mode) — uses same RegimeConfig for consistency
    applied = (mode == FilterMode.ACTIVE and policy_result.stake_multiplier != 1.0)
    audit_dict = serialize_regime_audit(
        snapshot, policy_result, mode, cfg.mrf_version,
        strategy_type=strategy_type.value,
        applied=applied,
        regime_config=regime_cfg,
    )

    if mode == FilterMode.SHADOW:
        # SHADOW: log but return unmodified sizes
        logger.info(
            "mrf_shadow_log",
            decision_run_id=decision_run_id,
            phase=policy_result.phase.value,
            strategy=strategy_type.value,
            multiplier=policy_result.stake_multiplier,
            original_bet_size=bet_size_usdc,
            adjusted_bet_size=adjusted_bet_size,
        )
        return RegimeDecisionOutcome(
            regime_snapshot=snapshot,
            policy_result=policy_result,
            audit_dict=audit_dict,
            applied=False,
            original_bet_size=bet_size_usdc,
            adjusted_bet_size=bet_size_usdc,
            original_action=action,
            adjusted_action=action,
            global_phase=policy_result.phase.value,
            asset_phase=asset_phase_str,
            policy_version=cfg.mrf_version,
        )

    # ACTIVE: apply multiplier
    return RegimeDecisionOutcome(
        regime_snapshot=snapshot,
        policy_result=policy_result,
        audit_dict=audit_dict,
        applied=applied,
        original_bet_size=bet_size_usdc,
        adjusted_bet_size=adjusted_bet_size,
        original_action=action,
        adjusted_action=adjusted_action,
        skip_reason=skip_reason,
        global_phase=policy_result.phase.value,
        asset_phase=asset_phase_str,
        policy_version=cfg.mrf_version,
    )


def apply_regime_veto_gate(
    *,
    cfg: TradingConfig,
    snapshot: MarketRegimeSnapshot,
    asset_symbol: str,
    candidate_side: str,
    candidate_ask: float,
    net_edge: float,
    min_edge_used: float,
    bet_size_usdc: float,
    action: str,
    decision_run_id: str = "",
) -> RegimeDecisionOutcome:
    """Apply MRF v3's binary veto gate without resizing an allowed bet."""
    mode = FilterMode(cfg.mrf_mode)
    candidate_direction = {
        "BUY_YES": 1.0,
        "BUY_NO": -1.0,
    }.get(candidate_side)
    if candidate_direction is None:
        raise ValueError(f"Unsupported MRF candidate side: {candidate_side}")

    regime_cfg = _build_regime_config(cfg)
    gate_cfg = VetoGateConfig(
        asset_weight=cfg.mrf_asset_weight,
        global_weight=cfg.mrf_global_weight,
        veto_threshold=cfg.mrf_veto_threshold,
        edge_override_margin=cfg.mrf_edge_override_margin,
    )
    gate_result = evaluate_veto_gate(
        snapshot=snapshot,
        asset_symbol=asset_symbol,
        candidate_direction=candidate_direction,
        net_edge=net_edge,
        min_edge_used=min_edge_used,
        config=gate_cfg,
        regime_config=regime_cfg,
    )

    effective_block = (
        mode == FilterMode.ACTIVE
        and gate_result.would_block
        and action != "SKIP"
    )
    adjusted_action = "SKIP" if effective_block else action
    adjusted_bet_size = 0.0 if effective_block else bet_size_usdc
    skip_reason = None
    if effective_block:
        skip_reason = (
            f"MRF:V3:{gate_result.global_phase.value}:"
            f"{gate_result.asset_phase.value}:{gate_result.reason}"
        )
        logger.info(
            "mrf_v3_blocked",
            decision_run_id=decision_run_id,
            reason=gate_result.reason,
            regime_evidence=gate_result.regime_evidence,
            edge_margin=gate_result.edge_margin,
        )

    audit_dict = serialize_regime_audit(
        snapshot=snapshot,
        policy_result=None,
        mode=mode,
        mrf_version=3,
        applied=effective_block,
        regime_config=regime_cfg,
        gate_result=gate_result,
        effective_block=effective_block,
        candidate_role=(
            "OUTSIDER"
            if candidate_ask is not None and candidate_ask < 0.50
            else "FAVORITE"
        ),
    )

    # SHADOW computes and records the gate but leaves the decision untouched.
    return RegimeDecisionOutcome(
        regime_snapshot=snapshot,
        policy_result=None,
        audit_dict=audit_dict,
        applied=effective_block,
        original_bet_size=bet_size_usdc,
        adjusted_bet_size=adjusted_bet_size,
        original_action=action,
        adjusted_action=adjusted_action,
        skip_reason=skip_reason,
        global_phase=gate_result.global_phase.value,
        asset_phase=gate_result.asset_phase.value,
        gate_result=gate_result,
        policy_version=3,
    )


def apply_market_regime_filter(
    *,
    cfg: TradingConfig,
    snapshot: MarketRegimeSnapshot,
    candidate_side: str | None,
    candidate_ask: float,
    fresh_yes_price: float,
    lgbm_applied: bool,
    bet_size_usdc: float,
    net_edge: float,
    min_edge_used: float,
    action: str,
    decision_run_id: str = "",
    asset_symbol: str = "",
) -> RegimeDecisionOutcome:
    """Dispatch MRF version while preserving the v1/v2 contract."""
    if cfg.mrf_version not in (1, 2, 3):
        raise ValueError(f"Unsupported MRF policy version: {cfg.mrf_version}")
    # OFF is a hard short-circuit for every policy version.  In particular,
    # v3 must not classify a snapshot or emit gate telemetry while disabled.
    if cfg.mrf_mode == "OFF":
        return apply_regime_policy(
            cfg=cfg,
            snapshot=snapshot,
            candidate_side=candidate_side,
            fresh_yes_price=fresh_yes_price,
            lgbm_applied=lgbm_applied,
            bet_size_usdc=bet_size_usdc,
            action=action,
            decision_run_id=decision_run_id,
            asset_symbol=asset_symbol,
        )
    if cfg.mrf_version in (1, 2):
        return apply_regime_policy(
            cfg=cfg,
            snapshot=snapshot,
            candidate_side=candidate_side,
            fresh_yes_price=fresh_yes_price,
            lgbm_applied=lgbm_applied,
            bet_size_usdc=bet_size_usdc,
            action=action,
            decision_run_id=decision_run_id,
            asset_symbol=asset_symbol,
        )
    if cfg.mrf_version == 3:
        if candidate_side is None:
            raise ValueError("MRF v3 requires candidate_side")
        return apply_regime_veto_gate(
            cfg=cfg,
            snapshot=snapshot,
            asset_symbol=asset_symbol,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            net_edge=net_edge,
            min_edge_used=min_edge_used,
            bet_size_usdc=bet_size_usdc,
            action=action,
            decision_run_id=decision_run_id,
        )
    raise AssertionError("unreachable MRF version dispatch")
