"""
Apply regime policy to decision flow (T11 of MRF plan).

Bridges the regime snapshot + policy result into the actual decision pipeline.
SHADOW: compute and log only. ACTIVE: apply multiplier to stake_size.
Does NOT touch entry model, funding veto, stop-loss, settlement, or daily limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from polyflip.crypto.market_regime import MarketRegimeSnapshot, MIN_HISTORY_CANDLES
from polyflip.crypto.market_regime_classifier import classify_global_regime, Regime
from polyflip.crypto.market_regime_policy import (
    FilterMode,
    PolicyConfig,
    PolicyResult,
    StrategyType,
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


def _determine_strategy_type(
    candidate_side: str | None,
    fresh_yes_price: float,
    lgbm_applied: bool,
) -> StrategyType:
    """Map decision context to strategy type for regime policy."""
    if candidate_side is None:
        return StrategyType.OTHER

    if lgbm_applied:
        if candidate_side == "BUY_YES" and fresh_yes_price < 0.50:
            return StrategyType.ML_TREND_FOLLOW
        if candidate_side == "BUY_NO" and fresh_yes_price >= 0.50:
            return StrategyType.ML_TREND_FOLLOW
        return StrategyType.ML_TREND_FADE

    # Without LightGBM, classify by price level
    if fresh_yes_price < 0.50:
        return StrategyType.OUTSIDER
    return StrategyType.ML_FAVORITE


def apply_regime_policy(
    cfg: TradingConfig,
    snapshot: MarketRegimeSnapshot,
    candidate_side: str | None,
    fresh_yes_price: float,
    lgbm_applied: bool,
    bet_size_usdc: float,
    action: str,
    decision_run_id: str = "",
) -> RegimeDecisionOutcome:
    """
    Apply regime policy to a decision.

    In SHADOW mode: compute policy, log, return unmodified sizes.
    In ACTIVE mode: apply multiplier to bet_size_usdc.
    In OFF mode: return unmodified sizes.

    Does NOT modify: entry logic, stop-loss, funding, daily limits.
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
        )

    strategy_type = _determine_strategy_type(candidate_side, fresh_yes_price, lgbm_applied)

    # Direction for ML_TREND strategies
    direction = 1.0 if candidate_side == "BUY_YES" else -1.0

    # Build policy config from TradingConfig
    policy_cfg = PolicyConfig(
        outsider_trend_multiplier=cfg.mrf_outsider_trend_multiplier,
        unknown_multiplier=cfg.mrf_unknown_multiplier,
    )

    policy_result = evaluate_policy(
        snapshot, strategy_type, direction, mode, config=policy_cfg,
    )

    # Compute adjusted bet size
    adjusted_bet_size = bet_size_usdc * policy_result.stake_multiplier
    adjusted_action = action

    if not policy_result.allow and action != "SKIP":
        adjusted_action = "SKIP"
        logger.info(
            "mrf_policy_blocked",
            decision_run_id=decision_run_id,
            regime=policy_result.regime.value,
            strategy=strategy_type.value,
            reason=policy_result.reason,
            original_action=action,
        )

    # Audit
    applied = (mode == FilterMode.ACTIVE and policy_result.stake_multiplier != 1.0)
    audit_dict = serialize_regime_audit(
        snapshot, policy_result, mode, cfg.mrf_version,
        strategy_type=strategy_type.value,
        applied=applied,
    )

    if mode == FilterMode.SHADOW:
        logger.info(
            "mrf_shadow_log",
            decision_run_id=decision_run_id,
            regime=policy_result.regime.value,
            strategy=strategy_type.value,
            multiplier=policy_result.stake_multiplier,
            original_bet_size=bet_size_usdc,
            adjusted_bet_size=adjusted_bet_size,
        )
        # SHADOW: return unmodified sizes
        return RegimeDecisionOutcome(
            regime_snapshot=snapshot,
            policy_result=policy_result,
            audit_dict=audit_dict,
            applied=False,
            original_bet_size=bet_size_usdc,
            adjusted_bet_size=bet_size_usdc,
            original_action=action,
            adjusted_action=action,
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
    )
