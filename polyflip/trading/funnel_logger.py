"""
FunnelLogger — записывает один DecisionFunnelLog на каждый вызов decide_ml_mode.
Вызывается в конце decide_ml_mode и decide_combined_mode.
Ошибки записи логируются, но НЕ пробрасываются — торговая логика не прерывается.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import DecisionFunnelLog

logger = structlog.get_logger(__name__)


async def log_funnel(
    db: AsyncSession,
    *,
    market_id: str,
    asset: str,
    trading_mode: str,
    execution_mode: Optional[str] = None,
    used_model: Optional[str] = None,
    p_flip: Optional[float],
    edge: Optional[float],
    fresh_price: Optional[float],
    threshold_lower: Optional[float],
    threshold_upper: Optional[float],
    min_edge_used: Optional[float],
    # Гейты
    g1_model_loaded: Optional[bool]   = None,
    g2_price_fetched: Optional[bool]  = None,
    g3_dead_zone: Optional[bool]      = None,
    g4_no_flip: Optional[bool]        = None,
    g5_min_edge: Optional[bool]       = None,
    g6_price_range: Optional[bool]    = None,
    g7_crypto_confirm: Optional[bool] = None,
    g8_combined_vote: Optional[bool]  = None,
    # Veto Passport (Legacy / Primary)
    primary_model_key: Optional[str] = None,
    primary_model_version: Optional[int] = None,
    confirm_model_key: Optional[str] = None,
    confirm_model_version: Optional[int] = None,
    proposed_action: Optional[str] = None,
    proposed_price: Optional[float] = None,
    proposed_amount_usdc: Optional[float] = None,
    confirm_direction: Optional[str] = None,
    confirm_passed: Optional[bool] = None,

    # Новая телеметрия COMBINED
    decision_run_id: Optional[str] = None,
    direction_status: Optional[str] = None,
    direction_model_key: Optional[str] = None,
    direction_model_version: Optional[int] = None,
    required_direction_model_key: Optional[str] = None,
    direction_regime: Optional[str] = None,
    direction_probability: Optional[float] = None,
    direction_p_up: Optional[float] = None,
    direction_p_down: Optional[float] = None,
    direction_threshold_up: Optional[float] = None,
    direction_threshold_down: Optional[float] = None,
    direction_value: Optional[str] = None,
    entry_requested_key: Optional[str] = None,
    entry_model_key: Optional[str] = None,
    entry_model_version: Optional[int] = None,
    entry_model_phase: Optional[str] = None,
    entry_model_source: Optional[str] = None,
    entry_status: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    p_candidate_win: Optional[float] = None,
    p_logreg_win: Optional[float] = None,
    direction_discount_mult: Optional[float] = None,
    combined_dir_discount_weight: Optional[float] = None,
    candidate_side: Optional[str] = None,
    candidate_ask: Optional[float] = None,
    gross_edge: Optional[float] = None,
    cost_buffer: Optional[float] = None,
    net_edge: Optional[float] = None,
    max_acceptable_price: Optional[float] = None,
    strike_source: Optional[str] = None,
    strike_proxy: Optional[float] = None,
    underlying_price: Optional[float] = None,
    distance_to_strike_pct: Optional[float] = None,
    # P0: детальная причина сбоя Direction Model
    direction_error_detail: Optional[str] = None,

    # Итог
    final_action: str = "SKIP",
    skip_reason: Optional[str] = None,
) -> None:
    try:
        row = DecisionFunnelLog(
            created_at=datetime.now(timezone.utc),
            market_id=market_id,
            asset=asset,
            trading_mode=trading_mode,
            execution_mode=execution_mode,
            used_model=used_model,
            p_flip=p_flip,
            edge=edge,
            fresh_price=fresh_price,
            threshold_lower=threshold_lower,
            threshold_upper=threshold_upper,
            min_edge_used=min_edge_used,
            g1_model_loaded=g1_model_loaded,
            g2_price_fetched=g2_price_fetched,
            g3_dead_zone=g3_dead_zone,
            g4_no_flip=g4_no_flip,
            g5_min_edge=g5_min_edge,
            g6_price_range=g6_price_range,
            g7_crypto_confirm=g7_crypto_confirm,
            g8_combined_vote=g8_combined_vote,
            primary_model_key=primary_model_key,
            primary_model_version=primary_model_version,
            confirm_model_key=confirm_model_key,
            confirm_model_version=confirm_model_version,
            proposed_action=proposed_action,
            proposed_price=proposed_price,
            proposed_amount_usdc=proposed_amount_usdc,
            confirm_direction=confirm_direction,
            confirm_passed=confirm_passed,

            decision_run_id=decision_run_id,
            direction_status=direction_status,
            direction_model_key=direction_model_key,
            direction_model_version=direction_model_version,
            direction_regime=direction_regime,
            direction_probability=direction_probability,
            direction_p_up=direction_p_up,
            direction_p_down=direction_p_down,
            direction_threshold_up=direction_threshold_up,
            direction_threshold_down=direction_threshold_down,
            direction_value=direction_value,
            entry_requested_key=entry_requested_key,
            entry_model_key=entry_model_key,
            entry_model_version=entry_model_version,
            entry_model_phase=entry_model_phase,
            entry_model_source=entry_model_source,
            entry_status=entry_status,
            fallback_reason=fallback_reason,
            p_candidate_win=p_candidate_win,
            p_logreg_win=p_logreg_win,
            direction_discount_mult=direction_discount_mult,
            combined_dir_discount_weight=combined_dir_discount_weight,
            candidate_side=candidate_side,
            candidate_ask=candidate_ask,
            gross_edge=gross_edge,
            cost_buffer=cost_buffer,
            net_edge=net_edge,
            max_acceptable_price=max_acceptable_price,
            strike_source=strike_source,
            strike_proxy=strike_proxy,
            underlying_price=underlying_price,
            distance_to_strike_pct=distance_to_strike_pct,
            direction_error_detail=direction_error_detail[:512] if direction_error_detail else None,

            final_action=final_action,
            skip_reason=skip_reason[:256] if skip_reason else None,
        )
        db.add(row)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("funnel_log_write_failed", asset=asset, error=str(exc))
