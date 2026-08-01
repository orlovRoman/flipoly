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
    used_model: Optional[str],
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
            final_action=final_action,
            skip_reason=skip_reason[:256] if skip_reason else None,
        )
        db.add(row)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("funnel_log_write_failed", asset=asset, error=str(exc))
