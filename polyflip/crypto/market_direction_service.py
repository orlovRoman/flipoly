"""
polyflip/crypto/market_direction_service.py

Persist one immutable LightGBM direction signal for the lifetime of a 15-minute market.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.predictor import CryptoPredictor, CryptoSignal
from polyflip.db.models import CryptoCandle, MarketDirectionSignal

logger = structlog.get_logger(__name__)


def _to_crypto_signal(row: MarketDirectionSignal) -> CryptoSignal:
    return CryptoSignal(
        symbol=row.symbol,
        model_key=row.model_key,
        direction=row.direction,
        p_up=row.p_up,
        p_down=row.p_down,
        signal_strength=row.signal_strength,
        strike=row.strike,
        threshold_up=row.threshold_up,
        threshold_down=row.threshold_down,
        model_version=row.model_version,
        features_ok=row.features_ok,
        risk_vetoed=row.risk_vetoed,
        risk_reason=row.risk_reason or "",
        stake_multiplier=row.stake_multiplier,
        funding_rate=row.funding_rate,
        ece=row.ece,
        regime=row.regime,
        status=row.status,
        inverted=row.inverted,
        p_up_raw=row.p_up_raw,
        p_down_raw=row.p_down_raw,
    )


async def get_or_create_market_direction_signal(
    db: AsyncSession,
    market: Any,
    candles: Sequence[CryptoCandle],
    predictor: CryptoPredictor,
    *,
    funding_rate: float | None = None,
    invert_lgbm_signal: bool = False,
) -> CryptoSignal:
    market_id = str(market.market_id)
    configured_symbol = getattr(market, "binance_symbol", None)
    symbol = (
        configured_symbol
        if isinstance(configured_symbol, str) and configured_symbol
        else f"{market.asset}USDT"
    )
    stmt = select(MarketDirectionSignal).where(
        MarketDirectionSignal.market_id == market_id
    )

    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        logger.info(
            "market_direction_signal_reused",
            market_id=market_id,
            asset=market.asset,
            model_key=existing.model_key,
            model_version=existing.model_version,
        )
        return _to_crypto_signal(existing)

    underlying_price = (
        float(market.underlying_price)
        if getattr(market, "underlying_price", None) is not None
        else 0.0
    )
    signal = predictor.predict(
        candles,
        symbol,
        funding_rate=funding_rate,
        invert_lgbm_signal=invert_lgbm_signal,
        underlying_price=underlying_price,
    )
    row = MarketDirectionSignal(
        market_id=market_id,
        asset=market.asset,
        symbol=symbol,
        regime=signal.regime,
        direction=signal.direction,
        p_up=signal.p_up,
        p_down=signal.p_down,
        signal_strength=signal.signal_strength,
        strike=signal.strike,
        threshold_up=signal.threshold_up,
        threshold_down=signal.threshold_down,
        model_key=signal.model_key,
        model_version=signal.model_version,
        features_ok=signal.features_ok,
        risk_vetoed=signal.risk_vetoed,
        risk_reason=signal.risk_reason,
        stake_multiplier=signal.stake_multiplier,
        funding_rate=signal.funding_rate,
        ece=signal.ece,
        status=signal.status,
        inverted=signal.inverted,
        p_up_raw=signal.p_up_raw,
        p_down_raw=signal.p_down_raw,
        created_at=datetime.now(timezone.utc),
    )

    try:
        db.add(row)
        await db.commit()
    except IntegrityError:
        # A concurrent worker won the unique(market_id) race. Its committed row
        # is the canonical signal; never return our independently computed value.
        await db.rollback()
        winner = (await db.execute(stmt)).scalar_one_or_none()
        if winner is None:
            raise
        logger.info(
            "market_direction_signal_race_reused",
            market_id=market_id,
            model_key=winner.model_key,
            model_version=winner.model_version,
        )
        return _to_crypto_signal(winner)
    except Exception:
        await db.rollback()
        logger.exception("market_direction_signal_save_failed", market_id=market_id)
        raise

    logger.info(
        "market_direction_signal_created",
        market_id=market_id,
        asset=market.asset,
        model_key=signal.model_key,
        model_version=signal.model_version,
    )
    return signal
