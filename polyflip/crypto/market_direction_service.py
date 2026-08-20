"""
polyflip/crypto/market_direction_service.py

Persist one immutable LightGBM direction signal for the lifetime of a 15-minute market.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.predictor import CryptoPredictor, CryptoSignal
from polyflip.db.models import CryptoCandle, MarketDirectionSignal
from polyflip.constants import resolve_binance_symbol

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
            raw_opinion=(
                "UP" if (row.p_up_raw if row.p_up_raw is not None else row.p_up) >= 0.5 else "DOWN"
            ),
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
    # LiveMarket.asset is not consistent across collector and trading paths:
    # older rows contain BTC while newer rows may already contain BTCUSDT.
    # Appending USDT blindly turns the latter into BTCUSDTUSDT and makes every
    # loaded model look unavailable. Resolve both forms canonically.
    symbol = resolve_binance_symbol(configured_symbol)
    if symbol is None:
        symbol = resolve_binance_symbol(getattr(market, "asset", None))
    if symbol is None:
        raw_asset = str(getattr(market, "asset", "")).strip().upper().split("_", 1)[0]
        symbol = raw_asset if raw_asset.endswith("USDT") else f"{raw_asset}USDT"
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
    # The frozen signal must use the same information boundary as training:
    # only candles closed by the market opening time.  Keep the optional kwarg
    # out for legacy/mock market objects that do not expose a real timestamp.
    prediction_kwargs = {}
    market_end = getattr(market, "end_time_est", None)
    if isinstance(market_end, datetime):
        if market_end.tzinfo is None:
            market_end = market_end.replace(tzinfo=timezone.utc)
        interval = "15m"
        get_interval = getattr(predictor, "get_interval", None)
        if callable(get_interval):
            configured_interval = get_interval(symbol)
            if isinstance(configured_interval, str):
                interval = configured_interval
        interval_minutes = {"15m": 15, "1h": 60, "4h": 240}.get(interval, 15)
        prediction_kwargs["decision_time"] = market_end - timedelta(minutes=interval_minutes)

    signal = predictor.predict(
        candles,
        symbol,
        funding_rate=funding_rate,
        invert_lgbm_signal=invert_lgbm_signal,
        underlying_price=underlying_price,
        market_context=market,
        **prediction_kwargs,
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
