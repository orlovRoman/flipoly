"""
polyflip/crypto/market_direction_service.py

Сервис получения или атомарного создания единого замороженного прогноза LightGBM (CryptoSignal)
на весь 15-минутный рынок.
Гарантирует 1 фиксированный прогноз на 1 market_id без дрейфа в рамках окна.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.predictor import CryptoPredictor, CryptoSignal
from polyflip.db.models import MarketDirectionSignal, CryptoCandle

logger = structlog.get_logger(__name__)


async def get_or_create_market_direction_signal(
    db: AsyncSession,
    market: any,
    candles: Sequence[CryptoCandle],
    predictor: CryptoPredictor,
) -> CryptoSignal:
    """
    1. Ищет сохранённый прогноз в market_direction_signals по market.market_id.
    2. Если уже создан — возвращает воссозданный CryptoSignal.
    3. Если не найден — генерирует новый через predictor.predict(), сохраняет в DB и возвращает.
    """
    market_id_str = str(market.market_id)
    symbol = getattr(market, "binance_symbol", None) or f"{market.asset}USDT"

    # 1. Поиск сохранённого прогноза по market_id
    stmt = select(MarketDirectionSignal).where(MarketDirectionSignal.market_id == market_id_str)
    existing = (await db.execute(stmt)).scalar_one_or_none()

    if existing:
        logger.info(
            "market_direction_signal_reused",
            market_id=market_id_str,
            asset=market.asset,
            direction=existing.direction,
            model_key=existing.model_key,
            model_version=existing.model_version,
        )
        return CryptoSignal(
            symbol=existing.symbol,
            model_key=existing.model_key,
            direction=existing.direction,
            p_up=existing.p_up,
            p_down=existing.p_down,
            signal_strength=existing.signal_strength,
            strike=existing.strike,
            threshold_up=existing.threshold_up,
            threshold_down=existing.threshold_down,
            model_version=existing.model_version,
            features_ok=existing.features_ok,
            risk_vetoed=existing.risk_vetoed,
            risk_reason=existing.risk_reason,
            regime=existing.regime,
            status=existing.status,
        )

    # 2. Вычисление нового прогноза
    underlying_p = float(market.underlying_price) if getattr(market, "underlying_price", None) is not None else None
    sig = predictor.predict(candles, symbol, underlying_price=underlying_p)

    # 3. Сохранение в БД
    try:
        new_row = MarketDirectionSignal(
            market_id=market_id_str,
            asset=market.asset,
            symbol=symbol,
            regime=sig.regime,
            direction=sig.direction,
            p_up=sig.p_up,
            p_down=sig.p_down,
            signal_strength=sig.signal_strength,
            strike=sig.strike,
            threshold_up=sig.threshold_up,
            threshold_down=sig.threshold_down,
            model_key=sig.model_key,
            model_version=sig.model_version,
            features_ok=sig.features_ok,
            risk_vetoed=sig.risk_vetoed,
            risk_reason=sig.risk_reason,
            status=sig.status,
            created_at=datetime.now(timezone.utc),
        )
        db.add(new_row)
        await db.flush()
        logger.info(
            "market_direction_signal_created",
            market_id=market_id_str,
            asset=market.asset,
            direction=sig.direction,
            model_key=sig.model_key,
        )
    except Exception as e:
        logger.warning("market_direction_signal_save_failed", error=str(e), market_id=market_id_str)

    return sig
