from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings
from polyflip.constants import (
    DEAD_ZONE_WIDTH,
    DAILY_LOSS_LIMIT_USDC,
    DEFAULT_TRADING_MODE,
    FAVORITE_MODE_ENTRY_SEC,
    LIVE_POLL_INTERVAL_SECONDS,
    MIN_EDGE,
    FAVORITE_MIN_EDGE,
    TRADE_ON_FLIP,
    FLIP_THRESHOLD,
    OUTSIDER_MAX_PRICE,
    NO_MIN_EDGE,
    AUTO_DEAD_ZONE,
    AUTO_DEAD_ZONE_WIDTH,
    MAX_EDGE_SCALING as MAX_BET_EDGE,
    CRYPTO_MIN_EDGE,
)

DEFAULTS = {
    "DEAD_ZONE_WIDTH": str(DEAD_ZONE_WIDTH),
    "DAILY_LOSS_LIMIT_USDC": str(DAILY_LOSS_LIMIT_USDC),
    "TRADING_MODE": DEFAULT_TRADING_MODE,
    "FAVORITE_MODE_ENTRY_SEC": str(FAVORITE_MODE_ENTRY_SEC),
    "LIVE_POLL_INTERVAL_SECONDS": str(LIVE_POLL_INTERVAL_SECONDS),
    "ENTRY_STRATEGY": "first",  # first, best_edge, confirmed
    "BET_SIZING_MODE": "scaled",
    "MAX_BET_SIZE_USDC": "50.0",
    "FAVORITE_THRESHOLD": "0.55",
    "FAVORITE_MIN_EDGE": str(FAVORITE_MIN_EDGE),
    "LIQUIDITY_FRACTION": "0.05",
    "MIN_EDGE": str(MIN_EDGE),
    "MAX_BET_EDGE": str(MAX_BET_EDGE),
    "TRADE_ON_FLIP": "false",
    "FLIP_THRESHOLD": str(FLIP_THRESHOLD),
    "OUTSIDER_MAX_PRICE": str(OUTSIDER_MAX_PRICE),
    "NO_MIN_EDGE": str(NO_MIN_EDGE),
    "MAX_PRICE_DRIFT": "0.10",
    "AUTO_DEAD_ZONE": "true",
    "AUTO_DEAD_ZONE_WIDTH": str(AUTO_DEAD_ZONE_WIDTH),
    "INITIAL_CAPITAL": "1000.0",
    "TRADE_BET_SIZE_USDC": "5.0",
    "TRADE_EXECUTION_TIME_SEC": "30",
    "TRADE_MIN_TIME_LEFT_SEC": "10",
    "TRADE_MAX_TIME_LEFT_SEC": "360",
    "TRADE_NO_FLIP_THRESHOLD": "0.15",
    "CRYPTO_MIN_EDGE": str(CRYPTO_MIN_EDGE),
    "USE_CRYPTO_CONFIRM": "false",
    "CRYPTO_STANDALONE": "false",
}

from datetime import datetime, timezone

async def seed_runtime_settings(session: AsyncSession):
    """Заполняет отсутствующие ключи дефолтами при старте."""
    for key, value in DEFAULTS.items():
        exists = await session.scalar(select(RuntimeSettings).where(RuntimeSettings.key == key))
        if not exists:
            session.add(RuntimeSettings(
                key=key,
                value=value,
                updated_by="system",
                updated_at=datetime.now(timezone.utc)
            ))
    await session.commit()
