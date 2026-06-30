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
    TRADE_ON_FLIP,
    FLIP_THRESHOLD,
    NO_MAX_PRICE,
    NO_MIN_EDGE,
    AUTO_DEAD_ZONE,
    AUTO_DEAD_ZONE_WIDTH,
    MAX_EDGE
)

DEFAULTS = {
    "DEAD_ZONE_WIDTH": str(DEAD_ZONE_WIDTH),
    "DAILY_LOSS_LIMIT_USDC": str(DAILY_LOSS_LIMIT_USDC),
    "TRADING_MODE": DEFAULT_TRADING_MODE,
    "FAVORITE_MODE_ENTRY_SEC": str(FAVORITE_MODE_ENTRY_SEC),
    "LIVE_POLL_INTERVAL_SECONDS": str(LIVE_POLL_INTERVAL_SECONDS),
    "MIN_EDGE": str(MIN_EDGE),
    "MAX_EDGE": str(MAX_EDGE),
    "TRADE_ON_FLIP": "false",
    "FLIP_THRESHOLD": str(FLIP_THRESHOLD),
    "NO_MAX_PRICE": str(NO_MAX_PRICE),
    "NO_MIN_EDGE": str(NO_MIN_EDGE),
    "AUTO_DEAD_ZONE": "true" if AUTO_DEAD_ZONE else "false",
    "AUTO_DEAD_ZONE_WIDTH": str(AUTO_DEAD_ZONE_WIDTH),
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
