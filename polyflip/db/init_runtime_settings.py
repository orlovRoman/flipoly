from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings
from polyflip.constants import (
    DEAD_ZONE_WIDTH,
    KELLY_MAX_FRACTION,
    DAILY_LOSS_LIMIT_USDC,
    DEFAULT_TRADING_MODE,
    FAVORITE_MODE_ENTRY_SEC,
    LIVE_POLL_INTERVAL_SECONDS
)

DEFAULTS = {
    "DEAD_ZONE_WIDTH": str(DEAD_ZONE_WIDTH),
    "KELLY_MAX_FRACTION": str(KELLY_MAX_FRACTION),
    "DAILY_LOSS_LIMIT_USDC": str(DAILY_LOSS_LIMIT_USDC),
    "TRADING_MODE": DEFAULT_TRADING_MODE,
    "FAVORITE_MODE_ENTRY_SEC": str(FAVORITE_MODE_ENTRY_SEC),
    "LIVE_POLL_INTERVAL_SECONDS": str(LIVE_POLL_INTERVAL_SECONDS),
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
