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
    MAX_EDGE_SCALING,   # потолок масштабирования ставки → MAX_BET_EDGE
    MAX_EDGE_FILTER,    # фильтр аномального edge
    CRYPTO_MIN_EDGE,
)

DEFAULTS = {
    "DEAD_ZONE_WIDTH": str(DEAD_ZONE_WIDTH),       # единственная ширина зоны
    "AUTO_DEAD_ZONE": "true",
    # AUTO_DEAD_ZONE_WIDTH больше не сидируется
    "DAILY_LOSS_LIMIT_USDC": str(DAILY_LOSS_LIMIT_USDC),
    "INITIAL_CAPITAL": "1000.0",
    "TRADING_MODE": DEFAULT_TRADING_MODE,
    "TRADING_ENABLED": "false",
    "FAVORITE_MODE_ENTRY_SEC": str(FAVORITE_MODE_ENTRY_SEC),
    "LIVE_POLL_INTERVAL_SECONDS": str(LIVE_POLL_INTERVAL_SECONDS),
    "TRADE_MIN_TIME_LEFT_SEC": "10",
    "TRADE_MAX_TIME_LEFT_SEC": "360",
    "TRADE_EXECUTION_TIME_SEC": "30",
    "ENTRY_STRATEGY": "first",  # first, best_edge, confirmed
    "BET_SIZING_MODE": "scaled",
    "MAX_BET_SIZE_USDC": "50.0",
    "TRADE_BET_SIZE_USDC": "5.0",
    "FAVORITE_THRESHOLD": "0.55",
    "FAVORITE_MIN_EDGE": str(FAVORITE_MIN_EDGE),
    "FAVORITE_MIN_PRICE": "0.55",
    "FAVORITE_MAX_PRICE": "0.95",
    "LIQUIDITY_FRACTION": "0.05",
    "MIN_EDGE": str(MIN_EDGE),
    "MAX_BET_EDGE": str(MAX_EDGE_SCALING),          # потолок масштабирования (0.40)
    "MAX_EDGE_FILTER": str(MAX_EDGE_FILTER),        # фильтр аномалий (0.20)
    "TRADE_ON_FLIP": "false",
    "FLIP_THRESHOLD": str(FLIP_THRESHOLD),
    "OUTSIDER_MAX_PRICE": str(OUTSIDER_MAX_PRICE),
    "NO_MIN_EDGE": str(NO_MIN_EDGE),
    "MAX_PRICE_DRIFT": "0.10",
    "TRADE_NO_FLIP_THRESHOLD": "0.15",
    "TRADE_MIN_PRICE": "0.05",
    "TRADE_MAX_PRICE": "0.95",
    "TRADE_ASSETS": "BTC,ETH",
    "ACTIVE_FEATURES": "time_left_min,mid_price,spread,volume_5min,price_velocity,hour_of_day",
    "CRYPTO_MIN_EDGE": str(CRYPTO_MIN_EDGE),
    "USE_CRYPTO_CONFIRM": "false",
    "CRYPTO_STANDALONE": "false",
    "BYPASS_BET_SIZE_CHECK": "false",
}

from datetime import datetime, timezone


async def migrate_auto_dead_zone_width(session: AsyncSession):
    """
    Разовая миграция: если в БД есть AUTO_DEAD_ZONE_WIDTH — переносим значение
    в DEAD_ZONE_WIDTH и удаляем старый ключ.
    """
    old = await session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "AUTO_DEAD_ZONE_WIDTH")
    )
    if old:
        target = await session.scalar(
            select(RuntimeSettings).where(RuntimeSettings.key == "DEAD_ZONE_WIDTH")
        )
        if target:
            target.value = old.value
            target.updated_by = "migration_auto_dead_zone"
            target.updated_at = datetime.now(timezone.utc)
        else:
            session.add(RuntimeSettings(
                key="DEAD_ZONE_WIDTH",
                value=old.value,
                updated_by="migration_auto_dead_zone",
                updated_at=datetime.now(timezone.utc),
            ))
        await session.delete(old)
        await session.commit()


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
