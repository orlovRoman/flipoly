"""
polyflip/db/init_runtime_settings.py

Инициализация runtime-настроек при старте.
DEFAULTS берётся из settings_registry.py — единственного источника истины.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings
from polyflip.settings_registry import registry_defaults

# Единый источник истины — реестр. Ручной словарь убран.
DEFAULTS = registry_defaults()

from datetime import datetime, timezone


async def migrate_auto_dead_zone_width(session: AsyncSession):
    """
    Разовая миграция: если в БД есть AUTO_DEAD_ZONE_WIDTH — переносим значение
    в DEAD_ZONE_WIDTH и удаляем старый ключ.
    Запускается при каждом старте, но NOOP если AUTO_DEAD_ZONE_WIDTH уже нет.
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


async def migrate_stop_loss_pct(session: AsyncSession):
    """
    Разовая миграция: если в БД есть STOP_LOSS_PCT (старый единый ключ) —
    копируем значение в STOP_LOSS_PCT_FAVORITE и STOP_LOSS_PCT_OUTSIDER,
    затем удаляем старый ключ.
    """
    old = await session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "STOP_LOSS_PCT")
    )
    if not old:
        return

    now = datetime.now(timezone.utc)
    for new_key in ("STOP_LOSS_PCT_FAVORITE", "STOP_LOSS_PCT_OUTSIDER"):
        existing = await session.scalar(
            select(RuntimeSettings).where(RuntimeSettings.key == new_key)
        )
        if not existing:
            session.add(RuntimeSettings(
                key=new_key,
                value=old.value,
                updated_by="migration_stop_loss_pct",
                updated_at=now,
            ))

    await session.delete(old)
    await session.commit()


async def migrate_crypto_to_lightgbm(session: AsyncSession):
    """
    Разовая миграция: если в БД есть TRADING_MODE = "CRYPTO" или TRADING_MODE_<ASSET> = "CRYPTO",
    меняем это значение на "lightgbm".
    """
    result = await session.execute(
        select(RuntimeSettings).where(
            (RuntimeSettings.key == "TRADING_MODE") |
            RuntimeSettings.key.like("TRADING_MODE_%")
        )
    )
    rows = result.scalars().all()
    updated = False
    for r in rows:
        if r.value == "CRYPTO":
            r.value = "lightgbm"
            r.updated_by = "migration_crypto_to_lightgbm"
            r.updated_at = datetime.now(timezone.utc)
            updated = True
    if updated:
        await session.commit()


async def seed_runtime_settings(session: AsyncSession):
    """Заполняет отсутствующие ключи дефолтами при старте."""
    for key, value in DEFAULTS.items():
        exists = await session.scalar(
            select(RuntimeSettings).where(RuntimeSettings.key == key)
        )
        if not exists:
            session.add(RuntimeSettings(
                key=key,
                value=value,
                updated_by="system",
                updated_at=datetime.now(timezone.utc),
            ))
    await session.commit()
