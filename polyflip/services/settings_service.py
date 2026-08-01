"""
polyflip/services/settings_service.py

Единственная точка async-чтения runtime-настроек.
Дефолты берутся из settings_registry.registry_defaults() — не из constants.py.
"""

from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.db.models import RuntimeSettings
from polyflip.settings_registry import registry_defaults

logger = structlog.get_logger(__name__)

# Кэш дефолтов из реестра
_DEFAULTS: dict[str, str] = registry_defaults()


SETTINGS_VALIDATORS = {
    "FLIP_THRESHOLD": lambda v: 0.0
    < (float(v) / 100.0 if float(v) > 1.0 else float(v))
    < 1.0,
    "NO_FLIP_THRESHOLD": lambda v: 0.0
    < (float(v) / 100.0 if float(v) > 1.0 else float(v))
    < 1.0,
    "TRADE_FLIP_THRESHOLD": lambda v: 0.0
    < (float(v) / 100.0 if float(v) > 1.0 else float(v))
    < 1.0,
    "TRADE_NO_FLIP_THRESHOLD": lambda v: 0.0
    < (float(v) / 100.0 if float(v) > 1.0 else float(v))
    < 1.0,
    "MIN_EDGE": lambda v: 0.0
    <= (float(v) / 100.0 if float(v) > 1.0 else float(v))
    < 1.0,
    "MAX_SPREAD_PCT": lambda v: 0.0
    < (float(v) / 100.0 if float(v) > 1.0 else float(v))
    < 1.0,
}


def validate_setting(key: str, value: str) -> None:
    if key in SETTINGS_VALIDATORS:
        try:
            val = float(value)
            if not SETTINGS_VALIDATORS[key](value):
                raise ValueError(
                    f"Значение {value!r} для {key} вне допустимого диапазона"
                )
        except ValueError as e:
            if "вне допустимого диапазона" in str(e):
                raise e
            raise ValueError(f"Значение {value!r} для {key} вне допустимого диапазона")


async def save_setting(db: AsyncSession, key: str, value: str) -> None:
    validate_setting(key, value)
    try:
        f_val = float(value)
        if f_val > 1.0 and ("THRESHOLD" in key or "EDGE" in key or "SPREAD" in key):
            value = str(round(f_val / 100.0, 4))
    except ValueError:
        pass
    existing = (
        await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
    ).scalar_one_or_none()
    if existing:
        existing.value = value
    else:
        db.add(RuntimeSettings(key=key, value=value))
    await db.commit()


async def get_setting(db: AsyncSession, key: str) -> str:
    """Читает настройку из БД. Если нет — возвращает дефолт из реестра."""
    row = (
        await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
    ).scalar_one_or_none()

    if row is not None:
        return row.value

    # Обновляем кэш дефолтов на случай динамически добавленных SettingDef
    defaults = registry_defaults()
    if key in defaults:
        logger.warning("setting_using_default", key=key, default=defaults[key])
        return defaults[key]

    raise KeyError(
        f"Настройка '{key}' не найдена в БД и отсутствует в settings_registry.REGISTRY. "
        f"Добавь SettingDef в REGISTRY."
    )


async def get_float(db: AsyncSession, key: str) -> float:
    val_str = await get_setting(db, key)
    val = float(val_str)
    if val > 1.0 and ("THRESHOLD" in key or "EDGE" in key or "SPREAD" in key):
        logger.warning(
            "normalizing_percent_setting", key=key, raw=val, normalized=val / 100.0
        )
        val = val / 100.0
    return val


async def get_int(db: AsyncSession, key: str) -> int:
    return int(await get_setting(db, key))


async def get_bool(db: AsyncSession, key: str) -> bool:
    val = await get_setting(db, key)
    return val.lower() in ("true", "1", "yes")


async def get_all_settings(db: AsyncSession) -> dict[str, str]:
    """Все настройки: дефолты из реестра + переопределения из БД."""
    rows = (await db.execute(select(RuntimeSettings))).scalars().all()
    result = registry_defaults()
    result.update({r.key: r.value for r in rows})  # БД побеждает
    return result
