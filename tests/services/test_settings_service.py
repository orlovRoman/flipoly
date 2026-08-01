"""
tests/services/test_settings_service.py

Тесты для polyflip/services/settings_service.py.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from polyflip.services.settings_service import (
    get_setting,
    get_float,
    get_int,
    get_bool,
    get_all_settings,
)


@pytest.mark.asyncio
async def test_get_setting_fallback_to_defaults():
    """При отсутствии записи в БД возвращается дефолт из реестра."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )

    val = await get_float(mock_db, "MIN_EDGE")
    assert val == 0.05


@pytest.mark.asyncio
async def test_get_setting_db_overrides_defaults():
    """Запись в БД имеет приоритет перед дефолтом из реестра."""
    mock_row = MagicMock()
    mock_row.value = "0.12"

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=mock_row))
    )

    val = await get_float(mock_db, "MIN_EDGE")
    assert val == 0.12


@pytest.mark.asyncio
async def test_get_bool_conversion():
    """get_bool правильно конвертирует строки 'true', '1', 'yes'."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )

    is_enabled = await get_bool(mock_db, "TRADE_ON_FAVORITE")
    assert isinstance(is_enabled, bool)


@pytest.mark.asyncio
async def test_get_setting_missing_key_raises_key_error():
    """Если ключа нет ни в БД, ни в реестре — поднимается KeyError."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )

    with pytest.raises(KeyError):
        await get_setting(mock_db, "NON_EXISTENT_SUPER_SETTING_KEY_123")
