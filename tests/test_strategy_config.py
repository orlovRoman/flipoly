import pytest
from datetime import datetime, timezone
from sqlalchemy import select
from fastapi import HTTPException

from polyflip.api.settings import update_setting, SettingValue
from polyflip.db.models import RuntimeSettings, StrategyConfig

# Патчим async_session в settings.py для использования тестовой db_session
class DummyAsyncContextManager:
    def __init__(self, session):
        self.session = session
    async def __aenter__(self):
        return self.session
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

def patch_session(db_session):
    return lambda: DummyAsyncContextManager(db_session)

@pytest.mark.asyncio
async def test_config_history_recorded(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # First set (None -> 0.03)
        await update_setting("MIN_EDGE", SettingValue(value="0.03"))
        
        # Second set (0.03 -> 0.05)
        await update_setting("MIN_EDGE", SettingValue(value="0.05"))
        
        # Check StrategyConfig rows
        result = await db_session.execute(
            select(StrategyConfig).where(StrategyConfig.key == "MIN_EDGE")
            .order_by(StrategyConfig.changed_at.asc())
        )
        rows = result.scalars().all()
            
        assert len(rows) == 2
        
        # Row 1: None -> 0.03
        assert rows[0].old_value is None
        assert rows[0].new_value == "0.03"
        assert rows[0].changed_by == "user"
        
        # Row 2: 0.03 -> 0.05
        assert rows[1].old_value == "0.03"
        assert rows[1].new_value == "0.05"
        assert rows[1].changed_by == "user"
            
    finally:
        settings_module.async_session = original_session


@pytest.mark.asyncio
async def test_config_history_with_ip(db_session):
    from unittest.mock import MagicMock
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.1"
        await update_setting("MIN_EDGE", SettingValue(value="0.03"), request=mock_request)
        
        result = await db_session.execute(select(StrategyConfig))
        row = result.scalar_one()
        assert row.source_ip == "192.168.1.1"
        
    finally:
        settings_module.async_session = original_session

