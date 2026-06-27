import pytest
from datetime import datetime, timezone
from fastapi import HTTPException
from polyflip.api.settings import get_all_settings, update_setting, SettingValue
from polyflip.db.models import RuntimeSettings
from sqlalchemy import select

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
async def test_get_all_settings_defaults(db_session):
    from polyflip.api.settings import async_session
    import polyflip.api.settings as settings_module
    
    # Патчим сессию в модуле
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        res = await get_all_settings()
        assert res["KELLY_MAX_FRACTION"] == "0.1"
        assert res["DAILY_LOSS_LIMIT_USDC"] == "-100.0"
        assert res["DEAD_ZONE_WIDTH"] == "0.15"
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_get_all_settings_from_db(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        now = datetime.now(timezone.utc)
        db_session.add(RuntimeSettings(key='DAILY_LOSS_LIMIT_USDC', value='-50.0', updated_at=now, updated_by='test'))
        db_session.add(RuntimeSettings(key='KELLY_MAX_FRACTION', value='0.25', updated_at=now, updated_by='test'))
        await db_session.commit()
        
        res = await get_all_settings()
        assert res["DAILY_LOSS_LIMIT_USDC"] == "-50.0"
        assert res["KELLY_MAX_FRACTION"] == "0.25"
    finally:
        settings_module.async_session = original_session


@pytest.mark.asyncio
async def test_update_setting_valid_kelly(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Проверяем сохранение процентов (15% -> должно преобразоваться в 0.15)
        await update_setting("KELLY_MAX_FRACTION", SettingValue(value="15"))
        
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "KELLY_MAX_FRACTION")
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.15
        
        # Проверяем сохранение доли (0.05 -> должно остаться 0.05)
        await update_setting("KELLY_MAX_FRACTION", SettingValue(value="0.05"))
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.05
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_update_setting_invalid_kelly(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("KELLY_MAX_FRACTION", SettingValue(value="150"))
        assert exc_info.value.status_code == 400
        
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("KELLY_MAX_FRACTION", SettingValue(value="-5"))
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_update_setting_valid_daily_limit(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        await update_setting("DAILY_LOSS_LIMIT_USDC", SettingValue(value="-250"))
        
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "DAILY_LOSS_LIMIT_USDC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == -250.0
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_update_setting_invalid_daily_limit(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # 0.0 не разрешено
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("DAILY_LOSS_LIMIT_USDC", SettingValue(value="0"))
        assert exc_info.value.status_code == 400
        
        # Положительные значения не разрешены
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("DAILY_LOSS_LIMIT_USDC", SettingValue(value="50"))
        assert exc_info.value.status_code == 400
        
        # > $100k не разрешено
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("DAILY_LOSS_LIMIT_USDC", SettingValue(value="-150000"))
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session
