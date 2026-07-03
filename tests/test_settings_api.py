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
        await db_session.commit()
        
        res = await get_all_settings()
        assert res["DAILY_LOSS_LIMIT_USDC"] == "-50.0"
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


@pytest.mark.asyncio
async def test_update_setting_valid_poll_interval(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        await update_setting("LIVE_POLL_INTERVAL_SECONDS", SettingValue(value="15"))
        
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_POLL_INTERVAL_SECONDS")
        row = (await db_session.execute(stmt)).scalar_one()
        assert int(row.value) == 15
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_update_setting_invalid_poll_interval(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Слишком мало (меньше 2)
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("LIVE_POLL_INTERVAL_SECONDS", SettingValue(value="1"))
        assert exc_info.value.status_code == 400
        
        # Слишком много (больше 300)
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("LIVE_POLL_INTERVAL_SECONDS", SettingValue(value="301"))
        assert exc_info.value.status_code == 400
        
        # Не число
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("LIVE_POLL_INTERVAL_SECONDS", SettingValue(value="abc"))
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session


@pytest.mark.asyncio
async def test_update_setting_valid_min_edge(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Проверяем сохранение процентов (8% -> 0.08)
        await update_setting("MIN_EDGE", SettingValue(value="8"))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "MIN_EDGE")
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.08
        
        # Проверяем сохранение доли (0.04 -> 0.04)
        await update_setting("MIN_EDGE", SettingValue(value="0.04"))
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.04
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_update_setting_invalid_min_edge(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Слишком много (больше 100)
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("MIN_EDGE", SettingValue(value="105"))
        assert exc_info.value.status_code == 400
        
        # Слишком мало (меньше 0)
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("MIN_EDGE", SettingValue(value="-1"))
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session


@pytest.mark.asyncio
async def test_update_settings_bulk(db_session):
    from polyflip.api.settings import update_settings_bulk, BulkSettings
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        payload = BulkSettings(settings={
            "TRADE_BET_SIZE_USDC": "25.0",
            "INITIAL_CAPITAL": "1500.0",
            "TRADING_ENABLED": "true"
        })
        res = await update_settings_bulk(payload)
        assert res["status"] == "ok"
        
        # Verify they are written in DB
        rows = (await db_session.execute(select(RuntimeSettings))).scalars().all()
        db_settings = {r.key: r.value for r in rows}
        assert db_settings["TRADE_BET_SIZE_USDC"] == "25.0"
        assert db_settings["INITIAL_CAPITAL"] == "1500.0"
        assert db_settings["TRADING_ENABLED"] == "true"
    finally:
        settings_module.async_session = original_session


@pytest.mark.asyncio
async def test_update_settings_bulk_partial_error(db_session):
    from polyflip.api.settings import update_settings_bulk, BulkSettings
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        payload = BulkSettings(settings={
            "TRADE_BET_SIZE_USDC": "30.0",  # valid
            "FAVORITE_MAX_PRICE": "0.99",          # valid
            "DAILY_LOSS_LIMIT_USDC": "100.0" # invalid (must be strictly negative)
        })
        res = await update_settings_bulk(payload)
        assert res["status"] == "partial"
        assert "TRADE_BET_SIZE_USDC" in res["saved"]
        assert "DAILY_LOSS_LIMIT_USDC" in res["errors"]
        
        # Verify valid key is written in DB
        rows = (await db_session.execute(select(RuntimeSettings))).scalars().all()
        db_settings = {r.key: r.value for r in rows}
        assert db_settings["TRADE_BET_SIZE_USDC"] == "30.0"
        # Verify invalid key is NOT updated in DB
        assert "DAILY_LOSS_LIMIT_USDC" not in db_settings
    finally:
        settings_module.async_session = original_session

