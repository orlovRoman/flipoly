import pytest
from fastapi import HTTPException
from polyflip.api.settings import update_setting, SettingValue
from polyflip.db.models import RuntimeSettings
from sqlalchemy import select

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
async def test_update_per_asset_trading_mode(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Valid values
        await update_setting("TRADING_MODE_BTC", SettingValue(value="ml"))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADING_MODE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == "ml"
        
        await update_setting("TRADING_MODE_DOGE", SettingValue(value="favorite"))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADING_MODE_DOGE")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == "favorite"

        await update_setting("TRADING_MODE_BTC", SettingValue(value=""))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADING_MODE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == ""

        # Invalid value
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("TRADING_MODE_BTC", SettingValue(value="invalid_mode"))
        assert exc_info.value.status_code == 400

        # Invalid asset
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("TRADING_MODE_INVALIDASSET", SettingValue(value="ml"))
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_update_per_asset_min_edge(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Valid values
        await update_setting("MIN_EDGE_BTC", SettingValue(value="0.05"))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "MIN_EDGE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.05
        
        await update_setting("MIN_EDGE_BTC", SettingValue(value=""))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "MIN_EDGE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == ""

        # Invalid value
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("MIN_EDGE_BTC", SettingValue(value="105"))
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            await update_setting("MIN_EDGE_BTC", SettingValue(value="-1"))
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session

@pytest.mark.asyncio
async def test_update_per_asset_max_price(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Valid values
        await update_setting("TRADE_MAX_PRICE_BTC", SettingValue(value="0.85"))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADE_MAX_PRICE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.85
        
        await update_setting("TRADE_MAX_PRICE_BTC", SettingValue(value=""))
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADE_MAX_PRICE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == ""

        # Invalid value
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("TRADE_MAX_PRICE_BTC", SettingValue(value="1.20"))
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session
