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
        # no-op to satisfy SonarQube rule
        pass

def patch_session(db_session):
    return lambda: DummyAsyncContextManager(db_session)

@pytest.mark.asyncio
async def test_update_per_asset_trading_mode(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    original_assets = settings_module.settings.ASSETS
    settings_module.settings.ASSETS = "BTC,ETH,SOL,XRP,DOGE"
    
    try:
        # Valid values
        await update_setting("TRADING_MODE_BTC", SettingValue(value="ml"), db=db_session)
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADING_MODE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == "ml"
        
        await update_setting("TRADING_MODE_DOGE", SettingValue(value="favorite"), db=db_session)
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADING_MODE_DOGE")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == "favorite"

        await update_setting("TRADING_MODE_BTC", SettingValue(value=""), db=db_session)
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADING_MODE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == ""

        # Invalid value
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("TRADING_MODE_BTC", SettingValue(value="invalid_mode"), db=db_session)
        assert exc_info.value.status_code == 400

        # Invalid asset
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("TRADING_MODE_INVALIDASSET", SettingValue(value="ml"), db=db_session)
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session
        settings_module.settings.ASSETS = original_assets

@pytest.mark.asyncio
async def test_update_per_asset_min_edge(db_session):
    import polyflip.api.settings as settings_module
    original_session = settings_module.async_session
    settings_module.async_session = patch_session(db_session)
    
    try:
        # Valid values
        await update_setting("MIN_EDGE_BTC", SettingValue(value="0.05"), db=db_session)
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "MIN_EDGE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.05
        
        await update_setting("MIN_EDGE_BTC", SettingValue(value=""), db=db_session)
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "MIN_EDGE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == ""

        # Invalid value
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("MIN_EDGE_BTC", SettingValue(value="105"), db=db_session)
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            await update_setting("MIN_EDGE_BTC", SettingValue(value="-1"), db=db_session)
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
        await update_setting("TRADE_MAX_PRICE_BTC", SettingValue(value="0.85"), db=db_session)
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADE_MAX_PRICE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert float(row.value) == 0.85
        
        await update_setting("TRADE_MAX_PRICE_BTC", SettingValue(value=""), db=db_session)
        stmt = select(RuntimeSettings).where(RuntimeSettings.key == "TRADE_MAX_PRICE_BTC")
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.value == ""

        # Invalid value
        with pytest.raises(HTTPException) as exc_info:
            await update_setting("TRADE_MAX_PRICE_BTC", SettingValue(value="1.20"), db=db_session)
        assert exc_info.value.status_code == 400
    finally:
        settings_module.async_session = original_session


@pytest.mark.asyncio
async def test_per_asset_empty_trading_mode_uses_global_not_empty_string(db_session):
    """
    Пустая строка per-asset TRADING_MODE_BTCUSDT не должна
    использоваться как режим — должен браться глобальный TRADING_MODE.
    """
    from polyflip.db.models import RuntimeSettings
    from polyflip.api.settings import get_all_settings
    import polyflip.api.settings as settings_module
    from datetime import datetime, timezone

    class DummyAsyncContextManager:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self.session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # no-op to satisfy SonarQube rule
            pass
    original_session = settings_module.async_session
    settings_module.async_session = lambda: DummyAsyncContextManager(db_session)

    try:
        now = datetime.now(timezone.utc)
        # Пустой per-asset + глобальный CRYPTO
        db_session.add(RuntimeSettings(
            key="TRADING_MODE_BTCUSDT", value="",
            updated_by="test", updated_at=now
        ))
        db_session.add(RuntimeSettings(
            key="TRADING_MODE", value="CRYPTO",
            updated_by="test", updated_at=now
        ))
        await db_session.commit()

        # Engine's logic:
        settings_db = await get_all_settings()
        mode = settings_db.get("TRADING_MODE_BTCUSDT") or settings_db.get("TRADING_MODE", "ML")
        
        # Пустая строка → fallback на global
        assert mode == "CRYPTO", f"Ожидали CRYPTO (global fallback), получили: {mode!r}"
        assert mode != "", "Пустая строка не должна быть режимом торговли"
    finally:
        settings_module.async_session = original_session
