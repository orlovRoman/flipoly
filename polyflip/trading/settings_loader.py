from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings
from polyflip.config import settings


async def load_trading_settings(
    db_session: AsyncSession,
    trade_assets: list[str] | None = None,
) -> dict[str, str]:
    """
    Загружает базовые ключи + per-asset пороги из RuntimeSettings.
    Возвращает сырой dict[str, str] — парсинг на стороне вызывающего.
    """
    settings_keys = [
        "TRADING_ENABLED", 
        "TRADE_MIN_TIME_LEFT_SEC",
        "TRADE_MAX_TIME_LEFT_SEC",
        "TRADE_BET_SIZE_USDC",
        "TRADE_NO_FLIP_THRESHOLD",
        "DEAD_ZONE_WIDTH",
        "DAILY_LOSS_LIMIT_USDC",
        "ACTIVE_FEATURES",
        "TRADE_MIN_PRICE",
        "TRADE_MAX_PRICE",
        "TRADE_ASSETS",
        "TRADING_MODE",
        "FAVORITE_MODE_ENTRY_SEC",
        "MIN_EDGE",
        "MAX_BET_EDGE",
        "MAX_EDGE_FILTER",
        "FAVORITE_THRESHOLD",
        "TRADE_ON_FAVORITE",
        "TRADE_ON_FLIP",
        "FLIP_THRESHOLD",
        "NO_MIN_EDGE",
        "AUTO_DEAD_ZONE",
        "MAX_PRICE_DRIFT",
        "BET_SIZING_MODE",
        "MAX_BET_SIZE_USDC",
        "FAVORITE_MIN_PRICE",
        "FAVORITE_MAX_PRICE",
        "FAVORITE_MIN_EDGE",
        "OUTSIDER_MAX_PRICE",
        "LIQUIDITY_FRACTION",
        "BYPASS_BET_SIZE_CHECK",
        "USE_CRYPTO_CONFIRM",
        "CRYPTO_STANDALONE",
        "CRYPTO_MIN_EDGE",
        "STOP_LOSS_ENABLED",
        "STOP_LOSS_PCT_FAVORITE",
        "STOP_LOSS_PCT_OUTSIDER",
        "TAKE_PROFIT_ENABLED",
        "TAKE_PROFIT_MULTIPLIER",
    ]
    
    stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(settings_keys))
    result = await db_session.execute(stmt)
    settings_db = {s.key: str(s.value) for s in result.scalars().all()}
    
    if trade_assets is None:
        trade_assets_str = settings_db.get("TRADE_ASSETS", settings.TRADE_ASSETS)
        trade_assets = [a.strip() for a in trade_assets_str.split(",") if a.strip()]
        
    threshold_keys = []
    for asset in trade_assets:
        asset_upper = asset.upper()
        threshold_keys.append(f"FLIP_THRESHOLD_{asset_upper}")
        threshold_keys.append(f"TRADE_FLIP_THRESHOLD_{asset_upper}")
        threshold_keys.append(f"TRADING_MODE_{asset_upper}")
        threshold_keys.append(f"MIN_EDGE_{asset_upper}")
        threshold_keys.append(f"TRADE_MAX_PRICE_{asset_upper}")
        threshold_keys.append(f"AUTO_FLIP_THRESHOLD_{asset_upper}")
        threshold_keys.append(f"AUTO_FLIP_THRESHOLD_{asset_upper}_contested")
        threshold_keys.append(f"AUTO_FLIP_THRESHOLD_{asset_upper}_leaning")
        threshold_keys.append(f"AUTO_FLIP_THRESHOLD_{asset_upper}_decided")
        
    if threshold_keys:
        t_stmt = select(RuntimeSettings).where(RuntimeSettings.key.in_(threshold_keys))
        t_result = await db_session.execute(t_stmt)
        for s in t_result.scalars().all():
            settings_db[s.key] = str(s.value)
            
    return settings_db
