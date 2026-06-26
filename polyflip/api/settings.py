from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from datetime import datetime, timezone
import structlog

from polyflip.db.connection import async_session
from polyflip.db.models import RuntimeSettings
from polyflip.api.auth import verify_api_key
from polyflip.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/settings", tags=["Settings"], dependencies=[Depends(verify_api_key)])

class SettingValue(BaseModel):
    value: str

@router.get("")
async def get_all_settings():
    """
    Возвращает текущие настройки (сначала из БД, если нет - из конфига).
    """
    # Дефолтные значения из config.py
    current_settings = {
        "ACTIVE_FEATURES": settings.ACTIVE_FEATURES,
        "TRADE_EXECUTION_TIME_SEC": str(settings.TRADE_EXECUTION_TIME_SEC),
        "TRADE_BET_SIZE_USDC": str(settings.TRADE_BET_SIZE_USDC),
        "TRADE_NO_FLIP_THRESHOLD": str(settings.TRADE_NO_FLIP_THRESHOLD),
        "TRADE_FLIP_THRESHOLD": str(settings.TRADE_FLIP_THRESHOLD),
        "TRADING_ENABLED": "true" if settings.TRADING_ENABLED else "false",
        "INITIAL_CAPITAL": str(getattr(settings, 'INITIAL_CAPITAL', 100.0)),
        "TRADE_ONLY_FAVORITE": "true" if getattr(settings, 'TRADE_ONLY_FAVORITE', False) else "false",
        "TRADE_MIN_PRICE": str(getattr(settings, 'TRADE_MIN_PRICE', 0.05)),
        "TRADE_MAX_PRICE": str(getattr(settings, 'TRADE_MAX_PRICE', 0.95))
    }

    async with async_session() as session:
        result = await session.execute(select(RuntimeSettings))
        db_settings = result.scalars().all()
        
        for s in db_settings:
            if s.key in current_settings:
                current_settings[s.key] = s.value

    return current_settings

@router.put("/{key}")
async def update_setting(key: str, payload: SettingValue):
    """
    Обновляет или создает настройку в БД.
    """
    valid_keys = [
        "ACTIVE_FEATURES", 
        "TRADE_EXECUTION_TIME_SEC", 
        "TRADE_BET_SIZE_USDC", 
        "TRADE_NO_FLIP_THRESHOLD", 
        "TRADE_FLIP_THRESHOLD", 
        "TRADING_ENABLED",
        "INITIAL_CAPITAL",
        "TRADE_ONLY_FAVORITE",
        "TRADE_MIN_PRICE",
        "TRADE_MAX_PRICE"
    ]
    
    if key not in valid_keys:
        raise HTTPException(status_code=400, detail="Invalid setting key")

    async with async_session() as session:
        result = await session.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
        setting = result.scalar_one_or_none()
        
        now = datetime.now(timezone.utc)
        
        if setting:
            setting.value = payload.value
            setting.updated_at = now
        else:
            setting = RuntimeSettings(
                key=key,
                value=payload.value,
                updated_at=now,
                updated_by="dashboard"
            )
            session.add(setting)
            
        await session.commit()
        logger.info("setting_updated", key=key, value=payload.value)
        
    return {"status": "ok", "key": key, "value": payload.value}
