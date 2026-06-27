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
        "TRADE_MIN_TIME_LEFT_SEC": str(getattr(settings, 'TRADE_MIN_TIME_LEFT_SEC', 10)),
        "TRADE_MAX_TIME_LEFT_SEC": str(getattr(settings, 'TRADE_MAX_TIME_LEFT_SEC', 360)),
        "TRADE_BET_SIZE_USDC": str(settings.TRADE_BET_SIZE_USDC),
        "TRADE_NO_FLIP_THRESHOLD": str(settings.TRADE_NO_FLIP_THRESHOLD),
        "TRADE_FLIP_THRESHOLD": str(settings.TRADE_FLIP_THRESHOLD),
        "TRADING_ENABLED": "true" if settings.TRADING_ENABLED else "false",
        "INITIAL_CAPITAL": str(getattr(settings, 'INITIAL_CAPITAL', 100.0)),
        "TRADE_ONLY_FAVORITE": "true" if getattr(settings, 'TRADE_ONLY_FAVORITE', False) else "false",
        "TRADE_MIN_PRICE": str(getattr(settings, 'TRADE_MIN_PRICE', 0.05)),
        "TRADE_MAX_PRICE": str(getattr(settings, 'TRADE_MAX_PRICE', 0.95)),
        "TRADE_ASSETS": str(getattr(settings, 'TRADE_ASSETS', 'BTC,ETH')),
        "KELLY_ENABLED": "true" if getattr(settings, 'KELLY_ENABLED', True) else "false"
    }

    async with async_session() as session:
        result = await session.execute(select(RuntimeSettings))
        db_settings = result.scalars().all()
        
        for s in db_settings:
            if s.key in current_settings:
                current_settings[s.key] = s.value

    return current_settings

@router.get("/recommended_thresholds")
async def get_recommended_thresholds():
    """
    Возвращает рекомендованные пороги на основе текущего flip_threshold.
    Рекомендованный no_flip = flip_threshold - 0.15.
    """
    async with async_session() as session:
        result = await session.execute(
            select(RuntimeSettings).where(
                RuntimeSettings.key.in_([
                    "TRADE_FLIP_THRESHOLD",
                    "TRADE_NO_FLIP_THRESHOLD",
                    *[f"TRADE_FLIP_THRESHOLD_{a.upper()}" for a in settings.asset_list]
                ])
            )
        )
        db = {s.key: s.value for s in result.scalars().all()}

    # Глобальный flip_threshold
    global_flip = float(db.get("TRADE_FLIP_THRESHOLD", settings.TRADE_FLIP_THRESHOLD))
    recommended_no_flip = round(global_flip - 0.15, 4)

    # Per-asset
    per_asset = {}
    for asset in settings.asset_list:
        key = f"TRADE_FLIP_THRESHOLD_{asset.upper()}"
        if key in db:
            asset_flip = float(db[key])
            per_asset[asset] = {
                "flip_threshold": asset_flip,
                "recommended_no_flip": round(asset_flip - 0.15, 4),
                "is_auto_calibrated": True  # выставлен trainer'ом
            }

    return {
        "global": {
            "flip_threshold": global_flip,
            "current_no_flip": float(db.get("TRADE_NO_FLIP_THRESHOLD", settings.TRADE_NO_FLIP_THRESHOLD)),
            "recommended_no_flip": recommended_no_flip,
            "dead_zone_pp": round((global_flip - recommended_no_flip) * 100, 1)
        },
        "per_asset": per_asset
    }

@router.api_route("/{key}", methods=["PUT", "POST"])
async def update_setting(key: str, payload: SettingValue):
    """
    Обновляет или создает настройку в БД.
    """
    valid_keys = [
        "ACTIVE_FEATURES", 
        "TRADE_EXECUTION_TIME_SEC", 
        "TRADE_MIN_TIME_LEFT_SEC",
        "TRADE_MAX_TIME_LEFT_SEC",
        "TRADE_BET_SIZE_USDC", 
        "TRADE_NO_FLIP_THRESHOLD", 
        "TRADE_FLIP_THRESHOLD", 
        "TRADING_ENABLED",
        "INITIAL_CAPITAL",
        "TRADE_ONLY_FAVORITE",
        "TRADE_MIN_PRICE",
        "TRADE_MAX_PRICE",
        "TRADE_ASSETS",
        "KELLY_ENABLED"
    ]
    
    if key not in valid_keys:
        raise HTTPException(status_code=400, detail="Invalid setting key")

    # Валидация и нормализация порогов вероятности флипа
    if key in ["TRADE_NO_FLIP_THRESHOLD", "TRADE_FLIP_THRESHOLD"]:
        try:
            val = float(payload.value)
            if val < 0.0 or val > 100.0:
                raise HTTPException(status_code=400, detail=f"Value for {key} must be between 0 and 100 (or 0.0 and 1.0)")
            # Если прислали проценты (больше 1.0), автоматически переводим в доли для хранения в БД
            if val > 1.0:
                payload.value = str(val / 100.0)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Value for {key} must be a number")

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
