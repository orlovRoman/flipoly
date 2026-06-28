from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from datetime import datetime, timezone
import structlog

from polyflip.db.connection import async_session
from polyflip.db.models import RuntimeSettings
from polyflip.api.auth import verify_api_key
from polyflip.config import settings
from polyflip.constants import KELLY_MAX_FRACTION, DAILY_LOSS_LIMIT_USDC

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/settings", tags=["Settings"], dependencies=[Depends(verify_api_key)])

class SettingValue(BaseModel):
    value: str

@router.get("")
async def get_all_settings():
    """
    Возвращает текущие настройки (сначала из БД, если нет - из конфига/констант).
    """
    async with async_session() as session:
        result = await session.execute(select(RuntimeSettings))
        db = {s.key: s.value for s in result.scalars().all()}

    return {
        "ACTIVE_FEATURES": db.get("ACTIVE_FEATURES", settings.ACTIVE_FEATURES),
        "TRADE_EXECUTION_TIME_SEC": db.get("TRADE_EXECUTION_TIME_SEC", str(settings.TRADE_EXECUTION_TIME_SEC)),
        "TRADE_MIN_TIME_LEFT_SEC": db.get("TRADE_MIN_TIME_LEFT_SEC", str(getattr(settings, 'TRADE_MIN_TIME_LEFT_SEC', 10))),
        "TRADE_MAX_TIME_LEFT_SEC": db.get("TRADE_MAX_TIME_LEFT_SEC", str(getattr(settings, 'TRADE_MAX_TIME_LEFT_SEC', 360))),
        "TRADE_BET_SIZE_USDC": db.get("TRADE_BET_SIZE_USDC", str(settings.TRADE_BET_SIZE_USDC)),
        "TRADE_NO_FLIP_THRESHOLD": db.get("TRADE_NO_FLIP_THRESHOLD", str(settings.TRADE_NO_FLIP_THRESHOLD)),
        "DEAD_ZONE_WIDTH": db.get("DEAD_ZONE_WIDTH", str(getattr(settings, 'DEAD_ZONE_WIDTH', 0.15))),
        "KELLY_MAX_FRACTION": db.get("KELLY_MAX_FRACTION", str(KELLY_MAX_FRACTION)),
        "DAILY_LOSS_LIMIT_USDC": db.get("DAILY_LOSS_LIMIT_USDC", str(DAILY_LOSS_LIMIT_USDC)),
        "TRADING_ENABLED": db.get("TRADING_ENABLED", "true" if settings.TRADING_ENABLED else "false"),
        "INITIAL_CAPITAL": db.get("INITIAL_CAPITAL", str(getattr(settings, 'INITIAL_CAPITAL', 100.0))),
        "TRADE_MIN_PRICE": db.get("TRADE_MIN_PRICE", str(getattr(settings, 'TRADE_MIN_PRICE', 0.05))),
        "TRADE_MAX_PRICE": db.get("TRADE_MAX_PRICE", str(getattr(settings, 'TRADE_MAX_PRICE', 0.95))),
        "TRADE_ASSETS": db.get("TRADE_ASSETS", str(getattr(settings, 'TRADE_ASSETS', 'BTC,ETH'))),
        "KELLY_ENABLED": db.get("KELLY_ENABLED", "true" if getattr(settings, 'KELLY_ENABLED', True) else "false"),
        "TRADING_MODE": db.get("TRADING_MODE", settings.TRADING_MODE),
        "FAVORITE_MODE_ENTRY_SEC": db.get("FAVORITE_MODE_ENTRY_SEC", str(settings.FAVORITE_MODE_ENTRY_SEC)),
        "LIVE_POLL_INTERVAL_SECONDS": db.get("LIVE_POLL_INTERVAL_SECONDS", str(settings.LIVE_POLL_INTERVAL_SECONDS)),
        "MIN_EDGE": db.get("MIN_EDGE", str(settings.MIN_EDGE))
    }

@router.get("/recommended_thresholds")
async def get_recommended_thresholds():
    """
    Возвращает рекомендованные пороги.
    Рекомендованный no_flip = flip_threshold - DEAD_ZONE_WIDTH.
    """
    async with async_session() as session:
        result = await session.execute(
            select(RuntimeSettings).where(
                RuntimeSettings.key.in_([
                    "DEAD_ZONE_WIDTH",
                    "TRADE_NO_FLIP_THRESHOLD",
                    *[f"TRADE_FLIP_THRESHOLD_{a.upper()}" for a in settings.asset_list]
                ])
            )
        )
        db = {s.key: s.value for s in result.scalars().all()}

    dead_zone = float(db.get("DEAD_ZONE_WIDTH", getattr(settings, 'DEAD_ZONE_WIDTH', 0.15)))

    # Per-asset
    per_asset = {}
    for asset in settings.asset_list:
        key = f"TRADE_FLIP_THRESHOLD_{asset.upper()}"
        if key in db:
            asset_flip = float(db[key])
            per_asset[asset] = {
                "flip_threshold": asset_flip,
                "recommended_no_flip": round(asset_flip - dead_zone, 4),
                "is_auto_calibrated": True  # выставлен trainer'ом
            }

    return {
        "global": {
            "dead_zone": dead_zone,
            "current_no_flip": float(db.get("TRADE_NO_FLIP_THRESHOLD", settings.TRADE_NO_FLIP_THRESHOLD)),
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
        "DEAD_ZONE_WIDTH", 
        "KELLY_MAX_FRACTION",
        "DAILY_LOSS_LIMIT_USDC",
        "TRADING_ENABLED",
        "INITIAL_CAPITAL",
        "TRADE_MIN_PRICE",
        "TRADE_MAX_PRICE",
        "TRADE_ASSETS",
        "KELLY_ENABLED",
        "TRADING_MODE",
        "FAVORITE_MODE_ENTRY_SEC",
        "LIVE_POLL_INTERVAL_SECONDS",
        "MIN_EDGE"
    ]
    
    if key not in valid_keys:
        raise HTTPException(status_code=400, detail="Invalid setting key")

    # Валидация и нормализация порогов вероятности флипа и мертвой зоны
    if key in ["TRADE_NO_FLIP_THRESHOLD", "DEAD_ZONE_WIDTH", "KELLY_MAX_FRACTION"]:
        try:
            val = float(payload.value)
            if val < 0.0 or val > 100.0:
                raise HTTPException(status_code=400, detail=f"Value for {key} must be between 0 and 100")
            # Если прислали проценты (больше 1.0), автоматически переводим в доли для хранения в БД
            if val > 1.0:
                payload.value = str(val / 100.0)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Value for {key} must be a number")

    if key == "MIN_EDGE":
        try:
            val = float(payload.value)
            # Разрешаем либо проценты [0.5, 50.0], либо доли [0.005, 0.50]
            if not ((0.5 <= val <= 50.0) or (0.005 <= val <= 0.50)):
                raise HTTPException(status_code=400, detail="MIN_EDGE must be between 0.5% and 50% (or 0.005 and 0.50)")
            if val > 0.50:
                payload.value = str(val / 100.0)
        except ValueError:
            raise HTTPException(status_code=400, detail="MIN_EDGE must be a number")

    if key == "DAILY_LOSS_LIMIT_USDC":
        try:
            val = float(payload.value)
            if val >= 0.0:
                raise HTTPException(status_code=400, detail="DAILY_LOSS_LIMIT_USDC must be strictly negative (e.g., -100)")
            if val < -100000.0:
                raise HTTPException(status_code=400, detail="Daily loss limit is too large (> $100k)")
        except ValueError:
            raise HTTPException(status_code=400, detail="Value for DAILY_LOSS_LIMIT_USDC must be a number")

    if key == "TRADING_MODE":
        if payload.value not in ("ml", "favorite"):
            raise HTTPException(status_code=400, detail="TRADING_MODE must be 'ml' or 'favorite'")

    if key == "FAVORITE_MODE_ENTRY_SEC":
        try:
            val = int(payload.value)
            if not (30 <= val <= 600):
                raise HTTPException(status_code=400, detail="FAVORITE_MODE_ENTRY_SEC must be between 30 and 600 seconds")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="FAVORITE_MODE_ENTRY_SEC must be an integer")

    if key == "LIVE_POLL_INTERVAL_SECONDS":
        try:
            val = int(payload.value)
            if not (2 <= val <= 300):
                raise HTTPException(status_code=400, detail="LIVE_POLL_INTERVAL_SECONDS must be between 2 and 300 seconds")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="LIVE_POLL_INTERVAL_SECONDS must be an integer")

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
