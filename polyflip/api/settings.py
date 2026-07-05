from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional, Any, Union
from pydantic import BaseModel
from sqlalchemy import select
from datetime import datetime, timezone
import structlog

from polyflip.db.connection import async_session
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings, StrategyConfig
from polyflip.api.auth import verify_api_key
from polyflip.config import settings
from polyflip.constants import (
    DAILY_LOSS_LIMIT_USDC, TRADE_ON_FLIP, FLIP_THRESHOLD,
    NO_MIN_EDGE, AUTO_DEAD_ZONE,
    FAVORITE_MIN_EDGE, CRYPTO_MIN_EDGE,
    MAX_EDGE_FILTER, MAX_EDGE_SCALING,
)
from polyflip.settings_registry import editable_keys as _registry_editable_keys

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

    settings_dict = {
        "ACTIVE_FEATURES": db.get("ACTIVE_FEATURES", settings.ACTIVE_FEATURES),
        "TRADE_EXECUTION_TIME_SEC": db.get("TRADE_EXECUTION_TIME_SEC", str(settings.TRADE_EXECUTION_TIME_SEC)),
        "TRADE_MIN_TIME_LEFT_SEC": db.get("TRADE_MIN_TIME_LEFT_SEC", str(settings.TRADE_MIN_TIME_LEFT_SEC)),
        "TRADE_MAX_TIME_LEFT_SEC": db.get("TRADE_MAX_TIME_LEFT_SEC", str(settings.TRADE_MAX_TIME_LEFT_SEC)),
        "BET_SIZING_MODE": db.get("BET_SIZING_MODE", settings.BET_SIZING_MODE),
        "MAX_BET_SIZE_USDC": db.get("MAX_BET_SIZE_USDC", str(settings.MAX_BET_SIZE_USDC)),
        "TRADE_BET_SIZE_USDC": db.get("TRADE_BET_SIZE_USDC", str(settings.TRADE_BET_SIZE_USDC)),
        "TRADE_NO_FLIP_THRESHOLD": db.get("TRADE_NO_FLIP_THRESHOLD", str(settings.TRADE_NO_FLIP_THRESHOLD)),
        "DEAD_ZONE_WIDTH": db.get("DEAD_ZONE_WIDTH", str(settings.DEAD_ZONE_WIDTH)),
        "DAILY_LOSS_LIMIT_USDC": db.get("DAILY_LOSS_LIMIT_USDC", str(settings.DAILY_LOSS_LIMIT_USDC)),
        "TRADING_ENABLED": db.get("TRADING_ENABLED", "true" if settings.TRADING_ENABLED else "false"),
        "INITIAL_CAPITAL": db.get("INITIAL_CAPITAL", str(settings.INITIAL_CAPITAL)),
        "TRADE_MIN_PRICE": db.get("TRADE_MIN_PRICE", str(settings.TRADE_MIN_PRICE)),
        "TRADE_MAX_PRICE": db.get("TRADE_MAX_PRICE", str(settings.TRADE_MAX_PRICE)),
        "TRADE_ASSETS": db.get("TRADE_ASSETS", settings.TRADE_ASSETS),
        "TRADING_MODE": db.get("TRADING_MODE", settings.TRADING_MODE),
        "FAVORITE_MODE_ENTRY_SEC": db.get("FAVORITE_MODE_ENTRY_SEC", str(settings.FAVORITE_MODE_ENTRY_SEC)),
        "LIVE_POLL_INTERVAL_SECONDS": db.get("LIVE_POLL_INTERVAL_SECONDS", str(settings.LIVE_POLL_INTERVAL_SECONDS)),
        "FAVORITE_THRESHOLD": db.get("FAVORITE_THRESHOLD", str(settings.FAVORITE_THRESHOLD)),
        "MIN_EDGE": db.get("MIN_EDGE", str(settings.MIN_EDGE)),
        "MAX_BET_EDGE": db.get("MAX_BET_EDGE", str(settings.MAX_BET_EDGE)),
        "TRADE_ON_FLIP": db.get("TRADE_ON_FLIP", "true" if settings.TRADE_ON_FLIP else "false"),
        "FLIP_THRESHOLD": db.get("FLIP_THRESHOLD", str(settings.FLIP_THRESHOLD)),
        "AUTO_DEAD_ZONE": db.get("AUTO_DEAD_ZONE", "true" if settings.AUTO_DEAD_ZONE else "false"),
        # AUTO_DEAD_ZONE_WIDTH удалён — вместо него движок читает DEAD_ZONE_WIDTH
        "FAVORITE_MIN_PRICE": db.get("FAVORITE_MIN_PRICE", "0.55"),
        "FAVORITE_MAX_PRICE": db.get("FAVORITE_MAX_PRICE", "0.95"),
        "MAX_PRICE_DRIFT": db.get("MAX_PRICE_DRIFT", str(settings.MAX_PRICE_DRIFT)),
        "OUTSIDER_MAX_PRICE": db.get("OUTSIDER_MAX_PRICE", str(settings.OUTSIDER_MAX_PRICE)),
        "FAVORITE_MIN_EDGE": db.get("FAVORITE_MIN_EDGE", str(FAVORITE_MIN_EDGE)),
        "NO_MIN_EDGE": db.get("NO_MIN_EDGE", str(NO_MIN_EDGE)),
        "CRYPTO_MIN_EDGE": db.get("CRYPTO_MIN_EDGE", str(CRYPTO_MIN_EDGE)),
        "MAX_EDGE_FILTER": db.get("MAX_EDGE_FILTER", str(MAX_EDGE_FILTER)),
        "USE_CRYPTO_CONFIRM": db.get("USE_CRYPTO_CONFIRM", "false"),
        "CRYPTO_STANDALONE": db.get("CRYPTO_STANDALONE", "false")
    }

    for asset in settings.asset_list:
        asset_upper = asset.upper()
        settings_dict[f"TRADING_MODE_{asset_upper}"] = db.get(f"TRADING_MODE_{asset_upper}", "")
        settings_dict[f"MIN_EDGE_{asset_upper}"] = db.get(f"MIN_EDGE_{asset_upper}", "")
        settings_dict[f"TRADE_MAX_PRICE_{asset_upper}"] = db.get(f"TRADE_MAX_PRICE_{asset_upper}", "")

    return settings_dict

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

class BulkSettings(BaseModel):
    settings: dict[str, Union[str, int, float, bool]]

from fastapi import Depends
from polyflip.db.connection import get_db_session

@router.put("/bulk")
async def update_settings_bulk(
    payload: BulkSettings, 
    request: Request = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Массовое обновление настроек за один запрос для обхода лимитов rate limiter.
    """
    errors = {}
    saved = []
    for key, val in payload.settings.items():
        if val is None:
            errors[key] = "Value cannot be null"
            continue
        if not isinstance(val, (str, int, float, bool)):
            errors[key] = f"Invalid type: {type(val).__name__}"
            continue
        try:
            val_str = str(val).lower() if isinstance(val, bool) else str(val)
            await update_setting(key, SettingValue(value=val_str), request=request, db=db)
            saved.append(key)
        except HTTPException as e:
            errors[key] = e.detail
        except Exception as e:
            errors[key] = str(e)
    
    return {"status": "partial" if errors else "ok", "saved": saved, "errors": errors}

@router.api_route("/{key}", methods=["PUT", "POST"])
async def update_setting(key: str, payload: SettingValue, request: Request = None, db: AsyncSession = Depends(get_db_session)):
    """
    Обновляет или создает настройку в БД.
    """
    class SessionContext:
        def __init__(self, passed_db):
            self.passed_db = passed_db
            self.session = None
            self.own = False

        async def __aenter__(self):
            if self.passed_db is not None:
                self.session = self.passed_db
            else:
                self.session = await async_session().__aenter__()
                self.own = True
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.own:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
    # Backward-compatibility: фронт может слать MAX_EDGE — редиректим в MAX_BET_EDGE
    if key == "MAX_EDGE":
        key = "MAX_BET_EDGE"
    # valid_keys берётся из реестра — единственного источника истины
    valid_keys = list(_registry_editable_keys())
    
    is_per_asset_key = False
    for asset in settings.asset_list:
        asset_upper = asset.upper()
        if key in [f"TRADING_MODE_{asset_upper}", f"MIN_EDGE_{asset_upper}", f"TRADE_MAX_PRICE_{asset_upper}"]:
            is_per_asset_key = True
            break
            
    if key not in valid_keys and not is_per_asset_key:
        raise HTTPException(status_code=400, detail="Invalid setting key")

    # Валидация и нормализация порогов вероятности флипа и мертвой зоны
    if key in ["TRADE_NO_FLIP_THRESHOLD", "DEAD_ZONE_WIDTH"]:
        try:
            val = float(payload.value)
            if val < 0.0 or val > 100.0:
                raise HTTPException(status_code=400, detail=f"Value for {key} must be between 0 and 100")
            # Если прислали проценты (больше 1.0), автоматически переводим в доли для хранения в БД
            if val > 1.0:
                payload.value = str(val / 100.0)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Value for {key} must be a number")

    if key in ["MIN_EDGE", "MAX_BET_EDGE", "MAX_EDGE_FILTER"] or key.startswith("MIN_EDGE_"):
        if key.startswith("MIN_EDGE_") and payload.value == "":
            pass
        else:
            try:
                val = float(payload.value)
                if val <= 0:
                    raise HTTPException(status_code=400, detail=f"{key} must be positive")
                if val > 1.0:
                    # Введено как процент (напр. 5.0 → 0.05)
                    if val > 100.0:
                        raise HTTPException(status_code=400, detail=f"{key} must be ≤ 100%")
                    payload.value = f"{val / 100.0:.6f}".rstrip('0').rstrip('.')
                else:
                    # Введено как доля (напр. 0.05 = 5%)
                    if val < 0.005:
                        raise HTTPException(status_code=400, detail=f"{key} as fraction must be ≥ 0.005 (0.5%)")
                    
                if key in ["MIN_EDGE", "MAX_BET_EDGE", "MAX_EDGE_FILTER"]:
                    # Cross-validation: MIN_EDGE < MAX_EDGE_FILTER ≤ MAX_BET_EDGE
                    norm_val = float(payload.value)
                    async with SessionContext(db) as session:
                        if key == "MAX_BET_EDGE":
                            min_edge_row = (await session.execute(select(RuntimeSettings).where(RuntimeSettings.key == "MIN_EDGE"))).scalar_one_or_none()
                            current_min = float(min_edge_row.value) if min_edge_row else settings.MIN_EDGE
                            if norm_val <= current_min:
                                raise HTTPException(status_code=400, detail=f"MAX_BET_EDGE ({norm_val}) must be greater than MIN_EDGE ({current_min})")
                        elif key == "MIN_EDGE":
                            max_edge_row = (await session.execute(select(RuntimeSettings).where(RuntimeSettings.key == "MAX_BET_EDGE"))).scalar_one_or_none()
                            current_max = float(max_edge_row.value) if max_edge_row else getattr(settings, 'MAX_BET_EDGE', 0.10)
                            if norm_val >= current_max:
                                raise HTTPException(status_code=400, detail=f"MIN_EDGE ({norm_val}) must be less than MAX_BET_EDGE ({current_max})")
                        elif key == "MAX_EDGE_FILTER":
                            # MAX_EDGE_FILTER (фильтр аномалий) должен быть ≤ MAX_BET_EDGE (масштабирование)
                            max_bet_row = (await session.execute(select(RuntimeSettings).where(RuntimeSettings.key == "MAX_BET_EDGE"))).scalar_one_or_none()
                            current_max_bet = float(max_bet_row.value) if max_bet_row else getattr(settings, 'MAX_BET_EDGE', 0.40)
                            if norm_val > current_max_bet:
                                raise HTTPException(status_code=400, detail=f"MAX_EDGE_FILTER ({norm_val}) не должен превышать MAX_BET_EDGE ({current_max_bet})")
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{key} must be a number")


    if key == "FAVORITE_THRESHOLD":
        try:
            val = float(payload.value)
            if not (0.5 <= val <= 0.99):
                raise HTTPException(status_code=400, detail="FAVORITE_THRESHOLD must be between 0.50 and 0.99")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="FAVORITE_THRESHOLD must be a number")

    if key == "DAILY_LOSS_LIMIT_USDC":
        try:
            val = float(payload.value)
            if val >= 0.0:
                raise HTTPException(status_code=400, detail="DAILY_LOSS_LIMIT_USDC must be strictly negative (e.g., -100)")
            if val < -100000.0:
                raise HTTPException(status_code=400, detail="Daily loss limit is too large (> $100k)")
        except ValueError:
            raise HTTPException(status_code=400, detail="Value for DAILY_LOSS_LIMIT_USDC must be a number")

    if key == "TRADING_MODE" or key.startswith("TRADING_MODE_"):
        allowed = ("ml", "favorite", "CRYPTO", "") if key.startswith("TRADING_MODE_") else ("ml", "favorite", "CRYPTO")
        if payload.value not in allowed:
            raise HTTPException(status_code=400, detail=f"{key} must be 'ml', 'favorite', 'CRYPTO' or empty")

    if key == "TRADE_MAX_PRICE" or key.startswith("TRADE_MAX_PRICE_"):
        if key.startswith("TRADE_MAX_PRICE_") and payload.value == "":
            pass
        else:
            try:
                val = float(payload.value)
                if not (0.01 <= val <= 0.99):
                    raise HTTPException(status_code=400, detail=f"{key} must be between 0.01 and 0.99")
                payload.value = str(val)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{key} must be a number")

    if key == "FAVORITE_MODE_ENTRY_SEC":
        try:
            val = int(payload.value)
            if not (30 <= val <= 600):
                raise HTTPException(status_code=400, detail="FAVORITE_MODE_ENTRY_SEC must be between 30 and 600 seconds")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="FAVORITE_MODE_ENTRY_SEC must be an integer")

    if key == "FLIP_THRESHOLD":
        try:
            val = float(payload.value)
            if not (0.50 <= val <= 0.99):
                raise HTTPException(status_code=400, detail="FLIP_THRESHOLD must be 0.50..0.99")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="FLIP_THRESHOLD must be a number")

    if key in ["FAVORITE_MIN_PRICE", "FAVORITE_MAX_PRICE"]:
        try:
            val = float(payload.value)
            if not (0.01 <= val <= 0.99):
                raise HTTPException(status_code=400, detail=f"{key} must be between 0.01 and 0.99")
            
            # Cross-validation: MIN < MAX
            async with async_session() as session:
                if key == "FAVORITE_MAX_PRICE":
                    current_min_row = (await session.execute(select(RuntimeSettings).where(RuntimeSettings.key == "FAVORITE_MIN_PRICE"))).scalar_one_or_none()
                    current_min = float(current_min_row.value) if current_min_row else 0.55
                    if val <= current_min:
                        raise HTTPException(status_code=400, detail=f"FAVORITE_MAX_PRICE ({val}) must be > FAVORITE_MIN_PRICE ({current_min})")
                elif key == "FAVORITE_MIN_PRICE":
                    current_max_row = (await session.execute(select(RuntimeSettings).where(RuntimeSettings.key == "FAVORITE_MAX_PRICE"))).scalar_one_or_none()
                    current_max = float(current_max_row.value) if current_max_row else 0.95
                    if val >= current_max:
                        raise HTTPException(status_code=400, detail=f"FAVORITE_MIN_PRICE ({val}) must be < FAVORITE_MAX_PRICE ({current_max})")
            
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{key} must be a number")

    if key == "MAX_BET_SIZE_USDC":
        try:
            val = float(payload.value)
            if val < 1.0:
                raise HTTPException(status_code=400, detail="MAX_BET_SIZE_USDC must be >= 1.0")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="MAX_BET_SIZE_USDC must be a number")

    if key == "NO_MIN_EDGE":
        try:
            val = float(payload.value)
            if not ((0.5 <= val <= 30.0) or (0.005 <= val <= 0.30)):
                raise HTTPException(status_code=400, detail="NO_MIN_EDGE must be between 0.5% and 30% (or 0.005 and 0.30)")
            if val > 0.30:
                payload.value = str(val / 100.0)
            else:
                payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="NO_MIN_EDGE must be a number")

    if key == "TRADE_ON_FLIP":
        if payload.value.lower() not in ("true", "false"):
            raise HTTPException(status_code=400, detail="TRADE_ON_FLIP must be 'true' or 'false'")
        payload.value = payload.value.lower()

    if key == "AUTO_DEAD_ZONE":
        if payload.value.lower() not in ("true", "false"):
            raise HTTPException(status_code=400, detail="AUTO_DEAD_ZONE must be 'true' or 'false'")
        payload.value = payload.value.lower()

    if key == "MAX_EDGE_FILTER":
        try:
            val = float(payload.value)
            if not (0.05 <= val <= 0.50):
                raise HTTPException(status_code=400, detail="MAX_EDGE_FILTER must be 0.05..0.50")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="MAX_EDGE_FILTER must be a number")

    if key == "LIVE_POLL_INTERVAL_SECONDS":
        try:
            val = int(payload.value)
            if not (2 <= val <= 300):
                raise HTTPException(status_code=400, detail="LIVE_POLL_INTERVAL_SECONDS must be between 2 and 300 seconds")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="LIVE_POLL_INTERVAL_SECONDS must be an integer")

    async with SessionContext(db) as session:
        # Получить старое значение перед изменением
        old_row = (await session.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == key)
        )).scalar_one_or_none()
        old_value = old_row.value if old_row else None

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
            
        # Записать историю
        config_log = StrategyConfig(
            key=key,
            old_value=old_value,
            new_value=payload.value,
            changed_at=now,
            changed_by="user",
            source_ip=request.client.host if (request and request.client) else None,
            note=None,
        )
        session.add(config_log)
            
        await session.commit()
        logger.info("setting_updated", key=key, value=payload.value)
        
    return {"status": "ok", "key": key, "value": payload.value}
