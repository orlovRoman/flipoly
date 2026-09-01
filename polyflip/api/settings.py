from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional, Any, Union
from pydantic import BaseModel
from sqlalchemy import select
from datetime import datetime, timezone
import structlog
import os

from polyflip.db.connection import async_session
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import RuntimeSettings, StrategyConfig
from polyflip.api.auth import verify_api_key
from polyflip.config import settings
from polyflip.settings_registry import registry_defaults, editable_keys as _registry_editable_keys
from polyflip.execution.config import (
    LIVE_MIN_GROSS_BUY_USDC,
    POLYMARKET_MIN_ORDER_SHARES,
)

logger = structlog.get_logger(__name__)

def get_request(request: Request) -> Request:
    return request

router = APIRouter(prefix="/api/settings", tags=["Settings"], dependencies=[Depends(verify_api_key)])

class SettingValue(BaseModel):
    value: str

async def get_all_settings(db: Optional[AsyncSession] = None):
    """
    Возвращает текущие настройки (сначала из БД, если нет - из конфига/констант).
    """
    async with (async_session() if db is None else db) as session:
        result = await session.execute(select(RuntimeSettings))
        db_settings = {s.key: s.value for s in result.scalars().all()}

    from polyflip.settings_registry import registry_defaults
    
    # Базовые дефолты из реестра + перекрытие из БД
    settings_dict = {**registry_defaults(), **db_settings}

    # Динамические поля по активам, которых нет в базовом реестре
    for asset in settings.asset_list:
        asset_upper = asset.upper()
        settings_dict[f"TRADING_MODE_{asset_upper}"] = db_settings.get(f"TRADING_MODE_{asset_upper}", "")
        settings_dict[f"OUTS_MIN_EDGE_{asset_upper}"] = db_settings.get(f"OUTS_MIN_EDGE_{asset_upper}", "")
        settings_dict[f"FAVORITE_MIN_EDGE_{asset_upper}"] = db_settings.get(f"FAVORITE_MIN_EDGE_{asset_upper}", "")
        settings_dict[f"TRADE_MAX_PRICE_{asset_upper}"] = db_settings.get(f"TRADE_MAX_PRICE_{asset_upper}", "")
        settings_dict[f"TRADE_FLIP_THRESHOLD_{asset_upper}"] = db_settings.get(f"TRADE_FLIP_THRESHOLD_{asset_upper}", "")

    return settings_dict

@router.get("")
async def api_get_all_settings():
    """
    Возвращает текущие настройки (сначала из БД, если нет - из конфига/констант).
    """
    return await get_all_settings()

def _runtime_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_float(values: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(values.get(key, default))
        return value if value == value and value not in {float("inf"), float("-inf")} else default
    except (TypeError, ValueError, OverflowError):
        return default


@router.get("/runtime-info")
async def api_get_runtime_info():
    """
    Returns the effective trading configuration and execution contracts.

    This endpoint intentionally exposes no credentials. It combines runtime
    settings from the database with process-level execution mode and the
    constants used by the release/execution code, so the UI can distinguish
    an active setting from a hardcoded guard.
    """
    values = await get_all_settings()
    order_mode = str(values.get("LIVE_ORDER_MODE", "FAK")).strip().upper()
    paper_profile = str(values.get("PAPER_EXECUTION_PROFILE", "LIVE_PARITY")).strip().upper()
    weighted_mode = str(values.get("TRADING_POLICY_MODE", "LEGACY")).strip().upper()
    mrf_mode = str(values.get("MARKET_REGIME_FILTER_MODE", "OFF")).strip().upper()
    lgbm_mode = str(values.get("LIGHTGBM_DECISION_MODE", "SHADOW")).strip().upper()

    return {
        "execution_mode": str(os.getenv("EXECUTION_MODE", "PAPER")).strip().upper(),
        "trading_enabled": _runtime_bool(values, "TRADING_ENABLED"),
        "trading_mode": str(values.get("TRADING_MODE", "combined")).strip().upper(),
        "lightgbm_decision_mode": lgbm_mode,
        "weighted_policy": {
            "mode": weighted_mode,
            "id": str(values.get("WEIGHTED_POLICY_ID", "UNVERSIONED")),
            "market_weight": _runtime_float(values, "WEIGHTED_MARKET_WEIGHT", 0.90),
            "logreg_weight": _runtime_float(values, "WEIGHTED_LOGREG_WEIGHT", 0.05),
            "lgbm_weight": _runtime_float(values, "WEIGHTED_LGBM_WEIGHT", 0.05),
            "mrf_beta": _runtime_float(values, "WEIGHTED_MRF_BETA", 0.0),
            "min_net_ev_favorite": _runtime_float(values, "WEIGHTED_MIN_NET_EV_FAVORITE", 0.03),
            "min_net_ev_outsider": _runtime_float(values, "WEIGHTED_MIN_NET_EV_OUTSIDER", 0.03),
            "fixed_bet_usdc": _runtime_float(values, "WEIGHTED_FIXED_BET_USDC", 1.0),
        },
        "mrf": {
            "mode": mrf_mode,
            "version": int(_runtime_float(values, "MARKET_REGIME_FILTER_VERSION", 1)),
        },
        "live": {
            "kill_switch": _runtime_bool(values, "LIVE_TRADING_ENABLED"),
            "mirror_enabled": _runtime_bool(values, "LIVE_MIRROR_ENABLED"),
            "release_mode": str(values.get("LIVE_RELEASE_MODE", "DISABLED")).strip().upper(),
        },
        "order": {
            "mode": order_mode,
            "gtc_ttl_seconds": _runtime_float(values, "LIVE_GTC_TTL_SECONDS", 5.0),
            "fak_retry_max_attempts": int(_runtime_float(values, "LIVE_FAK_RETRY_MAX_ATTEMPTS", 3)),
            "fak_retry_delay_sec": _runtime_float(values, "LIVE_FAK_RETRY_DELAY_SEC", 0.75),
            "maker_reprice_on_cross": _runtime_bool(values, "LIVE_MAKER_REPRICE_ON_CROSS", True),
            "maker_reprice_max_retries": int(_runtime_float(values, "LIVE_MAKER_REPRICE_MAX_RETRIES", 1)),
            "maker_tick_size": _runtime_float(values, "LIVE_MAKER_TICK_SIZE", 0.01),
        },
        "paper": {
            "profile": paper_profile,
            "live_delay_sec": _runtime_float(values, "PAPER_LIVE_DELAY_SEC", 2.0),
            "slippage_pct": _runtime_float(values, "PAPER_SLIPPAGE_PCT", 0.5),
            "fee_model": str(values.get("PAPER_FEE_MODEL", "FLAT_NOTIONAL")).strip().upper(),
            "fee_rate": _runtime_float(values, "PAPER_FEE_RATE", 0.002),
            "fee_exponent": _runtime_float(values, "PAPER_FEE_EXPONENT", 1.0),
            "min_order_shares": _runtime_float(
                values, "PAPER_MIN_ORDER_SHARES", float(POLYMARKET_MIN_ORDER_SHARES)
            ),
        },
        "limits": {
            "max_open_positions": int(_runtime_float(values, "MAX_OPEN_POSITIONS", 20)),
            "max_total_exposure_usdc": _runtime_float(values, "MAX_TOTAL_EXPOSURE_USDC", 50.0),
            "max_single_order_usdc": _runtime_float(values, "MAX_SINGLE_ORDER_USDC", 1.0),
            "confirm_threshold_usdc": _runtime_float(values, "CONFIRM_THRESHOLD_USDC", 5.0),
            "daily_loss_limit_usdc": _runtime_float(values, "DAILY_LOSS_LIMIT_USDC", -100.0),
        },
        "contracts": {
            "live_min_gross_buy_usdc": float(LIVE_MIN_GROSS_BUY_USDC),
            "maker_min_order_shares": float(POLYMARKET_MIN_ORDER_SHARES),
            "small_order_route": "FAK_RETRY",
            "maker_min_source": "execution.config constant",
            "live_min_source": "release_gate constant",
        },
        "costs": {
            "polymarket_fee_rate": _runtime_float(values, "POLYMARKET_FEE_RATE", 0.002),
        },
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
                    *[f"TRADE_FLIP_THRESHOLD_{a.upper()}" for a in settings.asset_list],
                    *[f"AUTO_FLIP_THRESHOLD_{a.upper()}" for a in settings.asset_list]
                ])
            )
        )
        db = {s.key: s.value for s in result.scalars().all()}

    dead_zone = float(db.get("DEAD_ZONE_WIDTH", getattr(settings, 'DEAD_ZONE_WIDTH', 0.15)))

    # Per-asset
    per_asset = {}
    for asset in settings.asset_list:
        manual_key = f"TRADE_FLIP_THRESHOLD_{asset.upper()}"
        auto_key = f"AUTO_FLIP_THRESHOLD_{asset.upper()}"
        
        manual_val = db.get(manual_key)
        if manual_val is not None and manual_val.strip() != "":
            key = manual_key
        else:
            key = auto_key
            
        val_str = db.get(key)
        if val_str is not None and val_str.strip() != "":
            try:
                asset_flip = float(val_str)
                per_asset[asset] = {
                    "flip_threshold": asset_flip,
                    "recommended_no_flip": round(asset_flip - dead_zone, 4),
                    "is_auto_calibrated": key == auto_key
                }
            except ValueError:
                pass

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
    request: Optional[Request] = Depends(get_request),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Массовое обновление настроек за один запрос для обхода лимитов rate limiter.
    """
    from fastapi.params import Depends
    if isinstance(request, Depends):
        request = None
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

@router.put("/security/{key}", operation_id="update_security_setting_put")
@router.post("/security/{key}", operation_id="update_security_setting_post")
async def update_security_setting(key: str, payload: SettingValue, request: Optional[Request] = Depends(get_request), db: AsyncSession = Depends(get_db_session)):
    """
    Отдельный эндпоинт для обновления флагов безопасности, которые недоступны через основной API.
    """
    from fastapi.params import Depends
    if isinstance(request, Depends):
        request = None
    if key not in ["TRADING_ENABLED", "BYPASS_BET_SIZE_CHECK"]:
        raise HTTPException(status_code=400, detail="Invalid security key")
    
    if payload.value not in ["true", "false"]:
        raise HTTPException(status_code=400, detail="Value must be 'true' or 'false'")
    
    from polyflip.db.models import RuntimeSettings
    from sqlalchemy import select

    # Если db передан снаружи (FastAPI Depends), не используем async with, так как сессия закроется.
    session = db if db is not None else async_session()
    own_session = db is None
    
    try:
        existing = (await session.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == key)
        )).scalar_one_or_none()

        now = datetime.now(timezone.utc)
        if existing:
            existing.value = payload.value
            existing.updated_at = now
            existing.updated_by = "api"
        else:
            session.add(RuntimeSettings(key=key, value=payload.value, updated_at=now, updated_by="api"))
        
        await session.commit()
    finally:
        if own_session:
            await session.close()

    return {"message": "Security setting updated", "key": key, "value": payload.value}


@router.put("/{key}", operation_id="update_setting_put")
@router.post("/{key}", operation_id="update_setting_post")
async def update_setting(key: str, payload: SettingValue, request: Optional[Request] = Depends(get_request), db: AsyncSession = Depends(get_db_session)):
    """
    Обновляет или создает настройку в БД.
    """
    from fastapi.params import Depends
    if isinstance(request, Depends):
        request = None
    class SessionContext:
        def __init__(self, passed_db):
            self.passed_db = passed_db
            self.session = None
            self.own = False

        async def __aenter__(self):
            from fastapi.params import Depends
            if self.passed_db is not None and not isinstance(self.passed_db, Depends):
                self.session = self.passed_db
            else:
                self.session = await async_session().__aenter__()
                self.own = True
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.own:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
    # Backward-compatibility: старый STOP_LOSS_PCT → STOP_LOSS_PCT_FAVORITE
    if key == "STOP_LOSS_PCT":
        logger.warning("deprecated_key_redirect", old_key="STOP_LOSS_PCT", new_key="STOP_LOSS_PCT_FAVORITE")
        key = "STOP_LOSS_PCT_FAVORITE"
    # valid_keys берётся из реестра — единственного источника истины
    valid_keys = list(_registry_editable_keys())
    
    is_per_asset_key = False
    for asset in settings.asset_list:
        asset_upper = asset.upper()
        if key in [
            f"TRADING_MODE_{asset_upper}", 
            f"OUTS_MIN_EDGE_{asset_upper}",
            f"FAVORITE_MIN_EDGE_{asset_upper}",
            f"TRADE_MAX_PRICE_{asset_upper}",
            f"FLIP_THRESHOLD_{asset_upper}",
            f"TRADE_FLIP_THRESHOLD_{asset_upper}"
        ]:
            is_per_asset_key = True
            break
            
    if key not in valid_keys and not is_per_asset_key:
        raise HTTPException(status_code=400, detail="Invalid setting key")

    # Валидация и нормализация порогов вероятности флипа и мертвой зоны
    is_threshold_key = (
        "THRESHOLD" in key or
        key in ["FLIP_THRESHOLD", "TRADE_FLIP_THRESHOLD", "NO_FLIP_THRESHOLD", "TRADE_NO_FLIP_THRESHOLD", "DEAD_ZONE_WIDTH"] or
        key.startswith("FLIP_THRESHOLD_") or key.startswith("TRADE_FLIP_THRESHOLD_") or key.startswith("NO_FLIP_THRESHOLD_") or key.startswith("TRADE_NO_FLIP_THRESHOLD_") or key.startswith("AUTO_FLIP_THRESHOLD_")
    )
    if is_threshold_key:
        if (key.startswith("FLIP_THRESHOLD_") or key.startswith("TRADE_FLIP_THRESHOLD_") or key.startswith("NO_FLIP_THRESHOLD_") or key.startswith("TRADE_NO_FLIP_THRESHOLD_") or key.startswith("AUTO_FLIP_THRESHOLD_")) and payload.value == "":
            pass
        else:
            try:
                val = float(payload.value)
                if val <= 0.0 or val >= 100.0:
                    raise HTTPException(status_code=400, detail=f"Value for {key} must be between 0 and 100")
                # Если прислали проценты (больше 1.0), автоматически переводим в доли для хранения в БД (например 40 -> 0.40)
                if val > 1.0:
                    val = val / 100.0
                if not (0.0 < val < 1.0):
                    raise HTTPException(status_code=400, detail=f"Value for {key} must be between 0.0 and 1.0")
                payload.value = str(round(val, 4))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Value for {key} must be a number")

    if key in ["OUTS_MIN_EDGE", "FAVORITE_MIN_EDGE"] or key.startswith("OUTS_MIN_EDGE_") or key.startswith("FAVORITE_MIN_EDGE_"):
        if (key.startswith("OUTS_MIN_EDGE_") or key.startswith("FAVORITE_MIN_EDGE_")) and payload.value == "":
            pass
        else:
            try:
                # В случае передачи строки с запятой (напр. "1,0") заменяем на точку
                val = float(str(payload.value).replace(",", "."))
                if key != "FAVORITE_MIN_EDGE" and val <= 0:
                    raise HTTPException(status_code=400, detail=f"{key} must be positive")
                if key == "FAVORITE_MIN_EDGE" and val < -100.0:
                    raise HTTPException(status_code=400, detail=f"{key} must be ≥ -100%")
                if abs(val) > 1.0:
                    # Введено как процент: 50 → 0.5, 5 → 0.05, -10 → -0.1
                    if abs(val) > 100.0:
                        raise HTTPException(status_code=400, detail=f"{key} must be ≤ 100%")
                    payload.value = f"{val / 100.0:.6f}".rstrip('0').rstrip('.')
                elif abs(val) == 1.0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} = 1.0 is ambiguous: enter as fraction (0.01) or percent (1)"
                    )
                else:
                    # Уже доля: 0.5 остаётся 0.5 (50%), 0.05 остаётся 0.05 (5%)
                    payload.value = f"{val:.6f}".rstrip('0').rstrip('.')
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{key} must be a number")


    if key == "FAVORITE_THRESHOLD":
        try:
            val = float(payload.value)
            if not (0.01 <= val <= 0.99):
                raise HTTPException(status_code=400, detail="FAVORITE_THRESHOLD must be between 0.01 and 0.99")
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
        if payload.value == "CRYPTO":
            payload.value = "lightgbm"
        allowed_per_asset = ("ml", "lightgbm", "combined", "")
        allowed_global   = ("ml", "lightgbm", "combined")
        allowed = allowed_per_asset if key.startswith("TRADING_MODE_") else allowed_global
        if payload.value not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must be 'ml', 'lightgbm' or 'combined'"
            )

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

    # Removed NO_MIN_EDGE block since it was replaced by OUTS_MIN_EDGE
    if key in ["TRADE_ON_FAVORITE", "TRADE_ON_FLIP"]:
        if payload.value.lower() not in ("true", "false"):
            raise HTTPException(status_code=400, detail=f"{key} must be 'true' or 'false'")
        payload.value = payload.value.lower()

    if key == "AUTO_DEAD_ZONE":
        if payload.value.lower() not in ("true", "false"):
            raise HTTPException(status_code=400, detail="AUTO_DEAD_ZONE must be 'true' or 'false'")
        payload.value = payload.value.lower()

    if key == "LIVE_POLL_INTERVAL_SECONDS":
        try:
            val_clean = str(payload.value).replace(",", ".")
            val = int(round(float(val_clean)))
            if not (2 <= val <= 300):
                raise HTTPException(status_code=400, detail="LIVE_POLL_INTERVAL_SECONDS must be between 2 and 300 seconds")
            payload.value = str(val)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="LIVE_POLL_INTERVAL_SECONDS must be an integer")

    if key == "STOP_LOSS_ENABLED":
        if payload.value.lower() not in ("true", "false"):
            raise HTTPException(status_code=400, detail="STOP_LOSS_ENABLED must be 'true' or 'false'")
        payload.value = payload.value.lower()

    if key in ["STOP_LOSS_PCT", "STOP_LOSS_PCT_FAVORITE", "STOP_LOSS_PCT_OUTSIDER"]:
        try:
            val = float(payload.value)
            if not (1.0 <= val <= 99.0):
                raise HTTPException(status_code=400, detail="STOP_LOSS_PCT must be between 1.0 and 99.0")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="STOP_LOSS_PCT must be a number")

    if key == "STOP_LOSS_CHECK_SEC":
        try:
            val = int(payload.value)
            if not (10 <= val <= 300):
                raise HTTPException(status_code=400, detail="STOP_LOSS_CHECK_SEC must be between 10 and 300 seconds")
            payload.value = str(val)
        except ValueError:
            raise HTTPException(status_code=400, detail="STOP_LOSS_CHECK_SEC must be an integer")

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
