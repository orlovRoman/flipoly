from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Literal
from sqlalchemy import select
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.db.connection import get_db_session
from polyflip.api.auth import verify_api_key
from polyflip.db.models import RuntimeSettings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/live-trading", tags=["Execution"], dependencies=[Depends(verify_api_key)])

class KillSwitchRequest(BaseModel):
    enabled: bool

from polyflip.execution.config import ExecutionSettings
from polyflip.execution.gateways.factory import build_execution_gateway

@router.get("/status")
async def get_live_trading_status(db: AsyncSession = Depends(get_db_session)):
    """
    Returns the current live trading status and execution mode.
    """
    key = "LIVE_TRADING_ENABLED"
    existing = (await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == key)
    )).scalar_one_or_none()
    
    enabled = existing is not None and existing.value.lower() == "true"
    settings = ExecutionSettings()
    
    return {
        "live_trading_enabled": enabled,
        "execution_mode": settings.execution_mode.value
    }

@router.put("/kill-switch")
async def toggle_kill_switch(payload: KillSwitchRequest, db: AsyncSession = Depends(get_db_session)):
    """
    Управляет глобальным рубильником LIVE-торговли.
    Проверяет готовность системы перед включением.
    """
    if payload.enabled:
        settings = ExecutionSettings()
        if settings.execution_mode.value != "LIVE":
            raise HTTPException(status_code=400, detail="Cannot enable LIVE trading: Execution mode is not LIVE")
            
        try:
            gateway = build_execution_gateway(settings)
            balance = await gateway.get_balance_allowance(asset_type="COLLATERAL")
            if balance.balance_usdc < 5:
                raise ValueError(f"Insufficient USDC balance: {balance.balance_usdc}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"System not ready for LIVE trading: {str(e)}")

    key = "LIVE_TRADING_ENABLED"
    value = "true" if payload.enabled else "false"
    
    try:
        existing = (await db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == key)
        )).scalar_one_or_none()

        now = datetime.now(timezone.utc)
        if existing:
            existing.value = value
            existing.updated_at = now
        else:
            db.add(RuntimeSettings(key=key, value=value, updated_at=now, updated_by="api"))
        
        await db.commit()
        logger.info("kill_switch_toggled", enabled=payload.enabled)
        return {"status": "ok", "live_trading_enabled": payload.enabled}
    except Exception as e:
        logger.exception("kill_switch_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
