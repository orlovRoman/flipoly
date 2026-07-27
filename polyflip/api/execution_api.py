from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Literal, Optional, List
from sqlalchemy import select, and_, or_, desc
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from polyflip.db.connection import get_db_session
from polyflip.api.auth import verify_api_key
from polyflip.db.models import RuntimeSettings
from polyflip.db.execution_models import ExecutionWorkerStatus
from polyflip.execution.config import ExecutionSettings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/execution", tags=["Execution"], dependencies=[Depends(verify_api_key)])

class KillSwitchRequest(BaseModel):
    enabled: bool

@router.get("/status")
async def get_live_trading_status(db: AsyncSession = Depends(get_db_session)):
    """
    Returns the current live trading status and execution mode, plus worker status.
    """
    key = "LIVE_TRADING_ENABLED"
    existing = (await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == key)
    )).scalar_one_or_none()
    
    enabled = existing is not None and existing.value.lower() == "true"
    settings = ExecutionSettings()
    
    # Get latest worker status
    worker_status = (await db.execute(
        select(ExecutionWorkerStatus)
        .where(ExecutionWorkerStatus.execution_mode == "LIVE")
        .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    
    worker_data = None
    if worker_status:
        # Convert to dict manually or rely on fastapi response model (but we don't have one here)
        worker_data = {
            "worker_id": worker_status.worker_id,
            "execution_mode": worker_status.execution_mode,
            "heartbeat_at": worker_status.heartbeat_at.isoformat() if worker_status.heartbeat_at else None,
            "gateway_ready": worker_status.gateway_ready,
            "credentials_loaded": worker_status.credentials_loaded,
            "wallet_address": worker_status.wallet_address,
            "balance_usdc": float(worker_status.balance_usdc) if worker_status.balance_usdc is not None else None,
            "collateral_allowance_ready": worker_status.collateral_allowance_ready,
            "conditional_allowance_ready": worker_status.conditional_allowance_ready,
            "last_error_code": worker_status.last_error_code,
            "last_error_message": worker_status.last_error_message,
        }
    
    return {
        "live_trading_enabled": enabled,
        "execution_mode": settings.execution_mode.value,
        "worker_status": worker_data
    }

@router.put("/kill-switch")
async def toggle_kill_switch(payload: KillSwitchRequest, db: AsyncSession = Depends(get_db_session)):
    """
    Управляет глобальным рубильником LIVE-торговли.
    Перед включением проверяет свежий execution_worker_status из БД.
    """
    key = "LIVE_TRADING_ENABLED"
    
    # Защищаем чтение/запись блокировкой FOR UPDATE
    stmt = select(RuntimeSettings).where(RuntimeSettings.key == key)
    bind = db.bind
    if bind and bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update()
        
    existing = (await db.execute(stmt)).scalar_one_or_none()
    
    if payload.enabled:
        settings = ExecutionSettings()
        if settings.execution_mode.value != "LIVE":
            raise HTTPException(status_code=400, detail="Cannot enable LIVE trading: Execution mode is not LIVE")
            
        worker_status = (await db.execute(
            select(ExecutionWorkerStatus)
            .where(ExecutionWorkerStatus.execution_mode == "LIVE")
            .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        
        if not worker_status:
            raise HTTPException(status_code=400, detail="System not ready for LIVE trading: No worker status found in database")
            
        now = datetime.now(timezone.utc)
        if worker_status.heartbeat_at < now - timedelta(seconds=30):
            raise HTTPException(status_code=400, detail="System not ready for LIVE trading: Worker heartbeat is older than 30 seconds")
            
        if not worker_status.gateway_ready:
            raise HTTPException(status_code=400, detail=f"System not ready for LIVE trading: Worker gateway is not ready (Error: {worker_status.last_error_message})")
            
        if float(worker_status.balance_usdc or 0) < 5:
            raise HTTPException(status_code=400, detail=f"System not ready for LIVE trading: Insufficient USDC balance (Minimum $5, current {float(worker_status.balance_usdc or 0)})")
            
        if not worker_status.collateral_allowance_ready:
            raise HTTPException(status_code=400, detail="System not ready for LIVE trading: Collateral allowance is not ready")

    value = "true" if payload.enabled else "false"
    
    try:
        now = datetime.now(timezone.utc)
        if existing:
            existing.value = value
            existing.updated_at = now
            existing.updated_by = "api"
        else:
            db.add(RuntimeSettings(key=key, value=value, updated_at=now, updated_by="api"))
        
        await db.commit()
        logger.info("kill_switch_toggled", enabled=payload.enabled)
        return {"status": "ok", "live_trading_enabled": payload.enabled}
    except Exception as e:
        logger.exception("kill_switch_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.states import ACTIVE_POSITION_STATES

@router.get("/positions")
async def get_live_trading_positions(db: AsyncSession = Depends(get_db_session)):
    stmt = select(TradeHistory).where(
        TradeHistory.position_status.in_(ACTIVE_POSITION_STATES)
    ).order_by(TradeHistory.created_at.desc()).limit(100)
    
    res = await db.execute(stmt)
    trades = res.scalars().all()
    
    return [
        {
            "id": t.id,
            "market_id": t.market_id,
            "asset": t.asset,
            "outcome_bought": t.outcome_bought,
            "mode": t.mode,
            "entry_filled_shares": float(t.entry_filled_shares or 0),
            "entry_cost_usdc": float(t.entry_cost_usdc or 0),
            "remaining_shares": float(t.remaining_shares or 0),
            "realized_pnl_usdc": float(t.realized_pnl_usdc or 0),
            "position_status": t.position_status,
            "stop_loss_status": t.stop_loss_status,
            "take_profit_status": t.take_profit_status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in trades
    ]

@router.get("/requests")
async def get_live_trading_requests(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db_session)
):
    stmt = select(ExecutionRequest).order_by(ExecutionRequest.created_at.desc()).limit(limit)
    res = await db.execute(stmt)
    requests = res.scalars().all()
    
    return [
        {
            "id": str(r.id),
            "trade_history_id": r.trade_history_id,
            "intent": r.intent,
            "trigger_reason": r.trigger_reason,
            "market_id": r.market_id,
            "asset": r.asset,
            "state": r.state,
            "requested_mode": r.requested_mode,
            "requested_shares": float(r.requested_shares) if r.requested_shares else None,
            "limit_price": float(r.limit_price) if r.limit_price else None,
            "target_amount_usdc": float(r.target_amount_usdc) if r.target_amount_usdc else None,
            "filled_shares": float(r.filled_shares) if r.filled_shares else 0,
            "filled_cost_usdc": float(r.filled_cost_usdc) if r.filled_cost_usdc else 0,
            "ttl_seconds": r.ttl_seconds,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "error_reason": r.error_reason,
        }
        for r in requests
    ]
