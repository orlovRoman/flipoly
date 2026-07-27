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
from polyflip.db.execution_models import (
    ExecutionWorkerStatus,
    ExecutionRequest,
    ExecutionEvent,
)
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

    # Статус воркера для ТЕКУЩЕГО режима (не всегда LIVE)
    worker_status = (await db.execute(
        select(ExecutionWorkerStatus)
        .where(ExecutionWorkerStatus.execution_mode == settings.execution_mode.value)
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
        "kill_switch_available": settings.execution_mode.value == "LIVE",
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
async def get_live_trading_positions(
    mode: Optional[str] = Query(None, description="Фильтр по режиму: PAPER, SHADOW, LIVE"),
    db: AsyncSession = Depends(get_db_session),
):
    settings = ExecutionSettings()
    effective_mode = mode or settings.execution_mode.value

    stmt = select(TradeHistory).where(
        TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
        TradeHistory.mode == effective_mode,
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
    mode: Optional[str] = Query(None, description="Фильтр по режиму: PAPER, SHADOW, LIVE"),
    db: AsyncSession = Depends(get_db_session),
):
    settings = ExecutionSettings()
    effective_mode = mode or settings.execution_mode.value

    stmt = (
        select(ExecutionRequest)
        .where(ExecutionRequest.requested_mode == effective_mode)
        .order_by(ExecutionRequest.created_at.desc())
        .limit(limit)
    )
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


# ---------------------------------------------------------------------------
# Manual review endpoint
# ---------------------------------------------------------------------------

class ResolveReviewRequest(BaseModel):
    action: Literal["REQUEUE_AFTER_ALLOWANCE", "MARK_FAILED_NO_FILL"]
    operator: str
    note: str = ""


@router.post("/requests/{request_id}/resolve-review")
async def resolve_manual_review(
    request_id: str,
    body: ResolveReviewRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Переводит заявку из MANUAL_REVIEW_REQUIRED в новое состояние.

    REQUEUE_AFTER_ALLOWANCE — оператор убедился, что allowance выставлен;
        заявка возвращается в READY для повторной попытки.
        ⚠ Если у попытки уже есть provider_order_id — вызов запрещён
        (требуется сверка fills).

    MARK_FAILED_NO_FILL — fill не произошёл, allowance недостаточен;
        заявка переходит в FAILED.
    """
    now = datetime.now(timezone.utc)

    req = (
        await db.execute(
            select(ExecutionRequest).where(ExecutionRequest.id == request_id)
        )
    ).scalar_one_or_none()

    if req is None:
        raise HTTPException(status_code=404, detail="Request not found")

    if req.state != "MANUAL_REVIEW_REQUIRED":
        raise HTTPException(
            status_code=409,
            detail=f"Request is in state {req.state!r}, not MANUAL_REVIEW_REQUIRED",
        )

    old_state = req.state

    if body.action == "REQUEUE_AFTER_ALLOWANCE":
        # Проверяем: если у попытки уже есть provider_order_id — запрещаем
        from polyflip.db.execution_models import ExecutionAttempt

        last_attempt = (
            await db.execute(
                select(ExecutionAttempt)
                .where(ExecutionAttempt.request_id == req.id)
                .order_by(ExecutionAttempt.started_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if last_attempt and last_attempt.provider_order_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Cannot requeue: last attempt has a provider_order_id "
                    f"({last_attempt.provider_order_id!r}). "
                    "Reconcile provider fills first."
                ),
            )

        req.state = "READY"
        req.error_reason = None
        req.updated_at = now

    elif body.action == "MARK_FAILED_NO_FILL":
        req.state = "FAILED"
        req.error_reason = f"Manually marked as failed by {body.operator}: {body.note}"
        req.updated_at = now

    db.add(
        ExecutionEvent(
            level="INFO",
            event_type=f"MANUAL_REVIEW_{body.action}",
            message=(
                f"Operator {body.operator!r} resolved MANUAL_REVIEW_REQUIRED → "
                f"{req.state!r}. Note: {body.note or '—'}"
            ),
            source="execution_api",
            request_id=req.id,
            trade_history_id=req.trade_history_id,
            payload={
                "operator": body.operator,
                "action": body.action,
                "old_state": old_state,
                "new_state": req.state,
                "note": body.note,
            },
        )
    )

    await db.commit()

    return {
        "request_id": str(req.id),
        "old_state": old_state,
        "new_state": req.state,
        "action": body.action,
        "operator": body.operator,
    }
