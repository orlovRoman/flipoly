from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Literal, Optional, List
from sqlalchemy import select, and_, or_, desc, func
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
    LiveMirrorCandidate,
)
from polyflip.execution.config import ExecutionSettings


logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/execution", tags=["Execution"], dependencies=[Depends(verify_api_key)])

class KillSwitchRequest(BaseModel):
    enabled: bool

@router.get("/status")
async def get_live_trading_status(db: AsyncSession = Depends(get_db_session)):
    """
    Returns live trading status, mirror status, and worker status for PAPER, SHADOW, and LIVE modes.
    """
    settings = ExecutionSettings()

    async def get_worker_dict(mode_name: str):
        ws = (await db.execute(
            select(ExecutionWorkerStatus)
            .where(ExecutionWorkerStatus.execution_mode == mode_name)
            .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if not ws:
            return None
        return {
            "worker_id": ws.worker_id,
            "execution_mode": ws.execution_mode,
            "heartbeat_at": ws.heartbeat_at.isoformat() if ws.heartbeat_at else None,
            "gateway_ready": ws.gateway_ready,
            "credentials_loaded": ws.credentials_loaded,
            "wallet_address": ws.wallet_address,
            "balance_usdc": float(ws.balance_usdc) if ws.balance_usdc is not None else None,
            "collateral_allowance_ready": ws.collateral_allowance_ready,
            "conditional_allowance_ready": ws.conditional_allowance_ready,
            "last_error_code": ws.last_error_code,
            "last_error_message": ws.last_error_message,
        }

    paper_worker = await get_worker_dict("PAPER")
    shadow_worker = await get_worker_dict("SHADOW")
    live_worker = await get_worker_dict("LIVE")

    async def _flag(key: str) -> bool:
        row = (await db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == key)
        )).scalar_one_or_none()
        return row is not None and row.value.lower() == "true"

    async def _flag_str(key: str, default: str) -> str:
        row = (await db.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == key)
        )).scalar_one_or_none()
        return row.value if row else default

    live_mirror_enabled = await _flag("LIVE_MIRROR_ENABLED")
    live_release_mode = await _flag_str("LIVE_RELEASE_MODE", "DISABLED")
    live_trading_enabled = await _flag("LIVE_TRADING_ENABLED")

    # Количество кандидатов по состояниям
    candidate_counts = {}
    for state in ("NEW", "ELIGIBLE", "REJECTED", "RELEASED"):
        cnt = await db.scalar(
            select(func.count()).select_from(LiveMirrorCandidate)
            .where(LiveMirrorCandidate.state == state)
        )
        candidate_counts[state] = cnt or 0

    return {
        "live_trading_enabled": live_trading_enabled,
        "execution_mode": settings.execution_mode.value,
        "kill_switch_available": live_worker is not None,
        "paper_worker": paper_worker,
        "shadow_worker": shadow_worker,
        "live_worker": live_worker,
        "worker_status": live_worker,
        "live_mirror_enabled": live_mirror_enabled,
        "live_release_mode": live_release_mode,
        "mirror_candidates": candidate_counts,
    }


@router.put("/kill-switch")
async def toggle_kill_switch(payload: KillSwitchRequest, db: AsyncSession = Depends(get_db_session)):
    """
    Управляет глобальным рубильником LIVE-торговли.
    Перед включением проверяет наличие и свежесть heartbeat от LIVE-воркера.
    """
    key = "LIVE_TRADING_ENABLED"
    
    # Защищаем чтение/запись блокировкой FOR UPDATE
    stmt = select(RuntimeSettings).where(RuntimeSettings.key == key)
    bind = db.bind
    if bind and bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update()
        
    existing = (await db.execute(stmt)).scalar_one_or_none()
    
    if payload.enabled:
        worker_status = (await db.execute(
            select(ExecutionWorkerStatus)
            .where(ExecutionWorkerStatus.execution_mode == "LIVE")
            .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        
        if not worker_status:
            raise HTTPException(status_code=409, detail="Cannot enable LIVE trading: LIVE worker is not running (no status in DB)")
            
        now = datetime.now(timezone.utc)
        hb_at = worker_status.heartbeat_at
        if hb_at and hb_at.tzinfo is None:
            hb_at = hb_at.replace(tzinfo=timezone.utc)
            
        if not hb_at or hb_at < now - timedelta(seconds=30):
            raise HTTPException(status_code=409, detail="Cannot enable LIVE trading: LIVE worker heartbeat is older than 30 seconds")
            
        if not worker_status.gateway_ready:
            raise HTTPException(status_code=409, detail=f"Cannot enable LIVE trading: LIVE gateway is not ready ({worker_status.last_error_message})")
            
        if float(worker_status.balance_usdc or 0) < 5:
            raise HTTPException(status_code=409, detail=f"Cannot enable LIVE trading: Insufficient USDC balance (Minimum $5, current {float(worker_status.balance_usdc or 0)})")
            
        if not worker_status.collateral_allowance_ready:
            raise HTTPException(status_code=409, detail="Cannot enable LIVE trading: Collateral allowance is not ready")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("kill_switch_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")



from polyflip.db.models import TradeHistory
from polyflip.execution.states import ACTIVE_POSITION_STATES


# ---------------------------------------------------------------------------
# Этап 6: управление тремя рубильниками LIVE-архитектуры
# ---------------------------------------------------------------------------

_BOOL_SWITCH_KEYS = {"LIVE_MIRROR_ENABLED", "LIVE_TRADING_ENABLED"}
_STR_SWITCH_KEYS = {"LIVE_RELEASE_MODE"}
_LIVE_RELEASE_MODE_VALUES = {"DISABLED", "MANUAL", "AUTO"}


class SwitchBoolRequest(BaseModel):
    enabled: bool


class SwitchReleaseModeRequest(BaseModel):
    mode: Literal["DISABLED", "MANUAL", "AUTO"]


async def _set_runtime_flag(db: AsyncSession, key: str, value: str) -> None:
    now = datetime.now(timezone.utc)
    existing = (await db.execute(
        select(RuntimeSettings).where(RuntimeSettings.key == key)
    )).scalar_one_or_none()
    if existing:
        existing.value = value
        existing.updated_at = now
        existing.updated_by = "api"
    else:
        db.add(RuntimeSettings(key=key, value=value, updated_at=now, updated_by="api"))
    await db.commit()


from polyflip.execution.live_mirror_worker import set_mirror_enabled


@router.put("/mirror-switch",
             summary="Включить / выключить LIVE_MIRROR_ENABLED (mirror-воркер)")
async def toggle_mirror_switch(
    payload: SwitchBoolRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Рубильник 1: управляет LIVE_MIRROR_ENABLED.
    true  — mirror-воркер создаёт LiveMirrorCandidate для FILLED PAPER OPEN.
    false — воркер спит, ни одного кандидата не создаёт.
    """
    try:
        await set_mirror_enabled(db, enabled=payload.enabled, updated_by="api")
        await db.commit()
        logger.info("mirror_switch_toggled", enabled=payload.enabled)
        return {"status": "ok", "LIVE_MIRROR_ENABLED": payload.enabled}
    except Exception:
        await db.rollback()
        raise


@router.put("/release-mode",
             summary="Установить LIVE_RELEASE_MODE (DISABLED | MANUAL | AUTO)")
async def set_release_mode(
    payload: SwitchReleaseModeRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Рубильник 2: управляет способом выпуска кандидатов.
    DISABLED — release_gate спит.
    MANUAL   — release_gate ожидает явного /release-candidate через API.
    AUTO     — release_gate автоматически выпускает NEW→ELIGIBLE кандидатов.
    """
    await _set_runtime_flag(db, "LIVE_RELEASE_MODE", payload.mode)
    logger.info("release_mode_set", mode=payload.mode)
    return {"status": "ok", "LIVE_RELEASE_MODE": payload.mode}


@router.get("/candidates",
             summary="Просмотр LiveMirrorCandidate")
async def get_mirror_candidates(
    state: Optional[str] = Query(None, description="NEW | ELIGIBLE | REJECTED | RELEASED"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Возвращает список mirror-кандидатов (для операторского мониторинга).
    """
    q = select(LiveMirrorCandidate).order_by(LiveMirrorCandidate.created_at.desc()).limit(limit)
    if state:
        q = q.where(LiveMirrorCandidate.state == state)
    rows = (await db.execute(q)).scalars().all()
    return [
        {
            "id": str(r.id),
            "source_paper_request_id": str(r.source_paper_request_id),
            "source_paper_trade_id": r.source_paper_trade_id,
            "target_mode": r.target_mode,
            "state": r.state,
            "signal_hash": r.signal_hash,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "released_at": r.released_at.isoformat() if r.released_at else None,
            "released_trade_id": r.released_trade_id,
            "released_request_id": str(r.released_request_id) if r.released_request_id else None,
            "rejection_reason": r.rejection_reason,
        }
        for r in rows
    ]


@router.post("/candidates/{candidate_id}/release",
              summary="Операторский выпуск: ELIGIBLE → release_gate переведёт кандидата в RELEASED")
async def manual_release_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    При LIVE_RELEASE_MODE=MANUAL оператор явно одобряет кандидата.
    release_gate при следующем цикле заберёт его и создаст LIVE-заявку.
    """
    from uuid import UUID
    try:
        cid = UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID")

    candidate = await db.get(LiveMirrorCandidate, cid)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.state not in ("NEW", "ELIGIBLE"):
        raise HTTPException(
            status_code=409,
            detail=f"Candidate is in state {candidate.state!r}, cannot manually release",
        )

    candidate.state = "ELIGIBLE"
    candidate_id_str = str(cid)
    await db.commit()
    logger.info("candidate_marked_eligible", candidate_id=candidate_id_str)
    return {"status": "ok", "candidate_id": candidate_id_str, "new_state": "ELIGIBLE"}


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
