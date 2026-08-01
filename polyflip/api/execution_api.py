from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from sqlalchemy import select, and_, or_, desc, func
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import uuid
import structlog

from polyflip.db.connection import get_db_session
from polyflip.api.auth import verify_api_key
from polyflip.db.models import RuntimeSettings
from polyflip.db.execution_models import (
    ExecutionWorkerStatus,
    ExecutionRequest,
    ExecutionEvent,
    ExecutionAttempt,
    LiveMirrorCandidate,
)
from polyflip.execution.config import ExecutionSettings

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/execution", tags=["Execution"], dependencies=[Depends(verify_api_key)]
)


class KillSwitchRequest(BaseModel):
    enabled: bool


@router.get("/status")
async def get_live_trading_status(db: AsyncSession = Depends(get_db_session)):
    """
    Returns live trading status, mirror status, and worker status for PAPER, SHADOW, and LIVE modes.
    """
    settings = ExecutionSettings()

    async def get_worker_dict(mode_name: str):
        ws = (
            await db.execute(
                select(ExecutionWorkerStatus)
                .where(ExecutionWorkerStatus.execution_mode == mode_name)
                .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if not ws:
            return None
        return {
            "worker_id": ws.worker_id,
            "execution_mode": ws.execution_mode,
            "heartbeat_at": ws.heartbeat_at.isoformat() if ws.heartbeat_at else None,
            "gateway_ready": ws.gateway_ready,
            "credentials_loaded": ws.credentials_loaded,
            "wallet_address": ws.wallet_address,
            "balance_usdc": (
                float(ws.balance_usdc) if ws.balance_usdc is not None else None
            ),
            "collateral_allowance_ready": ws.collateral_allowance_ready,
            "conditional_allowance_ready": ws.conditional_allowance_ready,
            "last_error_code": ws.last_error_code,
            "last_error_message": ws.last_error_message,
        }

    paper_worker = await get_worker_dict("PAPER")
    shadow_worker = await get_worker_dict("SHADOW")
    live_worker = await get_worker_dict("LIVE")

    async def _flag(key: str) -> bool:
        row = (
            await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
        ).scalar_one_or_none()
        return row is not None and row.value.lower() == "true"

    async def _flag_str(key: str, default: str) -> str:
        row = (
            await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
        ).scalar_one_or_none()
        return row.value if row else default

    live_mirror_enabled = await _flag("LIVE_MIRROR_ENABLED")
    live_release_mode = await _flag_str("LIVE_RELEASE_MODE", "DISABLED")
    live_trading_enabled = await _flag("LIVE_TRADING_ENABLED")

    # Количество кандидатов по состояниям для режима LIVE
    candidate_counts = {}
    for state in ("NEW", "ELIGIBLE", "REJECTED", "RELEASED"):
        cnt = await db.scalar(
            select(func.count())
            .select_from(LiveMirrorCandidate)
            .where(
                LiveMirrorCandidate.state == state,
                LiveMirrorCandidate.target_mode == "LIVE",
            )
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
async def toggle_kill_switch(
    payload: KillSwitchRequest, db: AsyncSession = Depends(get_db_session)
):
    """
    Управляет экстренным выключением LIVE-торговли.
    Включение LIVE выполняется исключительно через активацию LIVE-сессии.
    """
    if payload.enabled:
        raise HTTPException(
            status_code=409,
            detail="Включение LIVE выполняется только через активацию LIVE-сессии",
        )

    key = "LIVE_TRADING_ENABLED"

    # Защищаем чтение/запись блокировкой FOR UPDATE
    stmt = select(RuntimeSettings).where(RuntimeSettings.key == key)
    bind = db.bind
    if bind and bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update()

    existing = (await db.execute(stmt)).scalar_one_or_none()

    if payload.enabled:
        worker_status = (
            await db.execute(
                select(ExecutionWorkerStatus)
                .where(ExecutionWorkerStatus.execution_mode == "LIVE")
                .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if not worker_status:
            raise HTTPException(
                status_code=409,
                detail="Cannot enable LIVE trading: LIVE worker is not running (no status in DB)",
            )

        now = datetime.now(timezone.utc)
        hb_at = worker_status.heartbeat_at
        if hb_at and hb_at.tzinfo is None:
            hb_at = hb_at.replace(tzinfo=timezone.utc)

        if not hb_at or hb_at < now - timedelta(seconds=30):
            raise HTTPException(
                status_code=409,
                detail="Cannot enable LIVE trading: LIVE worker heartbeat is older than 30 seconds",
            )

        if not worker_status.gateway_ready:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot enable LIVE trading: LIVE gateway is not ready ({worker_status.last_error_message})",
            )

        if float(worker_status.balance_usdc or 0) < 5:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot enable LIVE trading: Insufficient USDC balance (Minimum $5, current {float(worker_status.balance_usdc or 0)})",
            )

        if not worker_status.collateral_allowance_ready:
            raise HTTPException(
                status_code=409,
                detail="Cannot enable LIVE trading: Collateral allowance is not ready",
            )

    value = "true" if payload.enabled else "false"

    try:
        now = datetime.now(timezone.utc)
        if existing:
            existing.value = value
            existing.updated_at = now
            existing.updated_by = "api"
        else:
            db.add(
                RuntimeSettings(key=key, value=value, updated_at=now, updated_by="api")
            )

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
    existing = (
        await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
    ).scalar_one_or_none()
    if existing:
        existing.value = value
        existing.updated_at = now
        existing.updated_by = "api"
    else:
        db.add(RuntimeSettings(key=key, value=value, updated_at=now, updated_by="api"))
    await db.commit()


from polyflip.execution.live_mirror_worker import set_mirror_enabled


@router.put(
    "/mirror-switch", summary="Включить / выключить LIVE_MIRROR_ENABLED (mirror-воркер)"
)
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


@router.put(
    "/release-mode", summary="Установить LIVE_RELEASE_MODE (DISABLED | MANUAL | AUTO)"
)
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


@router.get("/candidates", summary="Просмотр LiveMirrorCandidate")
async def get_mirror_candidates(
    state: Optional[str] = Query(
        None, description="NEW | ELIGIBLE | REJECTED | RELEASED"
    ),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Возвращает список mirror-кандидатов (для операторского мониторинга).
    """
    q = (
        select(LiveMirrorCandidate)
        .order_by(LiveMirrorCandidate.created_at.desc())
        .limit(limit)
    )
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
            "released_request_id": (
                str(r.released_request_id) if r.released_request_id else None
            ),
            "rejection_reason": r.rejection_reason,
        }
        for r in rows
    ]


@router.post(
    "/candidates/{candidate_id}/release",
    summary="Операторский выпуск: ELIGIBLE → release_gate переведёт кандидата в RELEASED",
)
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
    mode: Optional[str] = Query(
        None, description="Фильтр по режиму: PAPER, SHADOW, LIVE"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    settings = ExecutionSettings()
    effective_mode = mode or settings.execution_mode.value

    stmt = (
        select(TradeHistory)
        .where(
            TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
            TradeHistory.mode == effective_mode,
        )
        .order_by(TradeHistory.created_at.desc())
        .limit(100)
    )

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
    mode: Optional[str] = Query(
        None, description="Фильтр по режиму: PAPER, SHADOW, LIVE"
    ),
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

    from polyflip.execution.serializers import serialize_execution_requests
    request_dtos = await serialize_execution_requests(db, requests)

    return request_dtos


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

    try:
        req_uuid = uuid.UUID(request_id) if isinstance(request_id, str) else request_id
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request UUID format")

    req = (
        await db.execute(
            select(ExecutionRequest).where(ExecutionRequest.id == req_uuid).with_for_update()
        )
    ).scalar_one_or_none()

    if req is None:
        raise HTTPException(status_code=404, detail="Request not found")

    if req.state == "MANUAL_REVIEW_FAILED":
        return {
            "request_id": str(req.id),
            "new_state": req.state,
            "already_resolved": True,
        }

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
        from polyflip.execution.manual_review_service import evaluate_no_fill_eligibility
        from polyflip.execution.outbox import finalize_request

        eligibility = await evaluate_no_fill_eligibility(db, req)

        if not eligibility.allowed:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Заявку нельзя завершить без исполнения",
                    "blockers": list(eligibility.blockers),
                },
            )

        await finalize_request(
            db,
            req,
            state="MANUAL_REVIEW_FAILED",
            error=(
                f"Confirmed no fill by {body.operator}: "
                f"{body.note or 'provider evidence absent'}"
            ),
        )

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
            created_at=now,
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


class BatchNoFillRequest(BaseModel):
    request_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)
    operator: str
    note: str = ""


@router.post("/requests/resolve-no-fill-batch")
async def resolve_no_fill_batch(
    payload: BatchNoFillRequest,
    db: AsyncSession = Depends(get_db_session)
):
    from polyflip.execution.manual_review_service import evaluate_no_fill_eligibility_batch
    from polyflip.execution.outbox import finalize_request
    from polyflip.db.execution_models import ExecutionEvent
    from datetime import datetime, timezone

    resolved = []
    skipped = []
    now = datetime.now(timezone.utc)

    requests = (
        await db.execute(
            select(ExecutionRequest)
            .where(ExecutionRequest.id.in_(payload.request_ids))
            .with_for_update()
        )
    ).scalars().all()

    eligibility_map = await evaluate_no_fill_eligibility_batch(db, requests)

    for req in requests:
        eligibility = eligibility_map.get(req.id)

        if not eligibility or not eligibility.allowed:
            skipped.append({
                "request_id": str(req.id),
                "blockers": list(eligibility.blockers) if eligibility else ["Unknown"],
            })
            continue

        await finalize_request(
            db,
            req,
            state="MANUAL_REVIEW_FAILED",
            error=f"Batch no-fill confirmation by {payload.operator}",
        )

        db.add(
            ExecutionEvent(
                level="INFO",
                event_type="MANUAL_REVIEW_BATCH_NO_FILL",
                message=f"Operator {payload.operator!r} resolved MANUAL_REVIEW_REQUIRED via batch. Note: {payload.note or '—'}",
                source="execution_api",
                request_id=req.id,
                trade_history_id=req.trade_history_id,
                created_at=now,
                payload={
                    "operator": payload.operator,
                    "action": "MARK_FAILED_NO_FILL",
                    "batch": True,
                    "note": payload.note,
                },
            )
        )

        resolved.append(str(req.id))

    await db.commit()

    return {
        "resolved": resolved,
        "skipped": skipped,
    }



# ── LIVE Sessions, Readiness & Control Endpoints ──────────────────────────────

from decimal import Decimal
from pydantic import model_validator
from polyflip.db.execution_models import LiveTradingSession
from polyflip.execution.outbox import enqueue_close_request, EnqueueDisposition
from polyflip.execution.live_session_service import (
    get_active_session_for_update,
    count_session_positions,
    get_session_budget_snapshot,
    evaluate_live_readiness,
    serialize_live_session_dto,
)


class CreateLiveSessionRequest(BaseModel):
    budget_usdc: Decimal
    max_single_order_usdc: Decimal
    max_open_positions: int
    max_total_exposure_usdc: Decimal

    @model_validator(mode="after")
    def validate_limits(self):
        from polyflip.execution.config import LIVE_MIN_GROSS_BUY_USDC

        if self.budget_usdc <= 0:
            raise ValueError("Бюджет должен быть больше нуля")

        if self.max_single_order_usdc < LIVE_MIN_GROSS_BUY_USDC:
            raise ValueError(
                "Максимальная LIVE-ставка должна быть "
                "не меньше 1.10 USDC"
            )

        if self.max_single_order_usdc > self.budget_usdc:
            raise ValueError(
                "Максимальная ставка не может превышать бюджет сессии"
            )

        if self.max_total_exposure_usdc > self.budget_usdc:
            raise ValueError(
                "Максимальная экспозиция не может превышать бюджет"
            )

        if not 1 <= self.max_open_positions <= 100:
            raise ValueError("Некорректный лимит открытых позиций")

        return self


@router.post("/live/sessions")
async def create_live_session(
    payload: CreateLiveSessionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    active_or_draft = await get_active_session_for_update(db)

    if active_or_draft:
        raise HTTPException(
            status_code=409,
            detail=f"Управляемая LIVE-сессия уже существует в статусе {active_or_draft.status}",
        )

    session = LiveTradingSession(
        status="DRAFT",
        budget_usdc=payload.budget_usdc,
        reserved_usdc=Decimal("0"),
        filled_usdc=Decimal("0"),
        max_single_order_usdc=payload.max_single_order_usdc,
        max_open_positions=payload.max_open_positions,
        max_total_exposure_usdc=payload.max_total_exposure_usdc,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    budget_snap = await get_session_budget_snapshot(db, session)
    return serialize_live_session_dto(session, budget_snap)


@router.post("/live/sessions/{session_id}/readiness")
async def check_live_session_readiness(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id UUID format")

    session_obj = (
        await db.execute(
            select(LiveTradingSession).where(LiveTradingSession.id == sess_uuid)
        )
    ).scalar_one_or_none()

    if not session_obj:
        raise HTTPException(status_code=404, detail="LiveTradingSession not found")

    readiness = await evaluate_live_readiness(db, session_obj)

    if readiness.ready and session_obj.status == "DRAFT":
        session_obj.status = "READY"
        await db.commit()
        await db.refresh(session_obj)

    budget_snap = await get_session_budget_snapshot(db, session_obj)

    return {
        "ready": readiness.ready,
        "session": serialize_live_session_dto(session_obj, budget_snap),
        "checks": readiness.checks,
        "errors": readiness.errors,
        "warnings": readiness.warnings,
    }


@router.post("/live/sessions/{session_id}/activate")
async def activate_live_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id UUID format")

    session_obj = (
        await db.execute(
            select(LiveTradingSession)
            .where(LiveTradingSession.id == sess_uuid)
            .with_for_update()
        )
    ).scalar_one_or_none()

    if not session_obj:
        raise HTTPException(status_code=404, detail="LiveTradingSession not found")

    if session_obj.status not in {"DRAFT", "READY"}:
        raise HTTPException(
            status_code=409,
            detail=(
                "Активация разрешена только для сессии "
                f"DRAFT/READY, текущий статус: {session_obj.status}"
            ),
        )

    # Повторный прогон готовности непосредственно перед активацией
    readiness = await evaluate_live_readiness(db, session_obj)
    if not readiness.ready:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Сессия не прошла проверку готовности",
                "errors": readiness.errors,
            },
        )

    # Автоматически атомарно включаем все тумблера и метку старта в БД
    now = datetime.now(timezone.utc)
    for key, val in [
        ("LIVE_MIRROR_ENABLED", "true"),
        ("LIVE_RELEASE_MODE", "AUTO"),
        ("LIVE_TRADING_ENABLED", "true"),
        ("LIVE_MIRROR_STARTED_AT", now.isoformat()),
    ]:
        row = (
            await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
        ).scalar_one_or_none()
        if row:
            row.value = val
            row.updated_at = now
            row.updated_by = "session_activate"
        else:
            db.add(
                RuntimeSettings(
                    key=key, value=val, updated_at=now, updated_by="session_activate"
                )
            )

    session_obj.status = "ACTIVE"
    session_obj.started_at = now
    await db.commit()
    await db.refresh(session_obj)

    budget_snap = await get_session_budget_snapshot(db, session_obj)
    return serialize_live_session_dto(session_obj, budget_snap)


@router.post("/live/sessions/{session_id}/stop")
async def stop_live_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id UUID format")

    session_obj = (
        await db.execute(
            select(LiveTradingSession)
            .where(LiveTradingSession.id == sess_uuid)
            .with_for_update()
        )
    ).scalar_one_or_none()

    if not session_obj:
        raise HTTPException(status_code=404, detail="LiveTradingSession not found")

    now = datetime.now(timezone.utc)

    # Выключаем переключатели и сбрасываем метку в БД
    for key, val in [
        ("LIVE_MIRROR_ENABLED", "false"),
        ("LIVE_RELEASE_MODE", "DISABLED"),
        ("LIVE_TRADING_ENABLED", "false"),
        ("LIVE_MIRROR_STARTED_AT", ""),
    ]:
        row = (
            await db.execute(select(RuntimeSettings).where(RuntimeSettings.key == key))
        ).scalar_one_or_none()
        if row:
            row.value = val
            row.updated_at = now
            row.updated_by = "user_stop"
        else:
            db.add(
                RuntimeSettings(
                    key=key, value=val, updated_at=now, updated_by="user_stop"
                )
            )

    session_obj.status = "STOPPED"
    session_obj.stopped_at = now
    session_obj.stop_reason = "USER_STOP"
    await db.commit()
    await db.refresh(session_obj)

    budget_snap = await get_session_budget_snapshot(db, session_obj)
    return serialize_live_session_dto(session_obj, budget_snap)


@router.post("/live/sessions/{session_id}/finish")
async def finish_live_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id UUID format")

    session_obj = (
        await db.execute(
            select(LiveTradingSession)
            .where(LiveTradingSession.id == sess_uuid)
            .with_for_update()
        )
    ).scalar_one_or_none()

    if not session_obj:
        raise HTTPException(status_code=404, detail="LiveTradingSession not found")

    open_pos = await count_session_positions(db, session_obj.id)
    if open_pos > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Нельзя завершить сессию: осталось {open_pos} открытых/закрывающихся позиций",
        )

    active_cnt = (
        await db.scalar(
            select(func.count(ExecutionRequest.id)).where(
                ExecutionRequest.live_session_id == session_obj.id,
                ExecutionRequest.requested_mode == "LIVE",
                ExecutionRequest.state.in_(
                    [
                        "AWAITING_APPROVAL",
                        "READY",
                        "CLAIMED",
                        "SUBMITTING",
                        "ACCEPTED",
                        "UNKNOWN",
                        "PARTIALLY_FILLED",
                        "RECONCILING",
                        "MANUAL_REVIEW_REQUIRED",
                    ]
                ),
            )
        )
    ) or 0

    if active_cnt > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Нельзя завершить сессию: {active_cnt} активных заявок в обработке",
        )

    session_obj.status = "STOPPED"
    session_obj.stopped_at = datetime.now(timezone.utc)
    session_obj.stop_reason = "COMPLETED"
    await db.commit()
    await db.refresh(session_obj)

    budget_snap = await get_session_budget_snapshot(db, session_obj)
    return serialize_live_session_dto(session_obj, budget_snap)


@router.post("/positions/{trade_id}/close")
async def close_live_position(
    trade_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    trade = (
        await db.execute(
            select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update()
        )
    ).scalar_one_or_none()

    if not trade or trade.mode != "LIVE":
        raise HTTPException(status_code=404, detail="LIVE-позиция не найдена")

    if trade.position_status in ("CLOSED", "REDEEMED"):
        raise HTTPException(status_code=409, detail="Позиция уже закрыта")

    if not trade.remaining_shares or trade.remaining_shares <= 0:
        raise HTTPException(
            status_code=409, detail="У позиции нет токенов для закрытия"
        )

    limit_price = float(trade.executed_price or 0.5)

    result = await enqueue_close_request(
        db,
        trade_id=trade.id,
        trigger_reason="MANUAL",
        limit_price=limit_price,
    )

    if result.disposition == EnqueueDisposition.CREATED:
        await db.commit()
        return {
            "status": "queued",
            "disposition": "CREATED",
            "request_id": str(result.request_id),
        }

    if result.disposition == EnqueueDisposition.DUPLICATE:
        return {
            "status": "already_queued",
            "disposition": "DUPLICATE",
            "request_id": str(result.request_id),
        }

    raise HTTPException(
        status_code=409, detail=f"Нельзя создать заявку закрытия: {result.reason}"
    )


@router.post("/live/sessions/{session_id}/close-all")
async def close_all_session_positions(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        sess_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id UUID format")

    open_trades = (
        (
            await db.execute(
                select(TradeHistory).where(
                    TradeHistory.mode == "LIVE",
                    TradeHistory.position_status.in_(["OPEN", "PARTIALLY_CLOSED"]),
                    TradeHistory.live_session_id == sess_uuid,
                )
            )
        )
        .scalars()
        .all()
    )

    results = []
    for trade in open_trades:
        limit_price = float(trade.executed_price or 0.5)
        res = await enqueue_close_request(
            db,
            trade_id=trade.id,
            trigger_reason="MANUAL",
            limit_price=limit_price,
        )
        results.append(
            {
                "trade_id": trade.id,
                "disposition": str(res.disposition),
                "request_id": str(res.request_id) if res.request_id else None,
                "reason": res.reason,
            }
        )

    await db.commit()
    return {
        "status": "completed",
        "total_positions": len(open_trades),
        "results": results,
    }


@router.get("/live/dashboard")
async def get_live_dashboard(db: AsyncSession = Depends(get_db_session)):
    active_session = (
        await db.execute(
            select(LiveTradingSession)
            .where(
                LiveTradingSession.status.in_(
                    ["DRAFT", "READY", "ACTIVE", "STOPPED"]
                )
            )
            .order_by(LiveTradingSession.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    worker_status = (
        await db.execute(
            select(ExecutionWorkerStatus)
            .where(ExecutionWorkerStatus.execution_mode == "LIVE")
            .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    candidates = (
        (
            await db.execute(
                select(LiveMirrorCandidate)
                .where(LiveMirrorCandidate.target_mode == "LIVE")
                .order_by(LiveMirrorCandidate.created_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )

    active_positions = (
        (
            await db.execute(
                select(TradeHistory)
                .where(
                    TradeHistory.mode == "LIVE",
                    TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
                    TradeHistory.remaining_shares > 0,
                    TradeHistory.entry_filled_shares > 0,
                )
                .order_by(TradeHistory.created_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )

    failed_entries = (
        (
            await db.execute(
                select(TradeHistory)
                .where(
                    TradeHistory.mode == "LIVE",
                    TradeHistory.position_status == "ENTRY_FAILED",
                )
                .order_by(TradeHistory.created_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )

    requests = (
        (
            await db.execute(
                select(ExecutionRequest)
                .where(ExecutionRequest.requested_mode == "LIVE")
                .order_by(ExecutionRequest.created_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )

    budget_snap = None
    if active_session:
        budget_snap = await get_session_budget_snapshot(db, active_session)

    from polyflip.execution.serializers import serialize_execution_requests
    request_dtos = await serialize_execution_requests(db, requests)

    def serialize_positions(pos_list):
        return [
            {
                "id": p.id,
                "asset": p.asset,
                "market_id": p.market_id,
                "outcome_bought": p.outcome_bought,
                "amount_usdc": p.amount_usdc,
                "executed_price": p.executed_price,
                "pnl": p.pnl,
                "position_status": p.position_status,
                "remaining_shares": (
                    float(p.remaining_shares) if p.remaining_shares else 0.0
                ),
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in pos_list
        ]


    readiness = None
    if active_session:
        readiness = await evaluate_live_readiness(db, active_session)

    status = active_session.status if active_session else None
    active_pos_count = len([p for p in active_positions if p.position_status not in ("CLOSED", "RESOLVED", "ENTRY_FAILED")])

    readiness_ready = bool(readiness and readiness.ready)
    session_actions = {
        "check_readiness": status in {"DRAFT", "READY", "STOPPED"},
        "activate": (
            readiness_ready
            and status in {"DRAFT", "READY"}
        ),
        "stop": status == "ACTIVE",
        "close_all": active_pos_count > 0,
        "finish": status in {"DRAFT", "READY", "STOPPED"},
    } if status else {
        "check_readiness": False,
        "activate": False,
        "stop": False,
        "close_all": False,
        "finish": False,
    }

    return {
        "available_actions": session_actions,
        "readiness": {
            "ready": readiness.ready,
            "checks": readiness.checks,
            "errors": readiness.errors,
            "warnings": readiness.warnings,
        } if readiness else None,
        "session": (
            serialize_live_session_dto(active_session, budget_snap)
            if active_session
            else None
        ),
        "worker": {
            "worker_id": worker_status.worker_id if worker_status else None,
            "heartbeat_at": (
                worker_status.heartbeat_at.isoformat()
                if worker_status and worker_status.heartbeat_at
                else None
            ),
            "gateway_ready": worker_status.gateway_ready if worker_status else False,
            "credentials_loaded": (
                worker_status.credentials_loaded if worker_status else False
            ),
            "wallet_address": (worker_status.wallet_address if worker_status else None),
            "balance_usdc": (
                float(worker_status.balance_usdc)
                if worker_status and worker_status.balance_usdc is not None
                else 0.0
            ),
            "collateral_allowance_ready": (
                worker_status.collateral_allowance_ready if worker_status else False
            ),
            "conditional_allowance_ready": (
                worker_status.conditional_allowance_ready if worker_status else False
            ),
        },
        "candidates": [
            {
                "id": str(c.id),
                "source_paper_request_id": str(c.source_paper_request_id),
                "state": c.state,
                "rejection_reason": c.rejection_reason,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in candidates
        ],
        "positions": serialize_positions(active_positions),
        "failed_entries": serialize_positions(failed_entries),
        "requests": request_dtos,
    }


@router.post("/requests/{request_id}/reconcile")
async def reconcile_request(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    req = await db.scalar(
        select(ExecutionRequest)
        .where(ExecutionRequest.id == request_id)
        .with_for_update()
    )

    if req is None:
        raise HTTPException(404, "Заявка не найдена")

    if req.requested_mode != "LIVE":
        raise HTTPException(409, "Сверка разрешена только для LIVE-заявок")

    if req.state == "RECONCILING":
        return {
            "request_id": str(req.id),
            "state": "RECONCILING",
            "idempotent": True,
        }

    allowed_states = {
        "SUBMITTING",
        "ACCEPTED",
        "UNKNOWN",
        "PARTIALLY_FILLED",
        "RECONCILING",
        "MANUAL_REVIEW_REQUIRED",
    }

    if req.state not in allowed_states:
        raise HTTPException(
            409,
            f"Заявку в статусе {req.state} нельзя переводить в RECONCILING",
        )

    provider_evidence = await db.scalar(
        select(func.count(ExecutionAttempt.id)).where(
            ExecutionAttempt.request_id == request_id,
            ExecutionAttempt.provider_order_id.is_not(None),
        )
    )

    if not provider_evidence:
        raise HTTPException(
            422,
            "Нет provider_order_id — сверка с Polymarket невозможна",
        )

    req.state = "RECONCILING"
    req.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {"request_id": str(req.id), "state": "RECONCILING"}
