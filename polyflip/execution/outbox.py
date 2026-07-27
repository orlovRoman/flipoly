import uuid
import structlog
from typing import Literal
from uuid import UUID
from dataclasses import dataclass
from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import TradeHistory, RuntimeSettings
from polyflip.db.execution_models import ExecutionRequest, ExposureReservation, ExecutionEvent
from polyflip.execution.config import ExecutionMode
from polyflip.execution.states import (
    TERMINAL_REQUEST_STATES, 
    FAILURE_TERMINAL_STATES,
    ACTIVE_REQUEST_STATES
)
from polyflip.execution.risk_checks import check_risk_limits
from decimal import Decimal
from datetime import datetime, timezone, timedelta

logger = structlog.get_logger(__name__)

@dataclass(frozen=True)
class EnqueueResult:
    request_id: UUID
    created: bool

async def enqueue_open_request(
    db: AsyncSession,
    *,
    trade_id: int,
    market_id: str,
    asset: str,
    outcome_to_buy: str,
    target_amount_usdc: float,
    limit_price: float,
    requested_mode: ExecutionMode,
) -> EnqueueResult | None:
    
    existing_id = (
        await db.execute(
            select(ExecutionRequest.id)
            .where(ExecutionRequest.requested_mode == requested_mode.value)
            .where(ExecutionRequest.market_id == market_id)
            .where(ExecutionRequest.intent == "OPEN")
            .where(ExecutionRequest.state.in_(ACTIVE_REQUEST_STATES))
        )
    ).scalar_one_or_none()
    
    if existing_id is not None:
        return EnqueueResult(request_id=existing_id, created=False)

    request_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)
    
    risk_error = await check_risk_limits(
        db,
        "OPEN",
        Decimal(str(target_amount_usdc)),
        requested_mode.value,
        trade_history_id=trade_id,
    )
    if risk_error:
        logger.warning("risk_limit_breached", reason=risk_error, market_id=market_id)
        return None

    # --- CONFIRM_THRESHOLD_USDC ---
    # Если сумма превышает порог — ставим AWAITING_APPROVAL, не READY
    threshold_set = (
        await db.execute(
            select(RuntimeSettings).where(
                RuntimeSettings.key == "CONFIRM_THRESHOLD_USDC"
            )
        )
    ).scalar_one_or_none()
    initial_state = "READY"
    if threshold_set:
        try:
            threshold = Decimal(threshold_set.value)
            if Decimal(str(target_amount_usdc)) > threshold:
                initial_state = "AWAITING_APPROVAL"
        except (ValueError, TypeError):
            pass  # некорректное значение — считаем порог неустановленным
        
    dialect_name = db.bind.dialect.name
    insert_func = sqlite_insert if dialect_name == 'sqlite' else pg_insert
    
    requested_shares = Decimal(str(target_amount_usdc)) / Decimal(str(limit_price)) if limit_price > 0 else Decimal("0")
    
    statement = (
        insert_func(ExecutionRequest)
        .values(
            id=request_id,
            idempotency_key=f"OPEN:{trade_id}",
            requested_mode=requested_mode.value,
            intent="OPEN",
            trigger_reason="STRATEGY",
            trade_history_id=trade_id,
            market_id=market_id,
            asset=asset,
            outcome_to_buy=outcome_to_buy,
            requested_shares=requested_shares,
            target_amount_usdc=Decimal(str(target_amount_usdc)),
            max_slippage_pct=2.0,
            limit_price=Decimal(str(limit_price)),
            max_spend_usdc=Decimal(str(target_amount_usdc)),
            ttl_seconds=60,
            expires_at=now_utc + timedelta(seconds=60),
            state=initial_state,
            created_at=now_utc,
            updated_at=now_utc,
        )
        .on_conflict_do_nothing(
            index_elements=["requested_mode", "market_id"],
            index_where=ExecutionRequest.ACTIVE_OPEN_PREDICATE,
        )
        .returning(ExecutionRequest.id)
    )
    created_id = (await db.execute(statement)).scalar_one_or_none()
    if created_id is not None:
        return EnqueueResult(request_id=created_id, created=True)
        
    existing_id = (
        await db.execute(
            select(ExecutionRequest.id)
            .where(ExecutionRequest.requested_mode == requested_mode.value)
            .where(ExecutionRequest.market_id == market_id)
            .where(ExecutionRequest.intent == "OPEN")
            .where(ExecutionRequest.state.in_(ACTIVE_REQUEST_STATES))
        )
    ).scalar_one_or_none()
    
    if existing_id is not None:
        return EnqueueResult(request_id=existing_id, created=False)
    return None

async def enqueue_close_request(
    db: AsyncSession,
    *,
    trade_id: int,
    trigger_reason: Literal["STOP_LOSS", "TAKE_PROFIT", "MANUAL", "RECOVERY", "STRATEGY"],
    limit_price: float,
) -> EnqueueResult | None:
    
    trade = (
        await db.execute(
            select(TradeHistory)
            .where(TradeHistory.id == trade_id)
            .with_for_update()
        )
    ).scalar_one()

    if trade.position_status == "CLOSED":
        return None
    
    if not trade.remaining_shares or trade.remaining_shares <= 0:
        return None

    existing_id = (
        await db.execute(
            select(ExecutionRequest.id)
            .where(ExecutionRequest.trade_history_id == trade.id)
            .where(ExecutionRequest.intent == "CLOSE")
            .where(ExecutionRequest.state.in_(ACTIVE_REQUEST_STATES))
        )
    ).scalar_one_or_none()

    if existing_id is not None:
        return EnqueueResult(request_id=existing_id, created=False)

    request_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)

    # Атомарно фиксируем текущую версию позиции для idempotency_key,
    # а затем увеличиваем её, чтобы следующий CLOSE получил новый ключ.
    # Оба изменения (ExecutionRequest + trade.position_version) находятся
    # в одной транзакции под SELECT ... FOR UPDATE.
    version_snapshot = trade.position_version or 0

    dialect_name = db.bind.dialect.name
    insert_func = sqlite_insert if dialect_name == "sqlite" else pg_insert

    statement = (
        insert_func(ExecutionRequest)
        .values(
            id=request_id,
            idempotency_key=f"CLOSE:{trade.id}:v{version_snapshot}",
            requested_mode=trade.mode,  # режим берём из исходной сделки
            intent="CLOSE",
            trigger_reason=trigger_reason,
            trade_history_id=trade.id,
            market_id=trade.market_id,
            asset=trade.asset,
            outcome_to_buy=trade.outcome_bought,
            requested_shares=Decimal(str(trade.remaining_shares)),
            target_amount_usdc=Decimal(str(trade.remaining_shares)) * Decimal(str(limit_price)),
            max_slippage_pct=2.0,
            limit_price=Decimal(str(limit_price)),
            ttl_seconds=60,
            expires_at=now_utc + timedelta(seconds=60),
            state="READY",
            created_at=now_utc,
            updated_at=now_utc,
        )
        .on_conflict_do_nothing(
            index_elements=["trade_history_id"],
            index_where=ExecutionRequest.ACTIVE_CLOSE_PREDICATE,
        )
        .returning(ExecutionRequest.id)
    )


    created_id = (await db.execute(statement)).scalar_one_or_none()
    if created_id is not None:
        trade.position_status = "EXIT_REQUESTED"
        trade.exit_reason = trigger_reason
        # Атомарно увеличиваем версию позиции — следующий CLOSE получит новый ключ
        trade.position_version = version_snapshot + 1
        return EnqueueResult(request_id=created_id, created=True)


    existing_id = (
        await db.execute(
            select(ExecutionRequest.id)
            .where(ExecutionRequest.trade_history_id == trade.id)
            .where(ExecutionRequest.intent == "CLOSE")
            .where(ExecutionRequest.state.in_(ACTIVE_REQUEST_STATES))
        )
    ).scalar_one_or_none()

    if existing_id is not None:
        return EnqueueResult(request_id=existing_id, created=False)

    return None

async def finalize_request(
    session: AsyncSession,
    req: ExecutionRequest,
    *,
    state: str,
    error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    req.state = state
    req.error_reason = error
    req.updated_at = now
    
    if req.intent == "OPEN" and state in TERMINAL_REQUEST_STATES:
        await session.execute(
            update(ExposureReservation)
            .where(
                ExposureReservation.request_id == req.id,
                ExposureReservation.released_at.is_(None),
            )
            .values(released_at=now)
        )
        
    trade = await session.get(
        TradeHistory, req.trade_history_id, with_for_update=True,
    )
    if not trade:
        return
        
    if state in FAILURE_TERMINAL_STATES and req.filled_shares == 0:
        if req.intent == "OPEN":
            trade.status = "FAILED"
            trade.position_status = "ENTRY_FAILED"
            trade.error_msg = error
        else:
            trade.position_status = (
                "PARTIALLY_CLOSED" 
                if Decimal(trade.remaining_shares or 0) < Decimal(trade.entry_filled_shares or 0) 
                else "OPEN"
            )
            trade.last_exit_error = error
            trade.exit_attempts = (trade.exit_attempts or 0) + 1
            
    session.add(ExecutionEvent(
        level="ERROR" if error else "INFO",
        event_type=f"REQUEST_{state}",
        message=error or f"Request transitioned to {state}",
        source="execution_worker",
        request_id=req.id,
        trade_history_id=req.trade_history_id,
    ))
