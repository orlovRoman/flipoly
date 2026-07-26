import uuid
from typing import Literal
from uuid import UUID
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.config import ExecutionMode
from decimal import Decimal
from datetime import datetime, timezone

ACTIVE_CLOSE_STATES = (
    "READY",
    "CLAIMED",
    "SUBMITTING",
    "ACCEPTED",
    "UNKNOWN",
    "PARTIALLY_FILLED",
    "RECONCILING",
)

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
) -> UUID | None:
    request_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)
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
            trade_history_id=None,
            market_id=market_id,
            asset=asset,
            outcome_to_buy=outcome_to_buy,
            requested_shares=requested_shares,
            target_amount_usdc=Decimal(str(target_amount_usdc)),
            max_slippage_pct=2.0,
            limit_price=Decimal(str(limit_price)),
            max_spend_usdc=Decimal(str(target_amount_usdc)),
            ttl_seconds=60,
            state="READY",
            created_at=now_utc,
            updated_at=now_utc,
        )
        .on_conflict_do_nothing(
            index_elements=["market_id"],
            index_where=text(
                "intent = 'OPEN' AND state IN "
                "('READY','CLAIMED','SUBMITTING','ACCEPTED',"
                "'UNKNOWN','PARTIALLY_FILLED','RECONCILING')"
            ),
        )
        .returning(ExecutionRequest.id)
    )
    return (await db.execute(statement)).scalar_one_or_none()

async def enqueue_close_request(
    db: AsyncSession,
    *,
    trade_id: int,
    trigger_reason: Literal["STOP_LOSS", "TAKE_PROFIT", "MANUAL", "RECOVERY", "STRATEGY"],
    limit_price: float,
    requested_mode: ExecutionMode,
) -> UUID | None:
    
    # Nested transaction is used because we don't want to rollback the whole parent transaction
    # just in case something fails here, but here we actually rely on ON CONFLICT.
    # However, since this modifies position_status, we probably do want it committed alongside
    # parent transaction. We'll just execute it without explicit begin if it's already in transaction?
    # Let's just do it directly. We assume db session is already active or handled by caller.
    
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

    request_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)
    
    dialect_name = db.bind.dialect.name
    insert_func = sqlite_insert if dialect_name == 'sqlite' else pg_insert
    
    statement = (
        insert_func(ExecutionRequest)
        .values(
            id=request_id,
            idempotency_key=f"CLOSE:{trade.id}:v{trade.position_version}:a{trade.exit_attempts}",
            requested_mode=requested_mode.value,
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
            state="READY",
            created_at=now_utc,
            updated_at=now_utc,
        )
        .on_conflict_do_nothing(
            index_elements=["trade_history_id"],
            index_where=text(
                "intent = 'CLOSE' AND state IN "
                "('READY','CLAIMED','SUBMITTING','ACCEPTED',"
                "'UNKNOWN','PARTIALLY_FILLED','RECONCILING')"
            ),
        )
        .returning(ExecutionRequest.id)
    )

    created_id = (await db.execute(statement)).scalar_one_or_none()

    if created_id is not None:
        trade.position_status = "EXIT_REQUESTED"
        trade.exit_reason = trigger_reason
        return created_id

    return None
