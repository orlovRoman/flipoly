"""
polyflip/trading/ct_reservation.py

Atomic reservation and idempotency management for CT Outsider decisions:
- Enforces unique key 'CT:{spec_id}:{market_id}' at the database level.
- Prevents concurrent race conditions via atomic INSERT ON CONFLICT DO NOTHING.
- Guarantees that completed/settled markets are not re-traded.
- Preserves the immutability of the initial decision while tracking repeat attempts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.models import CTDecisionReservation

logger = structlog.get_logger(__name__)


async def _get_dialect(db_session: AsyncSession) -> str:
    try:
        conn = await db_session.connection()
        return conn.dialect.name
    except Exception:
        bind = getattr(db_session, "bind", None)
        if bind and hasattr(bind, "dialect"):
            return bind.dialect.name
        return "sqlite"


async def get_ct_decision_reservation(
    db_session: AsyncSession,
    key: str,
) -> Optional[CTDecisionReservation]:
    """Look up an existing CT decision reservation by its canonical key."""
    stmt = select(CTDecisionReservation).where(CTDecisionReservation.key == str(key))
    result = await db_session.execute(stmt)
    return result.scalar_one_or_none()


async def reserve_ct_decision(
    db_session: AsyncSession,
    *,
    key: str,
    market_id: str,
    spec_id: str,
    action: str,
    decision_at: datetime,
    side: Optional[str] = None,
    limit_price: Optional[float] = None,
    budget_usdc: Optional[float] = None,
    reason: str = "",
    trade_history_id: Optional[int] = None,
    decision_details: Optional[dict[str, Any]] = None,
    increment_on_conflict: bool = True,
) -> tuple[bool, CTDecisionReservation]:
    """
    Atomically reserves a CT decision on the database level.

    Returns:
    - (True, reservation): The reservation was newly inserted (first execution).
    - (False, reservation): A reservation already exists (duplicate/repeat).
    """
    import json
    from sqlalchemy import update

    dialect_name = await _get_dialect(db_session)
    insert_func = sqlite_insert if dialect_name == "sqlite" else pg_insert

    now_utc = datetime.now(timezone.utc)
    dec_at = (
        decision_at
        if decision_at.tzinfo is not None
        else decision_at.replace(tzinfo=timezone.utc)
    )

    clean_details = None
    if decision_details is not None:
        try:
            clean_details = json.loads(json.dumps(decision_details, default=str))
        except Exception:
            clean_details = None

    stmt = (
        insert_func(CTDecisionReservation)
        .values(
            key=str(key),
            market_id=str(market_id),
            spec_id=str(spec_id),
            action=str(action),
            decision_at=dec_at,
            side=side,
            limit_price=limit_price,
            budget_usdc=budget_usdc,
            reason=str(reason),
            trade_history_id=trade_history_id,
            decision_details=clean_details,
            repeat_count=0,
            last_repeat_at=None,
            created_at=now_utc,
        )
        .on_conflict_do_nothing(index_elements=["key"])
        .returning(CTDecisionReservation.key)
    )

    result = await db_session.execute(stmt)
    inserted_key = result.scalar_one_or_none()

    if inserted_key is not None:
        new_res = await db_session.get(CTDecisionReservation, str(key))
        logger.info(
            "ct_decision_reserved",
            key=key,
            market_id=market_id,
            action=action,
            spec_id=spec_id,
        )
        return True, new_res

    # Conflict: the key was already reserved. Perform atomic SQL increment of repeat_count if requested
    if increment_on_conflict:
        update_stmt = (
            update(CTDecisionReservation)
            .where(CTDecisionReservation.key == str(key))
            .values(
                repeat_count=CTDecisionReservation.repeat_count + 1,
                last_repeat_at=now_utc,
            )
        )
        await db_session.execute(update_stmt)
        await db_session.flush()

    existing = await db_session.get(CTDecisionReservation, str(key))
    if existing is not None:
        logger.info(
            "ct_decision_repeat_detected",
            key=key,
            market_id=market_id,
            original_action=existing.action,
            repeat_count=existing.repeat_count,
        )
    return False, existing
