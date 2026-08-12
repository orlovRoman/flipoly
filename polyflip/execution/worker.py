import asyncio
import argparse
import sys
import structlog
import os
import socket
import ssl
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

from sqlalchemy import select, or_, and_, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.connection import async_session
from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExecutionFill,
    ExecutionWorkerStatus,
)
from polyflip.db.models import LiveMarket, TradeHistory, RuntimeSettings
from polyflip.execution.order_strategies import execute_gtc_ttl, execute_fak_retry
from polyflip.execution.config import ExecutionSettings
from polyflip.execution.gateways.factory import build_execution_gateway
from polyflip.execution.contracts import GatewayOrder, GatewayUnavailable
from polyflip.execution.gateways.exceptions import (
    GatewayOrderRejected,
    GatewaySubmissionUnknown,
)
from polyflip.execution.outbox import enqueue_close_request, finalize_request
from polyflip.execution.states import (
    RECONCILABLE_REQUEST_STATES,
)
from polyflip.execution.risk_checks import check_risk_limits

logger = structlog.get_logger(__name__)

# Advisory lock namespace: 2001 ‚Äî –æ–¥–∏–Ω lock –Ω–∞ —Ä–µ–∂–∏–º –∏—Å–ø–æ–ª–Ω–µ–Ω–∏—è,
# –∞ –Ω–µ –Ω–∞ —Ä—ã–Ω–æ–∫, —á—Ç–æ–±—ã –≥–ª–æ–±–∞–ª—å–Ω—ã–µ –ª–∏–º–∏—Ç—ã –Ω–µ –æ–±—Ö–æ–¥–∏–ª–∏—Å—å –ø–∞—Ä–∞–ª–ª–µ–ª—å–Ω–æ.
_MODE_LOCK_KEYS = {
    "PAPER": 1,
    "SHADOW": 2,
    "LIVE": 3,
}
_ADVISORY_LOCK_NAMESPACE = 2001


def _resolve_requested_shares(
    *,
    requested_shares: Decimal | None,
    max_spend_usdc: Decimal | None,
    limit_price: Decimal | None,
    side: str,
) -> Decimal:
    """Return a positive BUY size when a request omitted its token quantity.

    GTC/GTD orders are size-based at the Polymarket CLOB. Some release-gate
    requests intentionally leave requested_shares empty because they only
    carry a USDC budget. Derive that size from the same limit price used for
    the order, while keeping SELL requests and invalid budgets fail-closed.
    """
    if requested_shares is not None and requested_shares > 0:
        return requested_shares

    spend = max_spend_usdc or Decimal("0")
    price = limit_price or Decimal("0")
    if side.upper() == "BUY" and spend > 0 and price > 0:
        return spend / price
    return Decimal("0")


# –ß–µ—Ä–µ–∑ —Å–∫–æ–ª—å–∫–æ —Å–µ–∫—É–Ω–¥ –Ω–µ–æ–ø—Ä–µ–¥–µ–ª—ë–Ω–Ω–æ–≥–æ —Å–æ—Å—Ç–æ—è–Ω–∏—è –ø–µ—Ä–µ—Ö–æ–¥–∏–º –≤ MANUAL_REVIEW_REQUIRED
async def _enqueue_gtd_take_profit_after_fill(session, req) -> None:
    """Place a native GTD TP order after a LIVE BUY is fully filled.

    The close request is created in the same transaction as the BUY
    accounting update.  This prevents a scheduler race from creating a
    second TP request before the position becomes visible as EXIT_REQUESTED.
    """
    if req.intent != "OPEN" or req.state != "FILLED":
        return
    if str(req.requested_mode).upper() != "LIVE":
        return

    mode_row = await session.scalar(
        select(RuntimeSettings.value).where(
            RuntimeSettings.key == "TAKE_PROFIT_ORDER_MODE"
        )
    )
    if str(mode_row or "GTD").strip().upper() != "GTD":
        return

    trade = await session.get(TradeHistory, req.trade_history_id, with_for_update=True)
    if not trade or not trade.take_profit_enabled:
        return
    # Both the fill path and reconciliation can observe the same BUY fill.
    # Once a TP request is queued or terminal, this helper must be idempotent.
    if trade.take_profit_status in {"QUEUED", "FILLED", "EXPIRED"}:
        return
    if trade.position_status == "CLOSED" or not trade.take_profit_price:
        return
    remaining = Decimal(str(trade.remaining_shares or 0))
    if remaining <= 0:
        return

    market_end = trade.market_end_time
    if market_end is None:
        market_end = await session.scalar(
            select(LiveMarket.end_time_est).where(
                LiveMarket.market_id == trade.market_id
            )
        )
    if market_end is None:
        logger.warning("gtd_take_profit_missing_market_end", trade_id=trade.id)
        return
    if market_end.tzinfo is None:
        market_end = market_end.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if market_end <= now + timedelta(seconds=180):
        # Polymarket rejects GTD expirations inside its three-minute window.
        # Leave the position ACTIVE so the legacy trigger worker can use a
        # marketable close when the target is actually reached.
        logger.info(
            "gtd_take_profit_too_close_to_expiry",
            trade_id=trade.id,
            market_end=market_end.isoformat(),
        )
        return

    result = await enqueue_close_request(
        session,
        trade_id=trade.id,
        trigger_reason="TAKE_PROFIT",
        limit_price=float(trade.take_profit_price),
        expires_at=market_end,
    )
    if result.disposition.value == "CREATED":
        trade.take_profit_status = "QUEUED"
        trade.take_profit_sell_price = trade.take_profit_price
        logger.info(
            "gtd_take_profit_request_created",
            trade_id=trade.id,
            request_id=str(result.request_id),
            limit_price=trade.take_profit_price,
            expires_at=market_end.isoformat(),
        )


MAX_RECONCILIATION_AGE_SEC = 900  # 15 –º–∏–Ω—É—Ç


async def _get_dialect(session) -> str:
    """
    –í–æ–∑–≤—Ä–∞—â–∞–µ—Ç –∏–º—è –¥–∏–∞–ª–µ–∫—Ç–∞ –ë–î —á–µ—Ä–µ–∑ connection, –∞ –Ω–µ session.bind.
    AsyncSession.bind –≤—Å–µ–≥–¥–∞ None –ø—Ä–∏ async_sessionmaker ‚Äî –∏—Å–ø–æ–ª—å–∑–æ–≤–∞–Ω–∏–µ
    session.bind.dialect.name –≤—ã–∑—ã–≤–∞–µ—Ç AttributeError –≤ –ø—Ä–æ–¥–∞–∫—à–µ–Ω–µ.
    """
    conn = await session.connection()
    return conn.dialect.name


async def _acquire_mode_lock(session, requested_mode: str) -> None:
    """
    –ë–µ—Ä—ë—Ç PostgreSQL advisory lock –Ω–∞ —É—Ä–æ–≤–Ω–µ —Ä–µ–∂–∏–º–∞ –∏—Å–ø–æ–ª–Ω–µ–Ω–∏—è.
    –ì–∞—Ä–∞–Ω—Ç–∏—Ä—É–µ—Ç, —á—Ç–æ –≥–ª–æ–±–∞–ª—å–Ω—ã–µ –ª–∏–º–∏—Ç—ã (MAX_OPEN_POSITIONS, MAX_TOTAL_EXPOSURE)
    –ø—Ä–æ–≤–µ—Ä—è—é—Ç—Å—è –∏ –∏–∑–º–µ–Ω—è—é—Ç—Å—è –∞—Ç–æ–º–∞—Ä–Ω–æ: –¥–≤–∞ –≤–æ—Ä–∫–µ—Ä–∞ –æ–¥–Ω–æ–≥–æ —Ä–µ–∂–∏–º–∞ –Ω–µ –º–æ–≥—É—Ç
    –æ–¥–Ω–æ–≤—Ä–µ–º–µ–Ω–Ω–æ –ø—Ä–æ–π—Ç–∏ risk-check.
    –ù–∞ SQLite ‚Äî no-op (—Ç–µ—Å—Ç—ã).
    """
    # Bug #3 fix: –ø–æ–ª—É—á–∞–µ–º –¥–∏–∞–ª–µ–∫—Ç —á–µ—Ä–µ–∑ connection, –Ω–µ —á–µ—Ä–µ–∑ session.bind
    dialect_name = await _get_dialect(session)
    if dialect_name != "postgresql":
        return
    mode_key = _MODE_LOCK_KEYS.get(requested_mode, 0)
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:namespace, :mode_key)"),
        {"namespace": _ADVISORY_LOCK_NAMESPACE, "mode_key": mode_key},
    )


async def _persist_fills(
    session,
    attempt: ExecutionAttempt,
    fills,
) -> None:
    """
    –°–æ—Ö—Ä–∞–Ω—è–µ—Ç fills –∏–¥–µ–º–ø–æ—Ç–µ–Ω—Ç–Ω–æ —á–µ—Ä–µ–∑ ON CONFLICT DO NOTHING
    –ø–æ (gateway, provider_trade_id).
    –°–æ–≤–º–µ—Å—Ç–∏–º–æ —Å PostgreSQL –∏ SQLite (index_elements –≤–º–µ—Å—Ç–æ constraint=).
    """
    # Bug #4 fix: –∏—Å–ø–æ–ª—å–∑—É–µ–º _get_dialect –≤–º–µ—Å—Ç–æ session.bind.dialect.name
    dialect_name = await _get_dialect(session)
    insert_func = sqlite_insert if dialect_name == "sqlite" else pg_insert

    for f in fills:
        stmt = insert_func(ExecutionFill).values(
            attempt_id=attempt.id,
            provider_trade_id=f.provider_trade_id,
            gateway=f.gateway,
            gross_quote_usdc=f.gross_quote_usdc,
            price=f.price,
            shares=f.shares,
            fee_usdc=f.fee_usdc,
            timestamp=f.matched_at,
        )

        # –í PostgreSQL —è–≤–Ω–æ —Å—Å—ã–ª–∞–µ–º—Å—è –Ω–∞ constraint, –∫–æ—Ç–æ—Ä—ã–π —Å–æ–∑–¥–∞—ë—Ç—Å—è
        # –º–∏–≥—Ä–∞—Ü–∏–µ–π c4d5e6f7a8b9. –¢–∞–∫ drift –º–µ–∂–¥—É ORM –∏ production-—Å—Ö–µ–º–æ–π
        # –æ–±–Ω–∞—Ä—É–∂–∏–≤–∞–µ—Ç—Å—è —Å—Ä–∞–∑—É –∏ –Ω–µ –º–∞—Å–∫–∏—Ä—É–µ—Ç—Å—è –Ω–µ—Å–æ–≤–ø–∞–¥–∞—é—â–∏–º conflict target.
        if dialect_name == "postgresql":
            stmt = stmt.on_conflict_do_nothing(constraint="uq_execution_provider_trade")
        else:
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["gateway", "provider_trade_id"],
            )
        await session.execute(stmt)


async def _finish_submit_exception(
    session,
    *,
    request_id,
    attempt_id,
    attempt_no: int,
    requested_mode: str,
    error: str,
    is_deterministic_rejection: bool = False,
) -> None:
    """
    –í–æ—Å—Å—Ç–∞–Ω–∞–≤–ª–∏–≤–∞–µ—Ç —Å–µ—Å—Å–∏—é –ø–æ—Å–ª–µ –ª—é–±–æ–π –æ—à–∏–±–∫–∏ submit/persist/accounting.

    SQL-–æ—à–∏–±–∫–∞ –ø–µ—Ä–µ–≤–æ–¥–∏—Ç —Ç—Ä–∞–Ω–∑–∞–∫—Ü–∏—é SQLAlchemy –≤ failed state. –ü–æ—ç—Ç–æ–º—É –ø–µ—Ä–µ–¥
    –∏–∑–º–µ–Ω–µ–Ω–∏–µ–º ExecutionRequest –æ–±—è–∑–∞—Ç–µ–ª–µ–Ω rollback –∏ –ø–æ–≤—Ç–æ—Ä–Ω–∞—è –∑–∞–≥—Ä—É–∑–∫–∞ —Å—Ç—Ä–æ–∫.
    PAPER –º–æ–∂–Ω–æ –±–µ–∑–æ–ø–∞—Å–Ω–æ –ø–æ–≤—Ç–æ—Ä–∏—Ç—å: Fake gateway –Ω–µ —Å–æ–∑–¥–∞—ë—Ç –≤–Ω–µ—à–Ω–µ–≥–æ –æ—Ä–¥–µ—Ä–∞.
    LIVE/SHADOW –¥–µ—Ç–µ—Ä–º–∏–Ω–∏—Ä–æ–≤–∞–Ω–Ω–æ –æ—Ç–∫–ª–æ–Ω—è—é—Ç—Å—è –ø—Ä–∏ is_deterministic_rejection=True.
    –í –ø—Ä–æ—Ç–∏–≤–Ω–æ–º —Å–ª—É—á–∞–µ —Ç—Ä–µ–±—É—é—Ç —Ä—É—á–Ω–æ–π –ø—Ä–æ–≤–µ—Ä–∫–∏ (MANUAL_REVIEW_REQUIRED).
    """
    await session.rollback()

    req = (
        await session.execute(
            select(ExecutionRequest)
            .where(ExecutionRequest.id == request_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    attempt = await session.get(ExecutionAttempt, attempt_id)

    if not req:
        return

    now = datetime.now(timezone.utc)
    if attempt:
        attempt.status = "FAILED"
        attempt.error_msg = error[:2000]
        attempt.finished_at = now

    if requested_mode == "PAPER" and attempt_no < 3 and not is_deterministic_rejection:
        req.state = "READY"
        req.claimed_by = None
        req.claimed_at = None
        req.lease_expires_at = None
        req.expires_at = now + timedelta(seconds=max(req.ttl_seconds or 60, 60))
        req.error_reason = f"PAPER retry after submit error: {error}"[:2000]
        req.updated_at = now
    else:
        err_lower = error.lower()
        is_deterministic = (
            is_deterministic_rejection
            or ("invalid token" in err_lower)
            or ("not enough balance" in err_lower)
            or (
                "allowance" in err_lower
            )  # covers: insufficient allowance, allowance exceeded, erc20: insufficient allowance
        )
        terminal_state = (
            "REJECTED"
            if (is_deterministic or requested_mode == "PAPER")
            else "MANUAL_REVIEW_REQUIRED"
        )
        await finalize_request(
            session,
            req,
            state=terminal_state,
            error=f"Submit error: {error}"[:2000],
        )

    await session.commit()


async def claim_one(session, worker_mode: str) -> ExecutionRequest | None:
    now = datetime.now(timezone.utc)
    # Bug #4 fix: –∏—Å–ø–æ–ª—å–∑—É–µ–º _get_dialect –≤–º–µ—Å—Ç–æ session.bind.dialect.name
    dialect = await _get_dialect(session)

    where_clause = and_(
        ExecutionRequest.requested_mode == worker_mode,
        or_(
            ExecutionRequest.state == "READY",
            and_(
                ExecutionRequest.state == "CLAIMED",
                ExecutionRequest.lease_expires_at < now,
            ),
        ),
    )

    stmt = select(ExecutionRequest).where(where_clause).limit(1)
    if dialect != "sqlite":
        stmt = stmt.with_for_update(skip_locked=True)

    result = await session.execute(stmt)
    req = result.scalar_one_or_none()

    if req:
        expires_at = req.expires_at
        if expires_at is not None:
            # SQLite –≤–æ–∑–≤—Ä–∞—â–∞–µ—Ç naive datetime ‚Äî –Ω–æ—Ä–º–∞–ª–∏–∑—É–µ–º –∫ UTC
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < now:
                await finalize_request(
                    session, req, state="EXPIRED", error="TTL expired"
                )
                await session.commit()
                return None

        req.state = "CLAIMED"
        req.claimed_by = f"worker-{os.getpid()}"
        req.claimed_at = now
        req.lease_expires_at = now + timedelta(seconds=30)
        req.updated_at = now
        await session.commit()
        return req
    return None


async def process_ready_requests():
    settings = ExecutionSettings()
    worker_mode = settings.execution_mode.value

    async with async_session() as session:
        req = await claim_one(session, worker_mode)
        if not req:
            return

        logger.info(
            "execution_request_claimed",
            request_id=str(req.id),
            intent=req.intent,
            requested_mode=req.requested_mode,
        )

        # Bug #1 fix: –¥—É–±–ª–∏—Ä—É—é—â–∏–π kill-switch –±–ª–æ–∫ —É–¥–∞–ª—ë–Ω.
        # Kill-switch –¥–ª—è LIVE OPEN –ø–æ–ª–Ω–æ—Å—Ç—å—é –æ–±—Ä–∞–±–∞—Ç—ã–≤–∞–µ—Ç—Å—è –≤–Ω—É—Ç—Ä–∏
        # check_risk_limits() ‚Äî –µ–¥–∏–Ω–∞—è —Ç–æ—á–∫–∞ –ø—Ä–æ–≤–µ—Ä–∫–∏ –±–µ–∑ –¥–≤–æ–π–Ω–æ–≥–æ SELECT.

        gateway = build_execution_gateway(settings)

        if req.requested_mode == "LIVE" and gateway.name == "FAKE":
            await finalize_request(
                session,
                req,
                state="REJECTED",
                error="LIVE mode cannot be executed via fake gateway",
            )
            await session.commit()
            return

        market_stmt = select(LiveMarket).where(LiveMarket.market_id == req.market_id)
        market = (await session.execute(market_stmt)).scalar_one_or_none()

        if not market:
            await finalize_request(
                session, req, state="REJECTED", error="Market not found"
            )
            await session.commit()
            return

        token_id = (
            market.yes_token_id if req.outcome_to_buy == "YES" else market.no_token_id
        )
        side = "BUY" if req.intent == "OPEN" else "SELL"

        limit_price = req.limit_price or Decimal("0")
        max_spend_usdc = req.max_spend_usdc or Decimal("0")

        # --- Advisory lock –Ω–∞ —Ä–µ–∂–∏–º (–≥–ª–æ–±–∞–ª—å–Ω—ã–π) ---
        await _acquire_mode_lock(session, req.requested_mode)

        risk_err = await check_risk_limits(
            session,
            intent=req.intent,
            max_spend_usdc=max_spend_usdc,
            requested_mode=req.requested_mode,
            request_id=req.id,
            trade_history_id=req.trade_history_id,
        )
        if risk_err:
            logger.warning(
                "risk_limit_breached", request_id=str(req.id), error=risk_err
            )
            await finalize_request(
                session, req, state="REJECTED", error=f"Risk check failed: {risk_err}"
            )
 ◊n=Í⁄$z{-ÆÈ‹j◊ùTB ¢GFV◊BÊW'&˜%ˆ◊6r“$uDB˜&FW"Wáó&VBB÷&∂WBVÊB ¢GFV◊BÊfñÊó6ÜVEˆB“Ê˜p¢vóBfñÊ∆ó¶U˜&WVW7BÄ¢6W76ñˆ‚¿¢&W¿¢7FFS“$UÖï$TB"¿¢W'&˜#“$uDBF∂R◊&ˆfóB˜&FW"Wáó&VBB÷&∂WBVÊB"¿¢ê¢ñb&WÁG&FUˆÜó7F˜'ïˆñC†¢vóB&V'Vñ∆E˜G&FUˆ66˜VÁFñÊrá6W76ñˆ‚¬&WÁG&FUˆÜó7F˜'ïˆñBê¢G&FR“vóB6W76ñˆ‚ÊvWBÖG&FTÜó7F˜'í¬&WÁG&FUˆÜó7F˜'ïˆñBê¢ñbG&FRÊBG&FRÁ˜6óFñˆÂ˜7FGW2“$4ƒı4TB#†¢G&FRÁF∂U˜&ˆfóE˜7FGW2“$UÖï$TB ¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê¢6ˆÁFñÁVP†¢7V%˜&W2“vóBvFWvíÊvWEˆ˜&FW"ÜGFV◊BÁ&˜fñFW%ˆ˜&FW%ˆñBê¢GFV◊BÁ&˜fñFW%˜7FGW2“7V%˜&W2Á&˜fñFW%˜7FGW0¢ñb7V%˜&W2Á6WGF∆V÷VÁE˜7FFS†¢GFV◊BÁ6WGF∆V÷VÁE˜7FFR“7V%˜&W2Á6WGF∆V÷VÁE˜7FFP†¢T‰Dî‰uÙı$DU%ı5DEU4U2“g&˜¶VÁ6WBÄ¢≤$44UDTB"¬%T‰¥‰ıt‚"¬%T‰Dî‰r"¬$ƒïdR"¬$DTƒîTB'–¢ê†¢ñb7V%˜&W2Á&˜fñFW%˜7FGW2ÁWW"Çí”“$‘D4ÑTB#†¢GFV◊BÁ7FGW2“%T‰¥‰ıt‚ ¢GFV◊BÁ&˜fñFW%˜7FGW2“$‘D4ÑTB ¢&WÁ7FFR“%$T4Ù‰4îƒî‰r ¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê¢6ˆÁFñÁVP†¢ñbÄ¢7V%˜&W2Á&˜fñFW%˜7FGW2ÁWW"Çíñ‚T‰Dî‰uÙı$DU%ı5DEU4U0¢˜"7V%˜&W2Á6WGF∆V÷VÁE˜7FFRñ‚Ç%T‰Dî‰r"¬%T‰¥‰ıt‚"¬""ê¢ì†¢∆ˆvvW"ÊñÊfÚÄ¢'&V6ˆÊ6ñ∆U˜7Fñ∆≈˜VÊFñÊr"¿¢&WVW7EˆñC◊7G"á&WÊñBí¿¢&˜fñFW%˜7FGW3◊7V%˜&W2Á&˜fñFW%˜7FGW2¿¢ê¢&WÁ7FFR“%$T4Ù‰4îƒî‰r ¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê¢6ˆÁFñÁVP†¢ñb7V%˜&W2Á6WGF∆V÷VÁE˜7FFR”“$4Ù‰dï$‘TB#†¢∆ˆvvW"ÊñÊfÚÄ¢'&V6ˆÊ6ñ∆Uˆ6ˆÊfó&÷VEˆÊıˆfñ∆«5˜vóFñÊr"¿¢&WVW7EˆñC◊7G"á&WÊñBí¿¢ê¢&WÁ7FFR“%$T4Ù‰4îƒî‰r ¢&WÁWFFVEˆB“Ê˜p¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê¢6ˆÁFñÁVP†¢V∆ñb7V%˜&W2Á6WGF∆V÷VÁE˜7FFRñ‚Ä¢$dîƒTB"¿¢%$T§T5DTB"¿¢$UÖï$TB"¿¢$4‰4TƒTB"¿¢í˜"7V%˜&W2Á&˜fñFW%˜7FGW2ÁWW"Çíñ‚Ä¢%$T§T5DTB"¿¢$dîƒTB"¿¢$UÖï$TB"¿¢$4‰4TƒTB"¿¢%T‰‘D4ÑTB"¿¢ì†¢GFV◊BÁ7FGW2“$dîƒTB ¢vóBfñÊ∆ó¶U˜&WVW7BÄ¢6W76ñˆ‚¿¢&W¿¢7FFS“%$T§T5DTB"¿¢W'&˜#÷b$vFWví&WGW&ÊVBFW&÷ñÊ¬7FGW3¢ ¢b'∑7V%˜&W2Á6WGF∆V÷VÁE˜7FFR˜"7V%˜&W2Á&˜fñFW%˜7FGW7“"¿¢ê¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê¢V«6S†¢∆ˆvvW"ÊñÊfÚÄ¢'&V6ˆÊ6ñ∆U˜VÊ∂Ê˜vÂ˜7FGW2"¿¢&WVW7EˆñC◊7G"á&WÊñBí¿¢vFWvï˜7FGW3◊7V%˜&W2Á&˜fñFW%˜7FGW2¿¢ê¢&WÁ7FFR“%$T4Ù‰4îƒî‰r ¢&WÁWFFVEˆB“Ê˜p¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê†¢WÜ6WBvFWvïVÊfñ∆&∆R2S†¢∆ˆvvW"Áv&ÊñÊrÇ&vFWvï˜VÊfñ∆&∆U˜&V6ˆÊ6ñ∆R"¬W'&˜#◊7G"ÜRíê¢WÜ6WBWÜ6WFñˆ‚2S†¢∆ˆvvW"ÊWÜ6WFñˆ‚Ä¢'&V6ˆÊ6ñ∆Uˆfñ∆VB"¬&WVW7EˆñC◊7G"á&WÊñBí¬W'&˜#◊7G"ÜRê¢ê††•ˆ∆7EˆWFı˜&W6ˆ«fUˆ6ÜV6µˆ'ïˆ÷ˆFS¢Fñ7E∑7G"¬FFWFñ÷U““∑–§UDıı$U4Ù≈dUÙ4ÑT4µÙîÂDU%d≈ı4T2“c ††¶7ñÊ2FVbˆWFı˜&W6ˆ«fU˜7GV6µˆ÷ÁV≈˜&WfñWw2ÜWÜV7WFñˆÂˆ÷ˆFS¢7G"í”‚ÊˆÊS†¢"" ¢	--ÌÕ-ç}]≠Ç}≠Ω-]"}-çççR‘ÂT≈ı$UdîUuı$UTï$TB4ƒı4R›}˝-≠Ç-çRRÕç›="‡¢	˝]]B$T§T5DTB˝Ì-]˝Ì-Ú-Ì}ÕÌm›ΩRfñ∆«2}]]rvFWví‡¢	ç˝ÌΩÕ}=]"Ì--]››=‚ç}ÌΩçÌ-››=‚]ç‚		B-Ì--Ωç›=Ì¬c]¢›≠mMΩí]mç¬‡¢"" ¢Ê˜r“FFWFñ÷RÊÊ˜ráFñ÷W¶ˆÊRÁWF2ê¢∆7Eˆ6ÜV6≤“ˆ∆7EˆWFı˜&W6ˆ«fUˆ6ÜV6µˆ'ïˆ÷ˆFRÊvWBÜWÜV7WFñˆÂˆ÷ˆFRê¢ñbÄ¢∆7Eˆ6ÜV6∞¢ÊBÜÊ˜r“∆7Eˆ6ÜV6≤íÁF˜F≈˜6V6ˆÊG2Çí¬UDıı$U4Ù≈dUÙ4ÑT4µÙîÂDU%d≈ı4T0¢ì†¢&WGW&‡¢ˆ∆7EˆWFı˜&W6ˆ«fUˆ6ÜV6µˆ'ïˆ÷ˆFU∂WÜV7WFñˆÂˆ÷ˆFU““Ê˜p†¢6WGFñÊw2“WÜV7WFñˆÂ6WGFñÊw2Çê¢7ñÊ2vóFÇ7ñÊ5˜6W76ñˆ‚Çí26W76ñˆ„†¢÷ÁV≈ˆ7WFˆfb“Ê˜r“Fñ÷VFV«FÜ÷ñÁWFW3”Rê¢7F◊Eˆ÷ÁV¬“6V∆V7BÑWÜV7WFñˆÂ&WVW7BíÁvÜW&RÄ¢WÜV7WFñˆÂ&WVW7BÁ7FFR”“$‘ÂT≈ı$UdîUuı$UTï$TB"¿¢WÜV7WFñˆÂ&WVW7BÁ&WVW7FVEˆ÷ˆFR”“WÜV7WFñˆÂˆ÷ˆFR¿¢WÜV7WFñˆÂ&WVW7BÊñÁFVÁB”“$4ƒı4R"¿¢WÜV7WFñˆÂ&WVW7BÁWFFVEˆB√“÷ÁV≈ˆ7WFˆfb¿¢ê¢7GV6µˆ÷ÁV≈˜&W2“ÜvóB6W76ñˆ‚ÊWÜV7WFRá7F◊Eˆ÷ÁV¬ííÁ66∆'2ÇíÊ∆¬Çê¢ñbÊ˜B7GV6µˆ÷ÁV≈˜&W3†¢&WGW&‡†¢vFWví“'Vñ∆EˆWÜV7WFñˆÂˆvFWvíá6WGFñÊw2ê¢f˜"&Wñ‚7GV6µˆ÷ÁV≈˜&W3†¢∆ˆvvW"ÊñÊfÚÄ¢&WFı˜&W6ˆ«fñÊu˜7GV6µˆ÷ÁV≈˜&WfñWuˆ6∆˜6U˜&WVW7B"¿¢&WVW7EˆñC◊7G"á&WÊñBí¿¢G&FUˆÜó7F˜'ïˆñC◊&WÁG&FUˆÜó7F˜'ïˆñB¿¢ê¢GFV◊E˜7F◊B“Ä¢6V∆V7BÑWÜV7WFñˆ‰GFV◊Bê¢ÁvÜW&RÑWÜV7WFñˆ‰GFV◊BÁ&WVW7EˆñB”“&WÊñBê¢Ê˜&FW%ˆ'íÑWÜV7WFñˆ‰GFV◊BÊGFV◊EˆÊÚÊFW62Çíê¢Ê∆ñ÷óBÉê¢ê¢GFV◊B“ÜvóB6W76ñˆ‚ÊWÜV7WFRÜGFV◊E˜7F◊BííÁ66∆%ˆˆÊUˆ˜%ˆÊˆÊRÇê†¢fñ∆«2“µ–¢ñbGFV◊BÊBGFV◊BÁ&˜fñFW%ˆ˜&FW%ˆñC†¢G'ì†¢÷&∂WB“vóB6W76ñˆ‚Á66∆"Ä¢6V∆V7BÑ∆ófT÷&∂WBíÁvÜW&RÑ∆ófT÷&∂WBÊ÷&∂WEˆñB”“&WÊ÷&∂WEˆñBê¢ê¢Fˆ∂VÂˆñB“Ä¢÷&∂WBÁñW5˜Fˆ∂VÂˆñ@¢ñb÷&∂WBÊB&WÊ˜WF6ˆ÷U˜Fıˆ'Wí”“%îU2 ¢V«6RÜ÷&∂WBÊÊı˜Fˆ∂VÂˆñBñb÷&∂WBV«6R""ê¢ê¢ñbFˆ∂VÂˆñC†¢fñ∆«2“vóBvFWvíÊfWF6Öˆ˜&FW%ˆfñ∆«2Ä¢GFV◊BÁ&˜fñFW%ˆ˜&FW%ˆñB¬Fˆ∂VÂˆñ@¢ê¢WÜ6WBWÜ6WFñˆ‚2S†¢∆ˆvvW"Áv&ÊñÊrÄ¢&WFı˜&W6ˆ«fUˆfñ∆«5ˆfWF6Öˆfñ∆VB"¿¢&WVW7EˆñC◊7G"á&WÊñBí¿¢W'&˜#◊7G"ÜRí¿¢ê†¢ñbfñ∆«2ÊBGFV◊C†¢vóB˜W'6ó7Eˆfñ∆«2á6W76ñˆ‚¬GFV◊B¬fñ∆«2ê¢vóBfñÊ∆ó¶U˜&WVW7Bá6W76ñˆ‚¬&W¬7FFS“$dîƒƒTB"ê¢V«6S†¢vóBfñÊ∆ó¶U˜&WVW7BÄ¢6W76ñˆ‚¿¢&W¿¢7FFS“%$T§T5DTB"¿¢W'&˜#“$WFÚ◊&W6ˆ«fVC¢ÊÚfñ∆«26ˆÊfó&÷VBgFW"‘ÂT≈ı$UdîUrFñ÷V˜WB"¿¢ê†¢ñb&WÁG&FUˆÜó7F˜'ïˆñC†¢vóB&V'Vñ∆E˜G&FUˆ66˜VÁFñÊrá6W76ñˆ‚¬&WÁG&FUˆÜó7F˜'ïˆñBê¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê††¶FVb6∆76ñgï˜&VFñÊW75ˆW'&˜"ÜWÜ3¢WÜ6WFñˆ‚í”‚7G#†¢÷W76vR“7G"ÜWÜ2íÊ∆˜vW"Çê†¢ñbó6ñÁ7FÊ6RÜWÜ2¬7ñÊ6ñÚÂFñ÷V˜WDW'&˜"ì†¢&WGW&‚%$TDî‰U55ıDî‘TıUB †¢ñbó6ñÁ7FÊ6RÜWÜ2¬76¬Â54ƒW'&˜"ì†¢&WGW&‚%D≈5ıE$Â5ı%EÙU%$ı" †¢ñb'76¬"ñ‚÷W76vR˜"&&B&V6˜&B÷2"ñ‚÷W76vS†¢&WGW&‚%D≈5ıE$Â5ı%EÙU%$ı" †¢ñb&6ˆÊÊV7Fñˆ‚"ñ‚÷W76vR˜"'G&Á7˜'B"ñ‚÷W76vS†¢&WGW&‚$‰UEtı$µıE$Â5ı%EÙU%$ı" †¢&WGW&‚%$TDî‰U55ıT‰¥‰ıtÂÙU%$ı" ††¶7ñÊ2FVbV&∆ó6Öˆ∆ófVÊW75ˆˆÊ6RÄ¢6W76ñˆ„¢7ñÊ56W76ñˆ‚¿¢v˜&∂W%ˆñC¢7G"¿¢WÜV7WFñˆÂˆ÷ˆFS¢7G"¿¢ì†¢Ê˜r“FFWFñ÷RÊÊ˜ráFñ÷W¶ˆÊRÁWF2ê¢Fñ∆V7EˆÊ÷R“vóBˆvWEˆFñ∆V7Bá6W76ñˆ‚ê¢ñÁ6W'EˆgVÊ2“7∆óFUˆñÁ6W'BñbFñ∆V7EˆÊ÷R”“'7∆óFR"V«6RuˆñÁ6W'@†¢7F◊B“Ä¢ñÁ6W'EˆgVÊ2ÑWÜV7WFñˆÂv˜&∂W%7FGW2ê¢Áf«VW2Ä¢v˜&∂W%ˆñC◊v˜&∂W%ˆñB¿¢WÜV7WFñˆÂˆ÷ˆFS÷WÜV7WFñˆÂˆ÷ˆFR¿¢ÜV'F&VEˆC÷Ê˜r¿¢vFWvï˜&VGì‘f«6R¿¢ê¢ÊˆÂˆ6ˆÊf∆ñ7EˆFı˜WFFRÄ¢ñÊFWÖˆV∆V÷VÁG3’≤'v˜&∂W%ˆñB%“¿¢6WEÛ◊∞¢&ÜV'F&VEˆB#¢Ê˜r¿¢&WÜV7WFñˆÂˆ÷ˆFR#¢WÜV7WFñˆÂˆ÷ˆFR¿¢“¿¢ê¢ê†¢vóB6W76ñˆ‚ÊWÜV7WFRá7F◊Bê¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê††¶7ñÊ2FVbV&∆ó6Öˆ∆ófVÊW72áv˜&∂W%ˆñC¢7G"¬WÜV7WFñˆÂˆ÷ˆFS¢7G"ì†¢vÜñ∆RG'VS†¢G'ì†¢7ñÊ2vóFÇ7ñÊ5˜6W76ñˆ‚Çí26W76ñˆ„†¢vóBV&∆ó6Öˆ∆ófVÊW75ˆˆÊ6RÄ¢6W76ñˆ‚¿¢v˜&∂W%ˆñB¿¢WÜV7WFñˆÂˆ÷ˆFR¿¢ê¢WÜ6WB7ñÊ6ñÚ‰6Ê6V∆∆VDW'&˜#†¢&ó6P¢WÜ6WBWÜ6WFñˆ„†¢∆ˆvvW"ÊWÜ6WFñˆ‚Ç&ÜV'F&VEˆfñ∆VB"ê†¢vóB7ñÊ6ñÚÁ6∆VWÉê††¶7ñÊ2FVb&Vg&W6ÖˆvFWvï˜&VFñÊW75ˆˆÊ6RÄ¢6W76ñˆ„¢7ñÊ56W76ñˆ‚¿¢v˜&∂W%ˆñC¢7G"¿¢WÜV7WFñˆÂˆ÷ˆFS¢7G"¿¢vFWví¿¢ì†¢Ê˜r“FFWFñ÷RÊÊ˜ráFñ÷W¶ˆÊRÁWF2ê†¢w2“vóB6W76ñˆ‚ÊvWBÑWÜV7WFñˆÂv˜&∂W%7FGW2¬v˜&∂W%ˆñBê¢ñbÊ˜Bw3†¢&WGW&‡†¢&ˆ&U˜Fˆ∂VÂˆñB“vóB6W76ñˆ‚Á66∆"Ä¢6V∆V7BÑ∆ófT÷&∂WBÁñW5˜Fˆ∂VÂˆñBê¢ÁvÜW&RÄ¢∆ófT÷&∂WBÊVÊE˜Fñ÷UˆW7B‚Ê˜r¿¢∆ófT÷&∂WBÁñW5˜Fˆ∂VÂˆñBÊó5ˆÊ˜BÑÊˆÊRí¿¢ê¢Ê˜&FW%ˆ'íÑ∆ófT÷&∂WBÊ∆7E˜WFFVBÊFW62Çíê¢Ê∆ñ÷óBÉê¢ê†¢ñbÊ˜B&ˆ&U˜Fˆ∂VÂˆñC†¢w2ÊvFWvï˜&VGí“f«6P¢w2Ê6ˆÊFóFñˆÊ≈ˆ∆∆˜vÊ6U˜&VGí“ÊˆÊP¢w2Ê∆7EˆW'&˜%ˆ6ˆFR“$‰ıÙ$ıd≈ı$Ù$UıDÙ¥T‚ ¢w2Ê∆7EˆW'&˜%ˆ÷W76vR“Ä¢-	›]"≠-ç-›Ì=‚Ω›≠MΩÚ˝Ì-]≠Ç6ˆÊFóFñˆÊ¬Fˆ∂V‚&˜f¬ ¢ê¢w2Á&VFñÊW75ˆ6ÜV6∂VEˆB“Ê˜p¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê¢&WGW&‡†¢6ˆÊFóFñˆÊ≈˜Fˆ∂VÂˆñG2“á&ˆ&U˜Fˆ∂VÂˆñB¬ê¢$TDî‰U55ıDî‘TıUEı4T4Ù‰E2“P†¢G'ì†¢&VFñÊW72“vóB7ñÊ6ñÚÁvóEˆf˜"Ä¢vFWvíÊvWE˜&VFñÊW72Ä¢6ˆÊFóFñˆÊ≈˜Fˆ∂VÂˆñG3÷6ˆÊFóFñˆÊ≈˜Fˆ∂VÂˆñG2¿¢í¿¢Fñ÷V˜WC’$TDî‰U55ıDî‘TıUEı4T4Ù‰E2¿¢ê†¢ñbvWFGG"á&VFñÊW72¬&W'&˜%ˆ÷W76vR"¬ÊˆÊRì†¢W'&˜%ˆ6ˆFR“vWFGG"Ä¢&VFñÊW72¬&W'&˜%ˆ6ˆFR"¬ÊˆÊP¢í˜"6∆76ñgï˜&VFñÊW75ˆW'&˜"ÑWÜ6WFñˆ‚á&VFñÊW72ÊW'&˜%ˆ÷W76vRíê†¢ñbW'&˜%ˆ6ˆFRñ‚∞¢%D≈5ıE$Â5ı%EÙU%$ı""¿¢$‰UEtı$µıE$Â5ı%EÙU%$ı""¿¢%$TDî‰U55ıDî‘TıUB"¿¢”†¢G'ì†¢vóBvFWvíÊñÁf∆ñFFUˆ6∆ñVÁBÇê¢WÜ6WBWÜ6WFñˆ‚2WÜ3†¢∆ˆvvW"Áv&ÊñÊrÄ¢&vFWvïˆ6∆ñVÁEˆñÁf∆ñFFñˆÂˆfñ∆VB"¿¢W'&˜#◊7G"ÜWÜ2í¿¢ê†¢w2ÊvFWvï˜&VGí“f«6P¢w2Á&VFñÊW75ˆ6ÜV6∂VEˆB“Ê˜p¢w2Ê∆7EˆW'&˜%ˆ6ˆFR“W'&˜%ˆ6ˆFP¢w2Ê∆7EˆW'&˜%ˆ÷W76vR“&VFñÊW72ÊW'&˜%ˆ÷W76vP¢V«6S†¢w2ÊvFWvï˜&VGí“&VFñÊW72Á&VGê¢w2Á&VFñÊW75ˆ6ÜV6∂VEˆB“Ê˜p¢w2Á&VFñÊW75˜7V66W75ˆB“Ê˜p¢w2Ê∆7EˆW'&˜%ˆ6ˆFR“ÊˆÊP¢w2Ê∆7EˆW'&˜%ˆ÷W76vR“ÊˆÊP†¢ñb&VFñÊW72Ê&∆Ê6S†¢w2Ê&∆Ê6U˜W6F2“FV6ñ÷¬á7G"á&VFñÊW72Ê&∆Ê6RÊ&∆Ê6U˜W6F2íê¢ñb&VFñÊW72Ê6ˆ∆∆FW&≈ˆ∆∆˜vÊ6U˜&VGíó2Ê˜BÊˆÊS†¢w2Ê6ˆ∆∆FW&≈ˆ∆∆˜vÊ6U˜&VGí“&VFñÊW72Ê6ˆ∆∆FW&≈ˆ∆∆˜vÊ6U˜&VGê¢ñb&VFñÊW72Ê6ˆÊFóFñˆÊ≈ˆ∆∆˜vÊ6U˜&VGíó2Ê˜BÊˆÊS†¢w2Ê6ˆÊFóFñˆÊ≈ˆ∆∆˜vÊ6U˜&VGí“&VFñÊW72Ê6ˆÊFóFñˆÊ≈ˆ∆∆˜vÊ6U˜&VGê†¢w2Ê7&VFVÁFñ«5ˆ∆ˆFVB“&VFñÊW72Ê7&VFVÁFñ«5ˆ∆ˆFV@¢w2Áv∆∆WEˆFG&W72“&VFñÊW72Áv∆∆WEˆFG&W70¢w2ÊÊWGv˜&µˆ6ÜñÂˆñB“&VFñÊW72ÊÊWGv˜&µˆ6ÜñÂˆñ@†¢WÜ6WBWÜ6WFñˆ‚2WÜ3†¢ñbó6ñÁ7FÊ6RÜWÜ2¬7ñÊ6ñÚ‰6Ê6V∆∆VDW'&˜"ì†¢&ó6P†¢W'&˜%ˆ6ˆFR“6∆76ñgï˜&VFñÊW75ˆW'&˜"ÜWÜ2ê†¢2
Ìç-¬≠Ωç]›"˝Ç]-]-Ìíı54¬Ìçç≠P¢ñbW'&˜%ˆ6ˆFRñ‚∞¢%D≈5ıE$Â5ı%EÙU%$ı""¿¢$‰UEtı$µıE$Â5ı%EÙU%$ı""¿¢%$TDî‰U55ıDî‘TıUB"¿¢”†¢G'ì†¢vóBvFWvíÊñÁf∆ñFFUˆ6∆ñVÁBÇê¢WÜ6WBWÜ6WFñˆ‚2WÜ3†¢∆ˆvvW"Áv&ÊñÊrÄ¢&vFWvïˆ6∆ñVÁEˆñÁf∆ñFFñˆÂˆfñ∆VB"¿¢W'&˜#◊7G"ÜWÜ2í¿¢ê†¢w2ÊvFWvï˜&VGí“f«6P¢w2Á&VFñÊW75ˆ6ÜV6∂VEˆB“Ê˜p¢w2Ê∆7EˆW'&˜%ˆ6ˆFR“W'&˜%ˆ6ˆFP¢w2Ê∆7EˆW'&˜%ˆ÷W76vR“7G"ÜWÜ2ê†¢vóB6W76ñˆ‚Ê6ˆ÷÷óBÇê††¶7ñÊ2FVb&Vg&W6ÖˆvFWvï˜&VFñÊW72Ä¢v˜&∂W%ˆñC¢7G"¿¢WÜV7WFñˆÂˆ÷ˆFS¢7G"¿¢vFWví¿¢ì†¢vÜñ∆RG'VS†¢G'ì†¢7ñÊ2vóFÇ7ñÊ5˜6W76ñˆ‚Çí26W76ñˆ„†¢vóB&Vg&W6ÖˆvFWvï˜&VFñÊW75ˆˆÊ6RÄ¢6W76ñˆ‚¿¢v˜&∂W%ˆñB¿¢WÜV7WFñˆÂˆ÷ˆFR¿¢vFWví¿¢ê¢WÜ6WB7ñÊ6ñÚ‰6Ê6V∆∆VDW'&˜#†¢&ó6P¢WÜ6WBWÜ6WFñˆ„†¢∆ˆvvW"ÊWÜ6WFñˆ‚Ç&vFWvï˜&VFñÊW75˜&Vg&W6Öˆfñ∆VB"ê†¢vóB7ñÊ6ñÚÁ6∆VWÉ3ê††¶7ñÊ2FVbWÜV7WFñˆÂ˜v˜&∂W%ˆ∆ˆ˜Çì†¢∆ˆvvW"ÊñÊfÚÇ&WÜV7WFñˆÂ˜v˜&∂W%˜7F'FVB"ê¢6WGFñÊw2“WÜV7WFñˆÂ6WGFñÊw2Çê¢vFWví“'Vñ∆EˆWÜV7WFñˆÂˆvFWvíá6WGFñÊw2ê¢WÜV7WFñˆÂˆ÷ˆFR“6WGFñÊw2ÊWÜV7WFñˆÂˆ÷ˆFRÁf«VP¢v˜&∂W%ˆñB“b'∂WÜV7WFñˆÂˆ÷ˆFW”ß∑6ˆ6∂WBÊvWFÜ˜7FÊ÷RÇó”ß∂˜2ÊvWGñBÇó“ †¢ÜV'F&VE˜F6≤“7ñÊ6ñÚÊ7&VFU˜F6≤áV&∆ó6Öˆ∆ófVÊW72áv˜&∂W%ˆñB¬WÜV7WFñˆÂˆ÷ˆFRíê¢&VFñÊW75˜F6≤“7ñÊ6ñÚÊ7&VFU˜F6≤Ä¢&Vg&W6ÖˆvFWvï˜&VFñÊW72áv˜&∂W%ˆñB¬WÜV7WFñˆÂˆ÷ˆFR¬vFWvíê¢ê†¢G'ì†¢vÜñ∆RG'VS†¢G'ì†¢vóB&ˆ6W75˜&VGï˜&WVW7G2Çê¢vóB&V6ˆÊ6ñ∆Uˆ7FófU˜&WVW7G2Çê¢vóBˆWFı˜&W6ˆ«fU˜7GV6µˆ÷ÁV≈˜&WfñWw2ÜWÜV7WFñˆÂˆ÷ˆFRê¢WÜ6WB7ñÊ6ñÚ‰6Ê6V∆∆VDW'&˜#†¢&ó6P¢WÜ6WBWÜ6WFñˆ‚2WÜ3†¢∆ˆvvW"ÊWÜ6WFñˆ‚Ç&WÜV7WFñˆÂ˜v˜&∂W%ˆW'&˜""¬W'&˜#◊7G"ÜWÜ2íê†¢vóB7ñÊ6ñÚÁ6∆VWÉê¢fñÊ∆«ì†¢ÜV'F&VE˜F6≤Ê6Ê6V¬Çê¢&VFñÊW75˜F6≤Ê6Ê6V¬Çê¢vóB7ñÊ6ñÚÊvFÜW"ÜÜV'F&VE˜F6≤¬&VFñÊW75˜F6≤¬&WGW&ÂˆWÜ6WFñˆÁ3’G'VRê††¶ñbıˆÊ÷UıÚ”“%ıˆ÷ñÂıÚ#†¢'6W"“&w'6R‰&wV÷VÁE'6W"Çê¢'6W"ÊFEˆ&wV÷VÁBÄ¢"“÷G'í◊'V‚"¬7Fñˆ„“'7F˜&U˜G'VR"¬ÜV«“%'V‚ñ‚G'í◊'V‚÷ˆFRÊBWÜóB ¢ê¢&w2“'6W"Á'6Uˆ&w2Çê†¢ñb&w2ÊG'ï˜'V„†¢&ñÁBÇ$G'í'V‚7V66W76gV¬‚"ê¢7ó2ÊWÜóBÉê†¢7G'V7F∆ˆrÊ6ˆÊfñwW&RÄ¢&ˆ6W76˜'3’∞¢7G'V7F∆ˆrÁ&ˆ6W76˜'2ÂFñ÷U7F◊W"Üf◊C“&ó6Ú"í¿¢7G'V7F∆ˆrÁ&ˆ6W76˜'2‰•4ÙÂ&VÊFW&W"Çí¿¢–¢ê¢7ñÊ6ñÚÁ'V‚ÜWÜV7WFñˆÂ˜v˜&∂W%ˆ∆ˆ˜Çíê