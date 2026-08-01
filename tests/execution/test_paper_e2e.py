"""
Сквозной PAPER-тест.

Pipeline:
  TradeHistory (OPENING)
  → enqueue_open_request → ExecutionRequest (READY)
  → claim_one (CLAIMED)
  → FakeExecutionGateway.submit → fills
  → _persist_fills → ExecutionFill
  → rebuild_trade_accounting → TradeHistory (OPEN, entry_filled_shares)
  → settle_resolved_position → TradeHistory (CLOSED, PnL)

Ключевые инварианты:
- PAPER никогда не попадает в AWAITING_APPROVAL
- После fill: position_status = OPEN, remaining_shares = entry_filled_shares
- После settlement WIN: position_status = CLOSED, realized_pnl_usdc > 0
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import select

from polyflip.db.models import TradeHistory, LiveMarket
from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExecutionFill,
    ExposureReservation,
)
from polyflip.execution.outbox import enqueue_open_request, finalize_request
from polyflip.execution.worker import (
    claim_one,
    _persist_fills,
    rebuild_trade_accounting,
)
from polyflip.execution.settlement_service import settle_resolved_position
from polyflip.execution.gateways.fake import FakeExecutionGateway
from polyflip.execution.contracts import GatewayOrder
from polyflip.execution.config import ExecutionMode

# ─────────────── Fixtures ───────────────


def make_trade(
    db_session,
    *,
    trade_id: int = 100,
    market_id: str = "0xMARKET",
    asset="POL",
    outcome_bought="YES",
    mode="PAPER",
) -> TradeHistory:
    trade = TradeHistory(
        id=trade_id,
        market_id=market_id,
        asset=asset,
        outcome_bought=outcome_bought,
        amount_usdc=Decimal("10"),
        executed_price=Decimal("0.50"),
        predicted_flip_prob=0.65,
        active_features="{}",
        status="PENDING",
        mode=mode,
        position_status="OPENING",
        # version=0: fills ещё нет, constraint (version=0 OR fields NOT NULL) пройдёт
        position_accounting_version=0,
        position_version=0,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(trade)
    return trade


def make_market(db_session, *, market_id: str = "0xMARKET") -> LiveMarket:
    market = LiveMarket(
        market_id=market_id,
        question="Will ETH flip?",
        asset="POL",
        yes_token_id="YES_TOKEN",
        no_token_id="NO_TOKEN",
        active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(market)
    return market


# ─────────────── Tests ───────────────


@pytest.mark.asyncio
async def test_paper_request_goes_to_ready_not_awaiting_approval(db_session):
    """
    PAPER-запрос всегда READY — AWAITING_APPROVAL только для LIVE.
    """
    trade = make_trade(db_session)
    await db_session.flush()

    result = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id=trade.market_id,
        asset=trade.asset,
        outcome_to_buy="YES",
        target_amount_usdc=float(trade.amount_usdc),
        limit_price=0.50,
        requested_mode=ExecutionMode.PAPER,
    )

    assert result is not None
    assert result.disposition == "CREATED"

    req = await db_session.get(ExecutionRequest, result.request_id)
    assert req is not None
    assert req.state == "READY"
    assert req.position_version_snapshot == 0


@pytest.mark.asyncio
async def test_paper_e2e_fill_to_open(db_session):
    """
    Полный E2E: READY → CLAIMED → FILLED → TradeHistory OPEN.
    """
    trade = make_trade(db_session)
    await db_session.flush()

    # 1. Enqueue
    result = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id=trade.market_id,
        asset=trade.asset,
        outcome_to_buy="YES",
        target_amount_usdc=10.0,
        limit_price=0.50,
        requested_mode=ExecutionMode.PAPER,
    )
    assert result is not None
    await db_session.commit()

    # 2. Claim
    req = await claim_one(db_session, "PAPER")
    assert req is not None
    assert req.state == "CLAIMED"
    assert req.trade_history_id == trade.id

    # 3. Submit via FakeGateway
    gateway = FakeExecutionGateway()
    attempt = ExecutionAttempt(
        request_id=req.id,
        gateway=gateway.name,
        attempt_no=1,
        submission_key=f"{req.idempotency_key}:1",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(attempt)
    await db_session.flush()

    order = GatewayOrder(
        attempt_id=attempt.id,
        market_id=req.market_id,
        asset=req.asset,
        outcome_to_buy=req.outcome_to_buy,
        token_id="YES_TOKEN",
        side="BUY",
        limit_price=Decimal("0.50"),
        requested_shares=req.requested_shares or Decimal("20"),
        max_spend_usdc=req.max_spend_usdc or Decimal("10"),
    )
    sub_res = await gateway.submit(order)

    assert sub_res.accepted is True
    assert sub_res.settlement_state == "CONFIRMED"
    assert len(sub_res.fills) == 1

    fill = sub_res.fills[0]
    assert fill.shares == Decimal("20")  # 10 / 0.50

    # 4. Persist fills
    await _persist_fills(db_session, attempt, sub_res.fills)

    # 5. Finalize request
    req.filled_shares = fill.shares
    req.filled_cost_usdc = fill.gross_quote_usdc
    await finalize_request(db_session, req, state="FILLED")

    # 6. Rebuild accounting
    await rebuild_trade_accounting(db_session, trade.id)

    await db_session.commit()

    # 7. Reload and assert
    await db_session.refresh(trade)

    assert trade.position_status == "OPEN"
    assert trade.entry_filled_shares == Decimal("20")
    assert trade.remaining_shares == Decimal("20")
    assert trade.status == "SUCCESS"

    # Fills in DB
    fills_in_db = (
        (
            await db_session.execute(
                select(ExecutionFill).where(ExecutionFill.attempt_id == attempt.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(fills_in_db) == 1
    assert fills_in_db[0].shares == Decimal("20")


@pytest.mark.asyncio
async def test_paper_e2e_fill_then_settle_win(db_session):
    """
    Полный pipeline до CLOSED с WIN settlement.
    20 shares куплено по 0.50 (basis = 10 USDC).
    WIN → payout = 20 × 1.0 = 20, PnL = +10 USDC.
    """
    trade = make_trade(db_session)
    await db_session.flush()

    result = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id=trade.market_id,
        asset=trade.asset,
        outcome_to_buy="YES",
        target_amount_usdc=10.0,
        limit_price=0.50,
        requested_mode=ExecutionMode.PAPER,
    )
    await db_session.commit()

    req = await claim_one(db_session, "PAPER")
    assert req is not None

    gateway = FakeExecutionGateway()
    attempt = ExecutionAttempt(
        request_id=req.id,
        gateway=gateway.name,
        attempt_no=1,
        submission_key=f"{req.idempotency_key}:1",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(attempt)
    await db_session.flush()

    order = GatewayOrder(
        attempt_id=attempt.id,
        market_id=req.market_id,
        asset=req.asset,
        outcome_to_buy=req.outcome_to_buy,
        token_id="YES_TOKEN",
        side="BUY",
        limit_price=Decimal("0.50"),
        requested_shares=req.requested_shares or Decimal("20"),
        max_spend_usdc=req.max_spend_usdc or Decimal("10"),
    )
    sub_res = await gateway.submit(order)

    await _persist_fills(db_session, attempt, sub_res.fills)
    req.filled_shares = sub_res.fills[0].shares
    req.filled_cost_usdc = sub_res.fills[0].gross_quote_usdc
    await finalize_request(db_session, req, state="FILLED")
    await rebuild_trade_accounting(db_session, trade.id)
    await db_session.commit()

    await db_session.refresh(trade)
    assert trade.position_status == "OPEN"

    # Settlement: рынок разрешён YES → WIN
    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="YES",
        payout_per_share=Decimal("1.0"),
        settlement_fee_usdc=Decimal("0"),
    )
    await db_session.commit()

    await db_session.refresh(trade)
    assert trade.position_status == "CLOSED"
    assert trade.remaining_shares == Decimal("0")
    assert trade.realized_pnl_usdc == Decimal("10")  # 20 - 10


@pytest.mark.asyncio
async def test_persist_fills_idempotent(db_session):
    """
    _persist_fills с одинаковым provider_trade_id дважды → одна строка в БД.
    """
    trade = make_trade(db_session)
    await db_session.flush()

    result = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id=trade.market_id,
        asset=trade.asset,
        outcome_to_buy="YES",
        target_amount_usdc=10.0,
        limit_price=0.50,
        requested_mode=ExecutionMode.PAPER,
    )
    await db_session.flush()

    req = (
        await db_session.execute(
            select(ExecutionRequest).where(ExecutionRequest.id == result.request_id)
        )
    ).scalar_one()

    attempt = ExecutionAttempt(
        request_id=req.id,
        gateway="FAKE",
        attempt_no=1,
        submission_key="TEST:1",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(attempt)
    await db_session.flush()

    from polyflip.execution.contracts import TradeExecution

    fills = (
        TradeExecution(
            provider_trade_id="IDEMPOTENT_TRADE_ID",
            gateway="FAKE",
            gross_quote_usdc=Decimal("10"),
            price=Decimal("0.5"),
            shares=Decimal("20"),
            fee_usdc=Decimal("0"),
            matched_at=datetime.now(timezone.utc),
        ),
    )

    # Первый вызов
    await _persist_fills(db_session, attempt, fills)
    await db_session.flush()

    # Второй вызов — должен быть no-op
    await _persist_fills(db_session, attempt, fills)
    await db_session.commit()

    all_fills = (
        (
            await db_session.execute(
                select(ExecutionFill).where(ExecutionFill.attempt_id == attempt.id)
            )
        )
        .scalars()
        .all()
    )

    assert len(all_fills) == 1, f"Expected 1 fill, got {len(all_fills)}"
    assert all_fills[0].provider_trade_id == "IDEMPOTENT_TRADE_ID"
