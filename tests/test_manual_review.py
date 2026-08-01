import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone

from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import (
    ExecutionRequest,
    ExecutionAttempt,
    ExecutionFill,
)
from polyflip.execution.manual_review_service import evaluate_no_fill_eligibility


def make_trade():
    now = datetime.now(timezone.utc)
    return TradeHistory(
        market_id="test_market",
        asset="USDC",
        outcome_bought="YES",
        amount_usdc=Decimal("10.0"),
        predicted_flip_prob=0.6,
        active_features="test",
        model_version="1.0",
        edge=Decimal("0.1"),
        market_role="test",
        strategy_type="test",
        p_flip_effective=0.6,
        p_win_effective=0.6,
        stop_loss_pct=Decimal("0.1"),
        take_profit_enabled=False,
        take_profit_multiplier=Decimal("2.0"),
        mode="LIVE",
        status="PENDING",
        executed_price=0.0,
        position_status="OPENING",
        entry_filled_shares=Decimal("0"),
        entry_cost_usdc=Decimal("0"),
        remaining_shares=Decimal("0"),
        position_accounting_version=0,
        model_key="test",
        confirm_model_key="test",
        confirm_model_version="test",
        model_attribution_source="test",
        created_at=now,
        updated_at=now,
    )


def make_req(trade_id, mode="LIVE"):
    now = datetime.now(timezone.utc)
    return ExecutionRequest(
        id=uuid.uuid4(),
        trade_history_id=trade_id,
        intent="OPEN",
        trigger_reason="TEST",
        market_id="test_market",
        asset="USDC",
        outcome_to_buy="YES",
        state="MANUAL_REVIEW_REQUIRED",
        requested_mode=mode,
        target_amount_usdc=Decimal("10.0"),
        filled_shares=Decimal("0.0"),
        filled_cost_usdc=Decimal("0.0"),
        max_slippage_pct=Decimal("0.05"),
        ttl_seconds=30,
        expires_at=now,
        created_at=now,
        updated_at=now,
    )


def make_attempt(req_id, **kwargs):
    now = datetime.now(timezone.utc)
    base = {
        "id": uuid.uuid4(),
        "request_id": req_id,
        "gateway": "polymarket",
        "started_at": now,
        "attempt_no": 1,
        "status": "PENDING",
    }
    base.update(kwargs)
    return ExecutionAttempt(**base)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_allowed(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id)
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is True


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_provider_order_id(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, provider_order_id="0x")
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("provider_order_id" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_tx_hash(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, tx_hash="0x")
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("tx_hash" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_provider_trade_ids(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, provider_trade_ids="t1")
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("provider_trade_ids" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_transaction_hashes(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id, transaction_hashes="t1")
    db_session.add(attempt)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("transaction_hashes" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_execution_fill(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    db_session.add(req)
    await db_session.flush()

    attempt = make_attempt(req.id)
    db_session.add(attempt)
    await db_session.flush()

    fill = ExecutionFill(
        id=uuid.uuid4(),
        attempt_id=attempt.id,
        provider_trade_id="x",
        shares=Decimal("1.0"),
        price=Decimal("0.5"),
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(fill)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("execution_fills" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_filled_shares(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    req.filled_shares = Decimal("1.0")
    db_session.add(req)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("заполненные shares" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_filled_cost_usdc(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id)
    req.filled_cost_usdc = Decimal("1.0")
    db_session.add(req)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("стоимость исполнения" in blocker for blocker in eligibility.blockers)


@pytest.mark.asyncio
async def test_evaluate_no_fill_eligibility_denied_paper(db_session):
    trade = make_trade()
    db_session.add(trade)
    await db_session.flush()

    req = make_req(trade.id, mode="PAPER")
    db_session.add(req)
    await db_session.commit()

    eligibility = await evaluate_no_fill_eligibility(db_session, req)
    assert eligibility.allowed is False
    assert any("не относится к LIVE" in blocker for blocker in eligibility.blockers)
