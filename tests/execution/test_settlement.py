"""
Тесты settlement_service.settle_resolved_position().

Проверяемые сценарии:
1. PAPER WIN  — payout = shares × 1.0, PnL = payout - basis
2. PAPER LOSE — payout = 0, PnL = -basis
3. PAPER INVALID — payout = shares × 0.5, PnL = 0.5*shares - basis
4. Идемпотентность — повторный вызов на CLOSED → no-op
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from decimal import Decimal

from polyflip.db.models import TradeHistory
from polyflip.execution.settlement_service import (
    settle_resolved_position,
    AccountingInvariantError,
)


def make_trade(
    db_session,
    *,
    trade_id: int = 1,
    market_id: str = "0xabc",
    asset: str = "ETH",
    outcome_bought: str = "YES",
    mode: str = "PAPER",
    entry_filled_shares: str = "20",
    entry_cost_usdc: str = "10",  # avg price = 0.50
    remaining_shares: str = "20",
    realized_pnl_usdc: str = "0",
    position_accounting_version: int = 1,
    position_status: str = "OPEN",
) -> TradeHistory:
    trade = TradeHistory(
        id=trade_id,
        market_id=market_id,
        asset=asset,
        outcome_bought=outcome_bought,
        amount_usdc=Decimal(entry_cost_usdc),
        executed_price=Decimal("0.5"),
        predicted_flip_prob=0.6,
        active_features="{}",
        status="SUCCESS",
        mode=mode,
        position_status=position_status,
        # version=1 + entry fields переданы явно → constraint OK
        position_accounting_version=position_accounting_version,
        entry_filled_shares=Decimal(entry_filled_shares),
        entry_cost_usdc=Decimal(entry_cost_usdc),
        remaining_shares=Decimal(remaining_shares),
        realized_pnl_usdc=Decimal(realized_pnl_usdc),
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(trade)
    return trade


@pytest.mark.asyncio
async def test_paper_win(db_session):
    """
    PAPER YES позиция, рынок разрешён YES.
    20 акций × 0.50 cost = 10 USDC basis.
    Payout = 20 × 1.0 = 20 USDC.
    PnL = 20 - 10 = +10 USDC.
    """
    trade = make_trade(db_session)
    await db_session.flush()

    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="YES",
        payout_per_share=Decimal("1.0"),
        settlement_fee_usdc=Decimal("0"),
    )

    assert trade.position_status == "CLOSED"
    assert trade.remaining_shares == Decimal("0")
    assert trade.realized_pnl_usdc == Decimal("10")  # 20 - 10


@pytest.mark.asyncio
async def test_paper_lose(db_session):
    """
    PAPER YES позиция, рынок разрешён NO.
    Payout = 0. PnL = -10 USDC (полная потеря basis).
    """
    trade = make_trade(db_session, outcome_bought="YES")
    await db_session.flush()

    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="NO",
        payout_per_share=Decimal("1.0"),
        settlement_fee_usdc=Decimal("0"),
    )

    assert trade.position_status == "CLOSED"
    assert trade.remaining_shares == Decimal("0")
    assert trade.realized_pnl_usdc == Decimal("-10")  # 0 - 10


@pytest.mark.asyncio
async def test_paper_invalid(db_session):
    """
    INVALID рынок — payout = shares × 0.5 = 10 USDC.
    Basis = 10 USDC → PnL = 10 - 10 = 0.
    """
    trade = make_trade(db_session)
    await db_session.flush()

    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="INVALID",
        payout_per_share=Decimal("0.5"),
        settlement_fee_usdc=Decimal("0"),
    )

    assert trade.position_status == "CLOSED"
    assert trade.remaining_shares == Decimal("0")
    assert trade.realized_pnl_usdc == Decimal("0")  # 10 - 10


@pytest.mark.asyncio
async def test_invalid_with_higher_cost(db_session):
    """
    INVALID, куплено по 0.70 (выше 0.50 redemption).
    20 shares, basis = 14 USDC, payout = 20 × 0.5 = 10 USDC → PnL = -4 USDC.
    """
    trade = make_trade(
        db_session,
        entry_cost_usdc="14",  # 20 shares × 0.70
    )
    await db_session.flush()

    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="INVALID",
        payout_per_share=Decimal("0.5"),
        settlement_fee_usdc=Decimal("0"),
    )

    assert trade.position_status == "CLOSED"
    assert trade.realized_pnl_usdc == Decimal("-4")  # 10 - 14


@pytest.mark.asyncio
async def test_idempotent_on_closed(db_session):
    """
    Повторный вызов на уже CLOSED позиции → no-op, PnL не меняется.
    """
    trade = make_trade(
        db_session,
        position_status="CLOSED",
        realized_pnl_usdc="7",
        remaining_shares="0",
    )
    await db_session.flush()

    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="YES",
        payout_per_share=Decimal("1.0"),
        settlement_fee_usdc=Decimal("0"),
    )

    # PnL должен остаться 7, remaining_shares = 0 (no-op)
    assert trade.realized_pnl_usdc == Decimal("7")
    assert trade.position_status == "CLOSED"


@pytest.mark.asyncio
async def test_partial_remaining(db_session):
    """
    Позиция частично закрыта: 20 куплено, 10 продано ранее.
    remaining_shares = 10, entry_cost_usdc = 10 (basis = 0.50/share).
    WIN → payout = 10 × 1.0 = 10, PnL = 10 - 5 = 5.
    """
    trade = make_trade(
        db_session,
        entry_filled_shares="20",
        entry_cost_usdc="10",
        remaining_shares="10",
        realized_pnl_usdc="2",  # уже была частичная реализация
    )
    await db_session.flush()

    await settle_resolved_position(
        db_session,
        trade_id=trade.id,
        winning_outcome="YES",
        payout_per_share=Decimal("1.0"),
        settlement_fee_usdc=Decimal("0"),
    )

    assert trade.position_status == "CLOSED"
    # basis = 0.5 * 10 = 5; payout = 10; delta = +5; total = 2 + 5 = 7
    assert trade.realized_pnl_usdc == Decimal("7")
