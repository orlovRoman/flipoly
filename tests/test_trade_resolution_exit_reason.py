from decimal import Decimal
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from polyflip.db.models import TradeHistory
from polyflip.execution.states import ExitReason
from polyflip.execution.trade_lifecycle import mark_trade_resolved
from polyflip.execution.worker import rebuild_trade_accounting
from polyflip.execution.live_settlement_service import reconcile_live_resolution


def test_mark_trade_resolved_win():
    trade = TradeHistory(
        id=1,
        asset="ETH",
        outcome_bought="YES",
        remaining_shares=Decimal("10.0"),
    )
    mark_trade_resolved(trade, is_win=True)
    assert trade.position_status == "RESOLVED_REDEEMABLE"
    assert trade.exit_reason == ExitReason.SETTLEMENT
    assert trade.redemption_status == "PENDING"
    assert trade.remaining_shares == Decimal("10.0")


def test_mark_trade_resolved_loss():
    trade = TradeHistory(
        id=2,
        asset="ETH",
        outcome_bought="NO",
        remaining_shares=Decimal("10.0"),
    )
    mark_trade_resolved(trade, is_win=False)
    assert trade.position_status == "RESOLVED_LOST"
    assert trade.exit_reason == ExitReason.SETTLEMENT
    assert trade.redemption_status == "NOT_REQUIRED"
    assert trade.remaining_shares == Decimal("0")


@pytest.mark.asyncio
async def test_rebuild_accounting_early_return_on_finalized_trade():
    db_session = AsyncMock()
    for status in ("CLOSED", "RESOLVED_REDEEMABLE", "RESOLVED_LOST"):
        trade = TradeHistory(
            id=100,
            asset="BTC",
            position_status=status,
            exit_reason=ExitReason.SETTLEMENT,
            realized_pnl_usdc=Decimal("5.0"),
            closed_at=datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc),
        )
        exec_mock = MagicMock()
        exec_mock.scalar_one_or_none.return_value = trade
        db_session.execute.return_value = exec_mock

        result = await rebuild_trade_accounting(db_session, 100)
        assert result == trade
        assert trade.exit_reason == ExitReason.SETTLEMENT
        assert trade.position_status == status


@pytest.mark.asyncio
async def test_rebuild_accounting_preserves_settlement_exit_reason_on_open_trade():
    db_session = AsyncMock()
    trade = TradeHistory(
        id=101,
        asset="SOL",
        position_status="OPEN",
        exit_reason=ExitReason.SETTLEMENT,
        remaining_shares=Decimal("5.0"),
    )
    
    exec_mock_trade = MagicMock()
    exec_mock_trade.scalar_one_or_none.return_value = trade
    
    exec_mock_reqs = MagicMock()
    exec_mock_reqs.scalars.return_value.all.return_value = []

    db_session.execute.side_effect = [exec_mock_trade, exec_mock_reqs]

    await rebuild_trade_accounting(db_session, 101)
    assert trade.exit_reason == ExitReason.SETTLEMENT


@pytest.mark.asyncio
async def test_reconcile_live_resolution_sets_exit_reason():
    db_session = AsyncMock()
    trade = TradeHistory(
        id=201,
        mode="LIVE",
        market_id="m123",
        asset="DOGE",
        outcome_bought="YES",
        remaining_shares=Decimal("10.0"),
        entry_cost_usdc=Decimal("2.5"),
        position_status="OPEN",
    )
    db_session.scalar.return_value = trade

    mock_resolution = MagicMock()
    mock_resolution.final_outcome = "YES"

    with patch("polyflip.execution.live_settlement_service.fetch_polymarket_market", return_value={}), \
         patch("polyflip.execution.live_settlement_service.refresh_market_trading_state", return_value=None), \
         patch("polyflip.execution.live_settlement_service.parse_confirmed_resolution", return_value=mock_resolution), \
         patch("polyflip.execution.live_settlement_service.save_market_resolution", return_value=None):

        result = await reconcile_live_resolution(db_session, 201)
        assert result.exit_reason == ExitReason.SETTLEMENT
        assert result.position_status == "RESOLVED_REDEEMABLE"
        assert result.redemption_status == "PENDING"
