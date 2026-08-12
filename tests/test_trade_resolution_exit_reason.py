from decimal import Decimal
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.states import ExitReason
from polyflip.execution.trade_lifecycle import mark_trade_resolved
from polyflip.execution.worker import rebuild_trade_accounting
from polyflip.execution.live_settlement_service import reconcile_live_resolution


def _make_trade(**kwargs) -> TradeHistory:
    defaults = {
        "market_id": "m123",
        "asset": "BTC",
        "outcome_bought": "YES",
        "amount_usdc": 10.0,
        "executed_price": 0.50,
        "predicted_flip_prob": 0.50,
        "status": "SUCCESS",
        "active_features": "test_features",
        "model_version": 1,
        "mode": "PAPER",
        "position_status": "OPEN",
        "remaining_shares": Decimal("10.0"),
        "entry_filled_shares": Decimal("10.0"),
        "entry_cost_usdc": Decimal("5.0"),
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return TradeHistory(**defaults)


def _make_req(trade_id: int, asset: str, **kwargs) -> ExecutionRequest:
    now = datetime.now(timezone.utc)
    defaults = {
        "trade_history_id": trade_id,
        "market_id": "m123",
        "asset": asset,
        "outcome_to_buy": "YES",
        "target_amount_usdc": 5.0,
        "requested_shares": Decimal("10.0"),
        "max_slippage_pct": 1.0,
        "intent": "OPEN",
        "requested_mode": "PAPER",
        "state": "FILLED",
        "created_at": now,
        "updated_at": now,
    }
    defaults.update(kwargs)
    return ExecutionRequest(**defaults)


def test_mark_trade_resolved_win():
    trade = _make_trade(asset="ETH", outcome_bought="YES", remaining_shares=Decimal("10.0"))
    mark_trade_resolved(trade, is_win=True)
    assert trade.position_status == "RESOLVED_REDEEMABLE"
    assert trade.exit_reason == ExitReason.SETTLEMENT
    assert trade.redemption_status == "PENDING"
    assert trade.remaining_shares == Decimal("10.0")


def test_mark_trade_resolved_loss():
    trade = _make_trade(asset="ETH", outcome_bought="NO", remaining_shares=Decimal("10.0"))
    mark_trade_resolved(trade, is_win=False)
    assert trade.position_status == "RESOLVED_LOST"
    assert trade.exit_reason == ExitReason.SETTLEMENT
    assert trade.redemption_status == "NOT_REQUIRED"
    assert trade.remaining_shares == Decimal("0")


def test_mark_trade_resolved_does_not_trigger_warning(caplog):
    trade = _make_trade(asset="BTC", remaining_shares=Decimal("5.0"))
    with caplog.at_level("WARNING"):
        mark_trade_resolved(trade, is_win=True)
    assert "trade_resolved_without_exit_reason" not in caplog.text


def test_invalid_exit_reason_raises_value_error():
    trade = _make_trade(asset="SOL")
    with pytest.raises(ValueError, match="Invalid exit_reason"):
        trade.exit_reason = "SETLLEMENT_TYPO"


@pytest.mark.asyncio
async def test_rebuild_accounting_early_return_on_finalized_trade(db_session):
    for status in ("CLOSED", "RESOLVED_REDEEMABLE", "RESOLVED_LOST"):
        trade = _make_trade(
            asset="BTC",
            position_status=status,
            exit_reason=ExitReason.SETTLEMENT,
            realized_pnl_usdc=Decimal("5.0"),
            closed_at=datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc),
        )
        db_session.add(trade)
        await db_session.commit()

        result = await rebuild_trade_accounting(db_session, trade.id)
        assert result is not None
        assert result.id == trade.id
        assert result.exit_reason == ExitReason.SETTLEMENT
        assert result.position_status == status


@pytest.mark.asyncio
async def test_rebuild_accounting_preserves_settlement_exit_reason_real_db(db_session):
    trade = _make_trade(
        asset="SOL",
        position_status="OPEN",
        exit_reason=ExitReason.SETTLEMENT,
        remaining_shares=Decimal("5.0"),
        entry_filled_shares=Decimal("5.0"),
        entry_cost_usdc=Decimal("1.25"),
        executed_price=0.25,
    )
    db_session.add(trade)
    await db_session.flush()

    req = _make_req(trade.id, trade.asset)
    db_session.add(req)
    await db_session.commit()

    result = await rebuild_trade_accounting(db_session, trade.id)
    assert result is not None
    assert trade.exit_reason == ExitReason.SETTLEMENT
    assert trade.position_status == "OPEN"


@pytest.mark.asyncio
async def test_rebuild_accounting_normal_path_returns_trade(db_session):
    trade = _make_trade(
        asset="ETH",
        position_status="OPEN",
        exit_reason=None,
        remaining_shares=Decimal("10.0"),
        entry_filled_shares=Decimal("10.0"),
        entry_cost_usdc=Decimal("5.0"),
        executed_price=0.50,
    )
    db_session.add(trade)
    await db_session.flush()

    req = _make_req(trade.id, trade.asset)
    db_session.add(req)
    await db_session.commit()

    result = await rebuild_trade_accounting(db_session, trade.id)
    assert result is not None
    assert result.id == trade.id


@pytest.mark.asyncio
async def test_reconcile_live_resolution_sets_exit_reason_on_win(db_session):
    trade = _make_trade(
        mode="LIVE",
        market_id="m_win_123",
        asset="DOGE",
        outcome_bought="YES",
        remaining_shares=Decimal("10.0"),
        entry_cost_usdc=Decimal("2.5"),
        position_status="OPEN",
    )
    db_session.add(trade)
    await db_session.commit()

    mock_resolution = MagicMock()
    mock_resolution.final_outcome = "YES"

    with patch("polyflip.execution.live_settlement_service.fetch_polymarket_market", return_value={}), \
         patch("polyflip.execution.live_settlement_service.refresh_market_trading_state", return_value=None), \
         patch("polyflip.execution.live_settlement_service.parse_confirmed_resolution", return_value=mock_resolution), \
         patch("polyflip.execution.live_settlement_service.save_market_resolution", return_value=None):

        result = await reconcile_live_resolution(db_session, trade.id)
        await db_session.flush()
        assert result.exit_reason == ExitReason.SETTLEMENT
        assert result.position_status == "RESOLVED_REDEEMABLE"
        assert result.redemption_status == "PENDING"


@pytest.mark.asyncio
async def test_reconcile_live_resolution_sets_exit_reason_on_loss(db_session):
    trade = _make_trade(
        mode="LIVE",
        market_id="m_loss_456",
        asset="DOGE",
        outcome_bought="YES",
        remaining_shares=Decimal("10.0"),
        entry_cost_usdc=Decimal("2.5"),
        position_status="OPEN",
    )
    db_session.add(trade)
    await db_session.commit()

    mock_resolution = MagicMock()
    mock_resolution.final_outcome = "NO"

    with patch("polyflip.execution.live_settlement_service.fetch_polymarket_market", return_value={}), \
         patch("polyflip.execution.live_settlement_service.refresh_market_trading_state", return_value=None), \
         patch("polyflip.execution.live_settlement_service.parse_confirmed_resolution", return_value=mock_resolution), \
         patch("polyflip.execution.live_settlement_service.save_market_resolution", return_value=None):

        result = await reconcile_live_resolution(db_session, trade.id)
        await db_session.flush()
        assert result.exit_reason == ExitReason.SETTLEMENT
        assert result.position_status == "RESOLVED_LOST"
        assert result.redemption_status == "NOT_REQUIRED"
