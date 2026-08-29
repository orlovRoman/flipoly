from datetime import datetime, timedelta, timezone
from decimal import Decimal
import uuid

import pytest
from pydantic import ValidationError

from polyflip.api.execution_api import (
    CreateLiveSessionRequest,
    UpdateLiveSessionLimitsRequest,
    update_live_session_limits,
)
from polyflip.db.execution_models import (
    ExecutionRequest,
    LiveMirrorCandidate,
    LiveTradingSession,
)
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.execution.assets import (
    LIVE_TRADING_ASSETS,
    normalize_live_asset,
    normalize_live_assets,
)
from polyflip.execution.live_mirror_worker import _build_signal_snapshot, _compute_hash
from polyflip.execution.release_gate import ReleaseDeferred, validate_live_release


def test_live_asset_normalization_is_canonical_and_ordered():
    assert normalize_live_asset("btcusdt") == "BTC"
    assert normalize_live_assets(["doge", "BTC", "doge"]) == ["BTC", "DOGE"]
    assert normalize_live_assets(None) == list(LIVE_TRADING_ASSETS)


def test_live_session_request_rejects_empty_or_unknown_assets():
    kwargs = {
        "budget_usdc": Decimal("10"),
        "max_single_order_usdc": Decimal("1.10"),
        "max_open_positions": 3,
        "max_total_exposure_usdc": Decimal("10"),
    }

    with pytest.raises(ValidationError, match="хотя бы один"):
        CreateLiveSessionRequest(**kwargs, selected_assets=[])

    with pytest.raises(ValidationError, match="Недопустимый актив"):
        CreateLiveSessionRequest(**kwargs, selected_assets=["ADA"])


@pytest.mark.asyncio
async def test_update_live_session_persists_selected_assets(db_session):
    now = datetime.now(timezone.utc)
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="DRAFT",
        budget_usdc=Decimal("10"),
        reserved_usdc=Decimal("0"),
        filled_usdc=Decimal("0"),
        max_single_order_usdc=Decimal("2"),
        max_total_exposure_usdc=Decimal("10"),
        max_open_positions=3,
        selected_assets=["BTC", "ETH"],
        created_at=now,
    )
    db_session.add(session_obj)
    await db_session.commit()

    result = await update_live_session_limits(
        str(session_obj.id),
        UpdateLiveSessionLimitsRequest(selected_assets=["sol"]),
        db_session,
    )

    assert result["selected_assets"] == ["SOL"]
    await db_session.refresh(session_obj)
    assert session_obj.selected_assets == ["SOL"]


@pytest.mark.asyncio
async def test_release_gate_defers_unselected_live_asset(db_session):
    now = datetime.now(timezone.utc)
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="ACTIVE",
        budget_usdc=Decimal("10"),
        reserved_usdc=Decimal("0"),
        filled_usdc=Decimal("0"),
        max_single_order_usdc=Decimal("2"),
        max_total_exposure_usdc=Decimal("10"),
        max_open_positions=3,
        selected_assets=["BTC"],
        started_at=now,
        created_at=now,
    )
    db_session.add(session_obj)
    db_session.add(
        RuntimeSettings(
            key="LIVE_TRADING_ENABLED",
            value="true",
            updated_at=now,
            updated_by="test",
        )
    )
    trade = TradeHistory(
        market_id="asset-filter-test",
        asset="ETH",
        outcome_bought="YES",
        amount_usdc=5.0,
        executed_price=0.5,
        predicted_flip_prob=0.7,
        active_features="f1",
        status="SUCCESS",
        mode="PAPER",
        position_status="OPEN",
        entry_filled_shares=Decimal("10"),
        entry_cost_usdc=Decimal("5"),
        remaining_shares=Decimal("10"),
        realized_pnl_usdc=Decimal("0"),
        market_end_time=now + timedelta(hours=1),
        created_at=now,
    )
    db_session.add(trade)
    await db_session.flush()
    paper_request = ExecutionRequest(
        id=uuid.uuid4(),
        idempotency_key=f"asset-filter-{uuid.uuid4()}",
        requested_mode="PAPER",
        trade_history_id=trade.id,
        intent="OPEN",
        trigger_reason="STRATEGY",
        market_id=trade.market_id,
        asset="ETH",
        outcome_to_buy="YES",
        target_amount_usdc=Decimal("5"),
        requested_shares=Decimal("10"),
        limit_price=Decimal("0.5"),
        max_slippage_pct=0.02,
        max_spend_usdc=Decimal("5"),
        state="FILLED",
        created_at=now,
        updated_at=now,
    )
    db_session.add(paper_request)
    await db_session.flush()
    candidate = LiveMirrorCandidate(
        id=uuid.uuid4(),
        source_paper_request_id=paper_request.id,
        source_paper_trade_id=trade.id,
        target_mode="LIVE",
        state="ELIGIBLE",
        signal_snapshot=_build_signal_snapshot(paper_request, trade),
        signal_hash="",
        created_at=now,
    )
    candidate.signal_hash = _compute_hash(candidate.signal_snapshot)
    db_session.add(candidate)
    await db_session.commit()

    with pytest.raises(ReleaseDeferred, match="not selected"):
        await validate_live_release(
            db_session,
            candidate,
            paper_request,
            trade,
            "LIVE",
            fresh_prices={"best_ask": 0.5, "best_bid": 0.5},
        )
