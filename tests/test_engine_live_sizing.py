import pytest
import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from sqlalchemy import select
from polyflip.db.models import TradeHistory
from polyflip.db.execution_models import ExecutionRequest, ExposureReservation
from polyflip.execution.outbox import finalize_request
from polyflip.execution.worker import rebuild_trade_accounting
from polyflip.execution.gateways.exceptions import GatewayOrderRejected

@pytest.mark.asyncio
async def test_live_sizing_rejection_chain(db_session):
    """
    Тестирует цепочку отказа:
    1. Создаем заявку OPEN и резервируем средства.
    2. Вызываем finalize_request(REJECTED), как делает worker при GatewayOrderRejected.
    3. Проверяем, что резервы сняты, а позиция перешла в ENTRY_FAILED (и не стала OPEN).
    4. Вызываем rebuild_trade_accounting, проверяем, что статус сохранился.
    """
    trade = TradeHistory(
        id=999,
        asset="BTCUSDT",
        amount_usdc=Decimal("10.00"),
        executed_price=Decimal("0.50"),
        predicted_flip_prob=Decimal("0.60"),
        mode="LIVE",
        active_features="{}",
        model_version="1.0",
        model_key="test_key",
        status="PENDING",
        position_status="OPENING",
        market_id="test_market",
        outcome_bought="Yes",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(trade)
    await db_session.commit()

    req = ExecutionRequest(
        trade_history_id=trade.id,
        intent="OPEN",
        target_amount_usdc=Decimal("10.00"),
        state="SUBMITTED",
        filled_shares=Decimal("0.0"),
        market_id="test_market",
        asset="BTCUSDT",
        outcome_to_buy="Yes",
        requested_mode="LIVE",
        max_slippage_pct=Decimal("0.02"),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db_session.add(req)
    await db_session.commit()

    reservation = ExposureReservation(
        trade_history_id=trade.id,
        market_id="test_market",
        request_id=req.id,
        amount_usdc=Decimal("10.00"),
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc)
    )
    db_session.add(reservation)
    await db_session.commit()

    # Имитация исключения GatewayOrderRejected и обработки в worker.py
    await finalize_request(
        db_session,
        req,
        state="REJECTED",
        error="minimum order size"
    )
    await db_session.commit()

    # Проверки после finalize_request
    updated_req = await db_session.get(ExecutionRequest, req.id)
    assert updated_req.state == "REJECTED"
    assert "minimum order size" in updated_req.error_reason

    # Проверка снятия резерва
    await db_session.refresh(reservation)
    res = reservation
    assert res.released_at is not None

    # Проверка TradeHistory после finalize_request
    updated_trade = await db_session.get(TradeHistory, trade.id)
    assert updated_trade.status == "FAILED"
    assert updated_trade.position_status == "ENTRY_FAILED"

    # Вызов rebuild_trade_accounting (чтобы убедиться, что P0.3 работает и он не сбрасывает ENTRY_FAILED)
    await rebuild_trade_accounting(db_session, trade.id)
    await db_session.commit()

    final_trade = await db_session.get(TradeHistory, trade.id)
    # По логике P0.3: если open_shares == 0, и trade.status != "FAILED" / "ENTRY_FAILED", он ставит PENDING/OPENING.
    # Но так как он уже FAILED/ENTRY_FAILED, он должен остаться FAILED/ENTRY_FAILED.
    assert final_trade.status == "FAILED"
    assert final_trade.position_status == "ENTRY_FAILED"
