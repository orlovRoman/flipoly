import pytest
from datetime import datetime, timezone
from sqlalchemy import select, func
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.db.execution_models import ExecutionRequest
from polyflip.execution.outbox import enqueue_open_request, EnqueueDisposition
from polyflip.execution.config import ExecutionMode
from polyflip.execution.risk_checks import check_risk_limits
from decimal import Decimal


@pytest.mark.asyncio
async def test_live_disabled_does_not_change_paper_flow(db_session):
    """Когда LIVE выключен (LIVE_TRADING_ENABLED=false), поведение PAPER не меняется."""
    # 1. Отключаем LIVE
    setting = await db_session.scalar(
        select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_TRADING_ENABLED")
    )
    if setting:
        setting.value = "false"
        setting.updated_at = datetime.now(timezone.utc)
        setting.updated_by = "test"
    else:
        db_session.add(
            RuntimeSettings(
                key="LIVE_TRADING_ENABLED",
                value="false",
                updated_at=datetime.now(timezone.utc),
                updated_by="test",
            )
        )
    await db_session.commit()

    # 2. Создаем тестовую запись TradeHistory
    trade = TradeHistory(
        market_id="PAPER-ISOLATION-TEST",
        asset="BTC",
        outcome_bought="YES",
        amount_usdc=1.0,
        executed_price=0.5,
        predicted_flip_prob=0.4,
        active_features="test",
        status="PENDING",
        mode="PAPER",
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(trade)
    await db_session.commit()
    await db_session.refresh(trade)

    # 3. Отправляем PAPER-запрос
    res = await enqueue_open_request(
        db_session,
        trade_id=trade.id,
        market_id="PAPER-ISOLATION-TEST",
        asset="BTC",
        outcome_to_buy="YES",
        target_amount_usdc=1.0,
        limit_price=0.5,
        requested_mode=ExecutionMode.PAPER,
    )

    assert res.disposition == EnqueueDisposition.CREATED
    assert res.request_id is not None

    # 4. Проверяем, что создался ровно один PAPER запрос в состоянии READY
    paper_requests = (
        await db_session.scalars(
            select(ExecutionRequest).where(ExecutionRequest.requested_mode == "PAPER")
        )
    ).all()
    live_requests = (
        await db_session.scalars(
            select(ExecutionRequest).where(ExecutionRequest.requested_mode == "LIVE")
        )
    ).all()

    assert len(paper_requests) == 1
    assert len(live_requests) == 0
    assert paper_requests[0].state == "READY"

    # 5. Проверяем, что check_risk_limits для PAPER возвращает None (лимиты не блокируют PAPER)
    risk_err = await check_risk_limits(
        db_session,
        intent="OPEN",
        max_spend_usdc=Decimal("10.0"),
        requested_mode="PAPER",
    )
    assert risk_err is None
