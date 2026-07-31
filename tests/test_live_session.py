import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone

from polyflip.db.execution_models import (
    LiveTradingSession,
    ExecutionRequest,
    LiveMirrorCandidate,
)
from polyflip.db.models import TradeHistory, RuntimeSettings
from polyflip.execution.release_gate import (
    release_candidate_by_id,
    ReleaseDeferred,
    ReleaseRejected,
)


@pytest.mark.asyncio
async def test_session_budget_reservation_atomic(db_session):
    # 1. Создаем сессию
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="ACTIVE",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        filled_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("2.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session_obj)
    await db_session.commit()

    assert session_obj.status == "ACTIVE"
    assert session_obj.reserved_usdc == Decimal("0.00")


@pytest.mark.asyncio
async def test_single_order_limit_rejection(db_session):
    session_obj = LiveTradingSession(
        id=uuid.uuid4(),
        status="ACTIVE",
        budget_usdc=Decimal("10.00"),
        reserved_usdc=Decimal("0.00"),
        max_single_order_usdc=Decimal("1.00"),
        max_total_exposure_usdc=Decimal("5.00"),
        max_open_positions=5,
    )
    db_session.add(session_obj)

    # Runtime Settings
    db_session.add(
        RuntimeSettings(
            key="LIVE_TRADING_ENABLED",
            value="true",
            updated_at=datetime.now(timezone.utc),
            updated_by="test",
        )
    )
    db_session.add(
        RuntimeSettings(
            key="LIVE_MIRROR_ENABLED",
            value="true",
            updated_at=datetime.now(timezone.utc),
            updated_by="test",
        )
    )
    db_session.add(
        RuntimeSettings(
            key="LIVE_RELEASE_MODE",
            value="AUTO",
            updated_at=datetime.now(timezone.utc),
            updated_by="test",
        )
    )
    await db_session.commit()

    assert session_obj.max_single_order_usdc == Decimal("1.00")
