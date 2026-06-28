import pytest
import httpx
import respx
from sqlalchemy import select
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from polyflip.db.models import MarketSnapshot, TradeHistory
from polyflip.collector.resolver import resolve_pending_markets
from polyflip.scheduler.jobs import resolve_trades_job

@pytest.mark.asyncio
@respx.mock
async def test_resolver_skips_not_closed(db_session):
    snap = MarketSnapshot(
        market_id="test_m1", asset="BTC", time_left_min=10.0, mid_price=0.8,
        spread=0.01, volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
        final_outcome="PENDING", recorded_at=datetime.now(timezone.utc), flip_vs_final=False
    )
    db_session.add(snap)
    await db_session.commit()

    respx.get("https://gamma-api.polymarket.com/markets/test_m1").mock(
        return_value=httpx.Response(200, json={"closed": False})
    )

    await resolve_pending_markets(db_session)

    res = await db_session.execute(select(MarketSnapshot).where(MarketSnapshot.market_id == "test_m1"))
    updated_snap = res.scalar_one()
    assert updated_snap.final_outcome == "PENDING"

@pytest.mark.asyncio
@respx.mock
async def test_resolver_marks_yes_and_calculates_flip(db_session):
    # Market believed YES (0.8), but actually NO -> should be a flip
    snap1 = MarketSnapshot(
        market_id="test_m2", asset="BTC", time_left_min=5.0, mid_price=0.8,
        spread=0.01, volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
        final_outcome="PENDING", recorded_at=datetime.now(timezone.utc), flip_vs_final=False
    )
    # Market believed NO (0.2), actually NO -> not a flip
    snap2 = MarketSnapshot(
        market_id="test_m2", asset="BTC", time_left_min=2.0, mid_price=0.2,
        spread=0.01, volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
        final_outcome="PENDING", recorded_at=datetime.now(timezone.utc), flip_vs_final=False
    )
    db_session.add_all([snap1, snap2])
    await db_session.commit()

    respx.get("https://gamma-api.polymarket.com/markets/test_m2").mock(
        return_value=httpx.Response(200, json={
            "closed": True,
            "answer": "NO"
        })
    )

    await resolve_pending_markets(db_session)

    res = await db_session.execute(select(MarketSnapshot).where(MarketSnapshot.market_id == "test_m2").order_by(MarketSnapshot.id))
    snaps = res.scalars().all()
    
    assert snaps[0].final_outcome == "NO"
    assert snaps[0].flip_vs_final is True # 0.8 > 0.5 vs NO

    assert snaps[1].final_outcome == "NO"
    assert snaps[1].flip_vs_final is False # 0.2 < 0.5 vs NO


class AsyncSessionContext:
    def __init__(self, session):
        self.session = session
    async def __aenter__(self):
        return self.session
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_resolve_trades_job_invalid(db_session):
    # Создаем успешную сделку SUCCESS на YES без PnL
    trade = TradeHistory(
        market_id="test_m_inv", asset="BTC", outcome_bought="YES",
        amount_usdc=10.0, executed_price=0.5, predicted_flip_prob=0.1,
        active_features="mid_price", status="SUCCESS", pnl=None,
        created_at=datetime.now(timezone.utc)
    )
    # Создаем снепшот с исходом INVALID
    snap = MarketSnapshot(
        market_id="test_m_inv", asset="BTC", time_left_min=10.0, mid_price=0.5,
        spread=0.01, volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
        final_outcome="INVALID", recorded_at=datetime.now(timezone.utc), flip_vs_final=False
    )
    db_session.add_all([trade, snap])
    await db_session.commit()

    mock_session_fn = MagicMock(return_value=AsyncSessionContext(db_session))
    
    with patch("polyflip.scheduler.jobs.async_session", mock_session_fn):
        await resolve_trades_job()

    res = await db_session.execute(select(TradeHistory).where(TradeHistory.market_id == "test_m_inv"))
    updated_trade = res.scalar_one()
    assert updated_trade.pnl == 0.0
    assert updated_trade.status == "INVALID"
