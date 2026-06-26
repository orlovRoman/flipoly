import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy import select
from datetime import datetime, timezone
from polyflip.db.models import MarketSnapshot
from polyflip.collector.resolver import resolve_pending_markets

@pytest.fixture
def mock_httpx_client():
    mock_client_instance = AsyncMock()
    mock_client_context = AsyncMock()
    mock_client_context.__aenter__.return_value = mock_client_instance
    
    with patch('httpx.AsyncClient', return_value=mock_client_context) as mock_client_class:
        yield mock_client_instance

@pytest.mark.asyncio
async def test_resolver_skips_not_closed(db_session, mock_httpx_client):
    snap = MarketSnapshot(
        market_id="test_m1", asset="BTC", time_left_min=10.0, mid_price=0.8,
        spread=0.01, volume_5min=100.0, price_velocity=0.0, hour_of_day=12,
        final_outcome="PENDING", recorded_at=datetime.now(timezone.utc), flip_vs_final=False
    )
    db_session.add(snap)
    await db_session.commit()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"closed": False}
    mock_httpx_client.get.return_value = mock_response

    await resolve_pending_markets(db_session)

    res = await db_session.execute(select(MarketSnapshot).where(MarketSnapshot.market_id == "test_m1"))
    updated_snap = res.scalar_one()
    assert updated_snap.final_outcome == "PENDING"

@pytest.mark.asyncio
async def test_resolver_marks_yes_and_calculates_flip(db_session, mock_httpx_client):
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

    mock_response = MagicMock()
    mock_response.status_code = 200
    # Provide explicit answer
    mock_response.json.return_value = {"closed": True, "answer": "NO"}
    mock_httpx_client.get.return_value = mock_response

    await resolve_pending_markets(db_session)

    res = await db_session.execute(select(MarketSnapshot).where(MarketSnapshot.market_id == "test_m2").order_by(MarketSnapshot.id))
    snaps = res.scalars().all()
    
    assert snaps[0].final_outcome == "NO"
    assert snaps[0].flip_vs_final is True # 0.8 > 0.5 vs NO

    assert snaps[1].final_outcome == "NO"
    assert snaps[1].flip_vs_final is False # 0.2 < 0.5 vs NO
