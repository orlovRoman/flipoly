"""
tests/collector/test_parser_observations.py

Tests for parser.py collector cycle integration with ObservationWriter:
1. Empty active markets cycle completes cleanly without UnboundLocalError.
2. Markets with pricing errors / skipped markets complete cleanly.
3. Successful cycle records strikes and flushes pending observations.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.collector.client import StrikeProvenance
from polyflip.collector.parser import run_collector_cycle
from polyflip.crypto.underlying_observations import get_observation_writer, ObservationRepository
from polyflip.db.models import CollectorStatus, LiveMarket, MarketSnapshot


@pytest.mark.asyncio
async def test_run_collector_cycle_empty_active_markets(db_session: AsyncSession):
    """Empty active markets must not cause UnboundLocalError and must record status='success'."""
    with patch("polyflip.collector.parser.PolymarketClient") as MockClient:
        instance = AsyncMock()
        instance.get_active_15m_markets.return_value = []
        MockClient.return_value = instance

        await run_collector_cycle(db_session)

    # Check collector status recorded
    res = await db_session.execute(select(CollectorStatus).order_by(CollectorStatus.id.desc()).limit(1))
    status_row = res.scalar_one_or_none()
    assert status_row is not None
    assert status_row.status == "success"
    assert status_row.markets_found == 0
    assert status_row.markets_saved == 0
    assert status_row.error_message is None


@pytest.mark.asyncio
async def test_run_collector_cycle_skipped_markets_no_error(db_session: AsyncSession):
    """If all active markets have pricing errors or are skipped, cycle completes cleanly."""
    now = datetime.now(timezone.utc)
    mock_market = {
        "market_id": "test_m_1",
        "yes_token_id": "tok_1",
        "no_token_id": "tok_2",
        "asset": "BTC",
        "question": "Bitcoin Up or Down",
        "end_date_iso": (now + timedelta(minutes=15)).isoformat(),
    }
    with patch("polyflip.collector.parser.PolymarketClient") as MockClient:
        instance = AsyncMock()
        instance.get_active_15m_markets.return_value = [mock_market]
        # Pricing returns error
        instance.get_market_prices.return_value = {"error": "rate_limited"}
        MockClient.return_value = instance

        await run_collector_cycle(db_session)

    res = await db_session.execute(select(CollectorStatus).order_by(CollectorStatus.id.desc()).limit(1))
    status_row = res.scalar_one_or_none()
    assert status_row is not None
    assert status_row.status == "success"
    assert status_row.markets_found == 1
    assert status_row.markets_saved == 0


@pytest.mark.asyncio
async def test_run_collector_cycle_records_and_flushes_strike(db_session: AsyncSession):
    """Active market cycle records strike in ObservationWriter and flushes to database."""
    now = datetime.now(timezone.utc)
    mock_market = {
        "market_id": "test_m_btc",
        "yes_token_id": "tok_yes_btc",
        "no_token_id": "tok_no_btc",
        "asset": "BTC",
        "question": "Bitcoin Up or Down",
        "end_date_iso": (now + timedelta(minutes=15)).isoformat(),
        "underlying_price": 68500.0,
        "strike_provenance": StrikeProvenance(
            strike_value=68500.0,
            strike_source="market.strikePrice",
            strike_effective_at=now,
            strike_received_at=now,
        ),
    }

    mock_prices = {
        "current_yes_price": 0.52,
        "current_no_price": 0.48,
        "current_spread": 0.02,
        "best_bid": 0.51,
        "best_ask": 0.53,
        "best_bid_no": 0.47,
        "best_ask_no": 0.49,
    }

    with patch("polyflip.collector.parser.PolymarketClient") as MockClient:
        instance = AsyncMock()
        instance.get_active_15m_markets.return_value = [mock_market]
        instance.get_market_prices.return_value = mock_prices
        instance.get_recent_trades_volume.return_value = 1500.0
        MockClient.return_value = instance

        await run_collector_cycle(db_session)

    snapshot_res = await db_session.execute(
        select(MarketSnapshot).where(MarketSnapshot.market_id == "test_m_btc")
    )
    snapshot = snapshot_res.scalar_one()
    assert snapshot.poly_up_mid == 0.52
    assert snapshot.poly_down_mid == 0.48
    assert snapshot.poly_up_best_bid == 0.51
    assert snapshot.poly_down_best_ask == 0.49

    # Check that strike was flushed and is in underlying_observations
    repo = ObservationRepository(db_session)
    obs_res = await repo.get_latest_observation(instrument="BTC", source="ORACLE", as_of=now)
    assert obs_res.is_valid
    assert obs_res.price == 68500.0


@pytest.mark.asyncio
async def test_run_collector_cycle_one_sided_orderbook(db_session: AsyncSession):
    """One-sided orderbook correctly saves MarketSnapshot with nullable mid_price/spread and links depth."""
    now = datetime.now(timezone.utc)
    mock_market = {
        "market_id": "test_m_oneside",
        "yes_token_id": "tok_yes",
        "no_token_id": "tok_no",
        "asset": "BTC",
        "question": "One sided",
        "end_date_iso": (now + timedelta(minutes=15)).isoformat(),
    }
    
    # Mocking orderbook structure
    class MockOrderbook:
        def __init__(self, side):
            self.event_at = now
            self.received_at = now
            self.bids = [] if side == "NO" else [{"price": 0.05, "size": 100}]
            self.asks = [{"price": 0.05, "size": 100}] if side == "NO" else []
            self.sequence_id = "1"
            self.is_truncated = False
            self.depth_limit = 100
            self.source = "CLOB"
            self.quality_status = "VALID"
            self.quality_notes = None
            self.best_bid_price = None if side == "NO" else 0.05
            self.best_bid_size = None if side == "NO" else 100
            self.best_ask_price = 0.05 if side == "NO" else None
            self.best_ask_size = 100 if side == "NO" else None
            self.depth_usdc_bid = 0.0 if side == "NO" else 5.0
            self.depth_usdc_ask = 5.0 if side == "NO" else 0.0

    # Missing mid_price/spread due to empty side (e.g. YES has no bids, NO has no asks)
    mock_prices = {
        "current_yes_price": None,
        "current_no_price": None,
        "current_spread": None,
        "best_bid": None,
        "best_ask": 0.05,
        "yes_orderbook": MockOrderbook("YES"),
        "no_orderbook": MockOrderbook("NO"),
    }

    with patch("polyflip.collector.parser.PolymarketClient") as MockClient:
        instance = AsyncMock()
        instance.get_active_15m_markets.return_value = [mock_market]
        instance.get_market_prices.return_value = mock_prices
        instance.get_recent_trades_volume.return_value = 100.0
        MockClient.return_value = instance

        await run_collector_cycle(db_session)

    # Verify MarketSnapshot IS created with nullable mid_price and spread (Point 6/7/P0 fix)
    from polyflip.db.models import MarketSnapshot, OrderbookDepthSnapshot
    res = await db_session.execute(select(MarketSnapshot).where(MarketSnapshot.market_id == "test_m_oneside"))
    snapshot = res.scalar_one_or_none()
    assert snapshot is not None
    assert snapshot.mid_price is None
    assert snapshot.spread is None

    # Verify OrderbookDepthSnapshot is created and linked to snapshot.id
    res = await db_session.execute(select(OrderbookDepthSnapshot).where(OrderbookDepthSnapshot.market_id == "test_m_oneside"))
    depths = res.scalars().all()
    assert len(depths) == 2
    assert depths[0].snapshot_id == snapshot.id
    assert depths[1].snapshot_id == snapshot.id


@pytest.mark.asyncio
async def test_client_get_both_orderbooks_causal_pairing():
    """Client get_both_orderbooks enforces directed causal rule:
    1. 0 <= delta <= 5.0 -> VALID
    2. delta > 5.0 -> CAUSAL_PAIR_TIMEOUT
    3. delta < 0.0 -> NON_CAUSAL_PAIR
    """
    from polyflip.collector.client import PolymarketClient
    from polyflip.collector.orderbook_depth import OrderbookContract

    client = PolymarketClient()
    t0 = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Valid pairing within [0, 5]s
    ob_yes = OrderbookContract(
        market_id="m1", token_id="t1", outcome_side="YES",
        event_at=t0, received_at=t0, bids=[], asks=[], quality_status="VALID"
    )
    ob_no = OrderbookContract(
        market_id="m1", token_id="t2", outcome_side="NO",
        event_at=t0, received_at=t0 + timedelta(seconds=2), bids=[], asks=[], quality_status="VALID"
    )
    with patch.object(client, "get_single_orderbook", side_effect=[ob_yes, ob_no]):
        y, n = await client.get_both_orderbooks("m1", "t1", "t2")
        assert n.quality_status == "VALID"

    # 2. Timeout: NO received 6s after YES (> 5.0s) -> CAUSAL_PAIR_TIMEOUT
    ob_no_late = OrderbookContract(
        market_id="m1", token_id="t2", outcome_side="NO",
        event_at=t0, received_at=t0 + timedelta(seconds=6), bids=[], asks=[], quality_status="VALID"
    )
    with patch.object(client, "get_single_orderbook", side_effect=[ob_yes, ob_no_late]):
        y, n = await client.get_both_orderbooks("m1", "t1", "t2")
        assert n.quality_status == "CAUSAL_PAIR_TIMEOUT"
        assert "exceeded 5s" in n.quality_notes

    # 3. Non-causal: NO received before YES (< 0.0s) -> NON_CAUSAL_PAIR
    ob_no_past = OrderbookContract(
        market_id="m1", token_id="t2", outcome_side="NO",
        event_at=t0, received_at=t0 - timedelta(seconds=1), bids=[], asks=[], quality_status="VALID"
    )
    with patch.object(client, "get_single_orderbook", side_effect=[ob_yes, ob_no_past]):
        y, n = await client.get_both_orderbooks("m1", "t1", "t2")
        assert n.quality_status == "NON_CAUSAL_PAIR"
        assert "violates directed causality" in n.quality_notes
