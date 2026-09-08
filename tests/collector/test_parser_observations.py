"""
tests/collector/test_parser_observations.py

Tests for parser.py collector cycle integration with ObservationWriter:
1. Empty active markets cycle completes cleanly without UnboundLocalError.
2. Markets with pricing errors / skipped markets complete cleanly.
3. Successful cycle records strikes and flushes pending observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
    mock_market = {
        "market_id": "test_m_1",
        "yes_token_id": "tok_1",
        "no_token_id": "tok_2",
        "asset": "BTC",
        "question": "Bitcoin Up or Down",
        "end_date_iso": "2026-09-08T23:59:59Z",
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
        "end_date_iso": "2026-09-08T23:59:59Z",
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
    }

    with patch("polyflip.collector.parser.PolymarketClient") as MockClient:
        instance = AsyncMock()
        instance.get_active_15m_markets.return_value = [mock_market]
        instance.get_market_prices.return_value = mock_prices
        instance.get_recent_trades_volume.return_value = 1500.0
        MockClient.return_value = instance

        await run_collector_cycle(db_session)

    # Check that strike was flushed and is in underlying_observations
    repo = ObservationRepository(db_session)
    obs_res = await repo.get_latest_observation(instrument="BTC", source="ORACLE", as_of=now)
    assert obs_res.is_valid
    assert obs_res.price == 68500.0
