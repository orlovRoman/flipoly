"""
tests/crypto/test_spot_collector.py

Unit and integration tests for real-time spot tick collector:
1. BinanceSpotStream message parsing and event time handling.
2. BinanceRestPoller adaptive fallback ingestion.
3. PolymarketOracleCollector strike/underlying price extraction.
4. SpotCollectorDaemon lifecycle and telemetry healthcheck.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from polyflip.crypto.underlying_observations import ObservationWriter
from polyflip.crypto.spot_collector import (
    BinanceSpotStream,
    BinanceRestPoller,
    PolymarketOracleCollector,
    SpotCollectorDaemon,
)


@pytest.fixture
def writer() -> ObservationWriter:
    return ObservationWriter(buffer_capacity=100)


def test_binance_stream_message_parsing(writer: ObservationWriter):
    """BinanceSpotStream parses miniTicker payload and records Observation."""
    stream = BinanceSpotStream(assets=["BTC", "ETH"], writer=writer)

    # Combined stream format
    msg_combined = (
        '{"stream": "btcusdt@miniTicker", "data": {"s": "BTCUSDT", "c": "78500.50", "E": 1725800000000}}'
    )
    stream._handle_message(msg_combined)

    assert writer._ticks_received == 1
    obs = writer._buffer[0]
    assert obs.instrument == "BTC"
    assert obs.source == "BINANCE"
    assert obs.price == 78500.50
    assert obs.event_at == datetime.fromtimestamp(1725800000, tz=timezone.utc)

    # Raw single stream format
    msg_raw = '{"s": "ETHUSDT", "c": "3400.25", "E": 1725800005000}'
    stream._handle_message(msg_raw)

    assert writer._ticks_received == 2
    obs2 = writer._buffer[1]
    assert obs2.instrument == "ETH"
    assert obs2.source == "BINANCE"
    assert obs2.price == 3400.25

    # Unconfigured asset (e.g. DOGE not in ["BTC", "ETH"])
    msg_ignored = '{"s": "DOGEUSDT", "c": "0.15", "E": 1725800010000}'
    stream._handle_message(msg_ignored)
    assert writer._ticks_received == 2  # Not incremented


@pytest.mark.asyncio
async def test_binance_rest_poller_fallback(writer: ObservationWriter):
    """BinanceRestPoller queries HTTP ticker and records observations."""
    poller = BinanceRestPoller(assets=["BTC", "SOL"], writer=writer)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = [
        {"symbol": "BTCUSDT", "price": "79000.00"},
        {"symbol": "SOLUSDT", "price": "180.50"},
        {"symbol": "UNKNOWNUSDT", "price": "1.00"},
    ]

    mock_client = AsyncMock()
    mock_client.get.return_value = fake_response

    await poller._poll_once(mock_client, ["BTCUSDT", "SOLUSDT"])

    assert writer._ticks_received == 2
    prices = {o.instrument: o.price for o in writer._buffer}
    assert prices["BTC"] == 79000.00
    assert prices["SOL"] == 180.50


@pytest.mark.asyncio
async def test_polymarket_oracle_collector(writer: ObservationWriter):
    """PolymarketOracleCollector extracts strike prices as ORACLE and CHAINLINK observations."""
    collector = PolymarketOracleCollector(assets=["BTC"], writer=writer)

    from polyflip.collector.client import StrikeProvenance
    now = datetime.now(timezone.utc)
    mock_markets = [
        {
            "market_id": "12345",
            "condition_id": "0xabc",
            "asset": "BTC",
            "question": "Bitcoin Up or Down - 10:00AM-10:15AM ET",
            "strike_provenance": StrikeProvenance(
                strike_value=78950.0,
                strike_source="market.strikePrice",
                strike_effective_at=now,
                strike_received_at=now,
            ),
        }
    ]

    with patch("polyflip.crypto.spot_collector.PolymarketClient") as MockClient:
        instance = AsyncMock()
        instance.get_active_15m_markets.return_value = mock_markets
        MockClient.return_value = instance

        await collector._poll_oracle_ticks()

    # Emits both ORACLE and CHAINLINK observations
    assert writer._ticks_received == 2
    sources = [o.source for o in writer._buffer]
    assert "ORACLE" in sources
    assert "CHAINLINK" in sources
    for o in writer._buffer:
        assert o.price == 78950.0
        assert o.instrument == "BTC"


@pytest.mark.asyncio
async def test_spot_collector_daemon_lifecycle(engine):
    """SpotCollectorDaemon starts, manages tasks, reports health, and stops gracefully."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    daemon = SpotCollectorDaemon(
        assets=["BTC"],
        session_factory=session_factory,
        flush_interval_sec=0.05,
    )

    # Mock stream and poller runs to avoid real network requests in unit tests
    daemon.ws_stream.run = AsyncMock()
    daemon.rest_poller.run = AsyncMock()
    daemon.oracle_collector.run = AsyncMock()

    await daemon.start()
    assert daemon._running is True

    health = daemon.get_health()
    assert health["is_running"] is True
    assert "BTC" in health["assets"]
    assert "writer" in health
    assert "binance_ws" in health

    await daemon.stop()
    assert daemon._running is False


@pytest.mark.asyncio
async def test_polymarket_oracle_collector_deduplication(writer: ObservationWriter):
    """Calling _poll_oracle_ticks repeatedly with identical market strikes must not create duplicate ticks."""
    collector = PolymarketOracleCollector(assets=["BTC"], writer=writer)

    from polyflip.collector.client import StrikeProvenance
    now = datetime.now(timezone.utc)
    mock_markets = [
        {
            "market_id": "12345",
            "condition_id": "0xabc",
            "asset": "BTC",
            "question": "Bitcoin Up or Down - 10:00AM-10:15AM ET",
            "strike_provenance": StrikeProvenance(
                strike_value=78950.0,
                strike_source="market.strikePrice",
                strike_effective_at=now,
                strike_received_at=now,
            ),
        }
    ]

    with patch("polyflip.crypto.spot_collector.PolymarketClient") as MockClient:
        instance = AsyncMock()
        instance.get_active_15m_markets.return_value = mock_markets
        MockClient.return_value = instance

        # First poll records ORACLE and CHAINLINK (2 ticks)
        await collector._poll_oracle_ticks()
        assert writer._ticks_received == 2

        # Second poll with identical market and strike must NOT record again
        await collector._poll_oracle_ticks()
        assert writer._ticks_received == 2


@pytest.mark.asyncio
async def test_binance_rest_poller_network_error_fallback(writer: ObservationWriter):
    """BinanceRestPoller falls back to backup endpoint when primary endpoint raises network error."""
    poller = BinanceRestPoller(assets=["BTC"], writer=writer)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = [{"symbol": "BTCUSDT", "price": "79500.00"}]

    mock_client = AsyncMock()
    # First call (primary endpoint) raises ConnectError, second call (backup endpoint) returns 200
    mock_client.get.side_effect = [Exception("Connection timed out"), fake_response]

    await poller._poll_once(mock_client, ["BTCUSDT"])
    assert writer._ticks_received == 1
    assert writer._buffer[0].price == 79500.00


def test_binance_spot_stream_backup_url(writer: ObservationWriter):
    """BinanceSpotStream builds backup URL using BINANCE_WS_BACKUP when use_backup is True."""
    stream = BinanceSpotStream(assets=["BTC", "ETH"], writer=writer)
    primary_url = stream._build_stream_url(use_backup=False)
    backup_url = stream._build_stream_url(use_backup=True)

    assert "stream.binance.com" in primary_url
    assert "data-stream.binance.vision" in backup_url

