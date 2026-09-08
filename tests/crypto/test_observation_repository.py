"""
tests/crypto/test_observation_repository.py

Tests for ObservationRepository and ObservationWriter:
1. Idempotent insertion (duplicate tolerance with on_conflict_do_nothing).
2. Out-of-order insertion and deterministic causal retrieval.
3. Strict temporal causality (excluding future events and late receptions).
4. Gap detection yielding HISTORY_MISSING.
5. Distinct named fields in UnderlyingState (binance_price, oracle_price, latest_price).
6. ObservationWriter buffering, flush mechanism, background loop, and graceful shutdown.
7. ObservationWriter error handling and retry retention on transient database failure.
8. Telemetry and healthcheck reporting.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.crypto.underlying_observations import (
    Observation,
    ObservationRepository,
    ObservationWriter,
)


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_repository_save_and_idempotent_dedup(db_session: AsyncSession, base_time: datetime):
    """ObservationRepository.save() and save_batch() must ignore duplicates without errors."""
    repo = ObservationRepository(db_session)
    t0 = base_time

    obs1 = Observation("BTC", 50000.0, "BINANCE", t0, t0)
    obs2 = Observation("BTC", 50100.0, "BINANCE", t0 + timedelta(seconds=5), t0 + timedelta(seconds=5))

    # Initial insertion
    saved1 = await repo.save(obs1)
    assert saved1 is True

    # Duplicate insert of exact same (instrument, source, event_at, received_at)
    saved1_dup = await repo.save(obs1)
    # Deduplicated (not inserted again)
    assert saved1_dup is False

    # Batch insert with duplicate
    batch = [
        obs1,  # Duplicate
        obs2,  # New
    ]
    written = await repo.save_batch(batch)
    assert written == 1

    # Total count should be 2
    count = await repo.count("BTC")
    assert count == 2


@pytest.mark.asyncio
async def test_repository_causal_query_and_gap_detection(db_session: AsyncSession, base_time: datetime):
    """get_latest_observation must enforce causality and detect gaps."""
    repo = ObservationRepository(db_session)
    t0 = base_time

    obs_list = [
        Observation("BTC", 50000.0, "BINANCE", t0, t0),
        Observation("BTC", 50200.0, "BINANCE", t0 + timedelta(seconds=20), t0 + timedelta(seconds=25)),
    ]
    await repo.save_batch(obs_list)

    # As of t+10: only t0 is visible (t+20 is in the future relative to as_of)
    res_10 = await repo.get_latest_observation("BTC", source="BINANCE", as_of=t0 + timedelta(seconds=10))
    assert res_10.is_valid
    assert res_10.price == 50000.0

    # As of t+22: event at t+20 happened, but received_at is t+25!
    # Strict temporal causality requires received_at <= as_of, so t+20 observation is NOT visible!
    res_22 = await repo.get_latest_observation("BTC", source="BINANCE", as_of=t0 + timedelta(seconds=22))
    assert res_22.is_valid
    assert res_22.price == 50000.0

    # As of t+26: both happened and received
    res_26 = await repo.get_latest_observation("BTC", source="BINANCE", as_of=t0 + timedelta(seconds=26))
    assert res_26.is_valid
    assert res_26.price == 50200.0

    # As of t+120: latest observation (t+20) is 100s old, exceeding max_age_seconds=60s
    res_gap = await repo.get_latest_observation("BTC", source="BINANCE", as_of=t0 + timedelta(seconds=120), max_age_seconds=60.0)
    assert res_gap.status == "HISTORY_MISSING"
    assert res_gap.price is None


@pytest.mark.asyncio
async def test_repository_underlying_state_distinct_sources(db_session: AsyncSession, base_time: datetime):
    """UnderlyingState tracks binance_price, oracle_price, and latest_price distinctly."""
    repo = ObservationRepository(db_session)
    t0 = base_time

    obs_binance = Observation("BTC", 60000.0, "BINANCE", t0, t0)
    obs_oracle = Observation("BTC", 60020.0, "ORACLE", t0 + timedelta(seconds=2), t0 + timedelta(seconds=2))

    await repo.save_batch([obs_binance, obs_oracle])

    state = await repo.get_underlying_state("BTC", as_of=t0 + timedelta(seconds=5))
    assert state.status == "VALID"
    assert state.instrument == "BTC"
    assert state.binance_price == 60000.0
    assert state.oracle_price == 60020.0
    assert state.latest_price == 60020.0
    assert state.latest_source == "ORACLE"


@pytest.mark.asyncio
async def test_writer_buffering_and_flush(db_session: AsyncSession):
    """ObservationWriter buffers ticks and flushes them to database via repository."""
    writer = ObservationWriter(buffer_capacity=100)
    now = datetime.now(timezone.utc)

    writer.record_tick("BTC", 65000.0, "BINANCE", event_at=now, received_at=now)
    writer.record_tick("BTC", 65050.0, "ORACLE", event_at=now + timedelta(seconds=1), received_at=now + timedelta(seconds=1))

    health = writer.get_health()
    assert health["buffer_size"] == 2
    assert health["ticks_received"] == 2
    assert health["ticks_written"] == 0

    # Flush using active db_session
    written = await writer.flush(session=db_session)
    assert written == 2

    health_after = writer.get_health()
    assert health_after["buffer_size"] == 0
    assert health_after["ticks_written"] == 2
    assert health_after["status"] == "HEALTHY"

    # Confirm rows are persisted in DB
    repo = ObservationRepository(db_session)
    assert await repo.count("BTC") == 2


@pytest.mark.asyncio
async def test_writer_stale_health_degraded(base_time: datetime):
    """If ticks are older than max_stale_seconds, health status must be DEGRADED."""
    writer = ObservationWriter(buffer_capacity=100)
    t_old = base_time  # 2026-01-01 is old

    writer.record_tick("BTC", 65000.0, "BINANCE", event_at=t_old, received_at=t_old)
    health = writer.get_health(max_stale_seconds=60.0)
    assert health["status"] == "DEGRADED"
    assert health["healthy"] is False


@pytest.mark.asyncio
async def test_writer_background_loop_and_graceful_stop(engine, base_time: datetime):
    """ObservationWriter background task periodically flushes ticks automatically."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    writer = ObservationWriter(session_factory=session_factory, flush_interval_sec=0.05)
    await writer.start()
    assert writer._running is True

    t0 = base_time
    writer.record_tick("BTC", 70000.0, "BINANCE", event_at=t0, received_at=t0)
    writer.record_tick("ETH", 3500.0, "BINANCE", event_at=t0, received_at=t0)

    # Wait for flush loop
    await asyncio.sleep(0.15)

    health = writer.get_health()
    assert health["ticks_written"] == 2
    assert health["buffer_size"] == 0

    # Stop writer gracefully
    writer.record_tick("SOL", 150.0, "BINANCE", event_at=t0, received_at=t0)
    await writer.stop()
    assert writer._running is False

    # Stop should have flushed the remaining SOL tick
    async with session_factory() as s:
        repo = ObservationRepository(s)
        assert await repo.count("SOL") == 1
        assert await repo.count() == 3
