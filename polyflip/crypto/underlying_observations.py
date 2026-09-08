"""
polyflip/crypto/underlying_observations.py

High-frequency price observations collector and query adapter for underlying crypto assets (BTC).
Item 2.1: Stores (instrument, price, source, event_at, received_at).
Guarantees strict temporal causality (event_at <= decision_at AND received_at <= decision_at),
out-of-order normalization, duplicate tolerance, and explicit gap detection (HISTORY_MISSING).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence, Any, Literal
import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Observation:
    instrument: str
    price: float
    source: str
    event_at: datetime
    received_at: datetime
    extra_data: dict[str, Any] | None = None
    source_event_id: str | None = None

    def __post_init__(self):
        if self.price <= 0.0 or not np.isfinite(self.price):
            raise ValueError(f"Observation price must be positive finite float, got {self.price}")
        if self.source_event_id is not None:
            extra = dict(self.extra_data or {})
            if "source_event_id" not in extra:
                extra["source_event_id"] = self.source_event_id
            object.__setattr__(self, "extra_data", extra)
        if self.event_at.tzinfo is None:
            object.__setattr__(self, "event_at", self.event_at.replace(tzinfo=timezone.utc))
        if self.received_at.tzinfo is None:
            object.__setattr__(self, "received_at", self.received_at.replace(tzinfo=timezone.utc))


@dataclass(frozen=True)
class ObservationResult:
    price: float | None
    status: Literal["VALID", "HISTORY_MISSING", "NO_OBSERVATIONS", "OUT_OF_TOLERANCE"]
    event_at: datetime | None = None
    received_at: datetime | None = None
    source: str | None = None
    age_seconds: float | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "VALID" and self.price is not None


@dataclass(frozen=True)
class UnderlyingState:
    instrument: str
    binance_price: float | None
    oracle_price: float | None
    latest_price: float | None
    latest_source: str | None
    as_of: datetime
    status: str
    age_seconds: float | None = None


def filter_and_order_observations(
    observations: Sequence[Observation],
    as_of: datetime,
    instrument: str | None = None,
    source: str | None = None,
) -> list[Observation]:
    """
    Filters observations to only those that occurred AND were received on or before as_of.
    Deduplicates identical entries and orders deterministically by (event_at, received_at).
    """
    as_of_utc = pd.to_datetime(as_of, utc=True).to_pydatetime()
    candidates: list[Observation] = []
    seen = set()

    for obs in observations:
        # Strict temporal causality: must have both happened and been received by as_of
        obs_event_utc = pd.to_datetime(obs.event_at, utc=True).to_pydatetime()
        obs_recv_utc = pd.to_datetime(obs.received_at, utc=True).to_pydatetime()

        if obs_event_utc > as_of_utc or obs_recv_utc > as_of_utc:
            continue

        if instrument is not None and obs.instrument.upper() != instrument.upper():
            continue

        if source is not None and obs.source.upper() != source.upper():
            continue

        dedup_key = (obs.instrument.upper(), obs.source.upper(), obs_event_utc, obs_recv_utc, round(obs.price, 6))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Normalize observation with timezone-aware UTC datetime
        candidates.append(
            Observation(
                instrument=obs.instrument.upper(),
                price=float(obs.price),
                source=obs.source.upper(),
                event_at=obs_event_utc,
                received_at=obs_recv_utc,
                extra_data=obs.extra_data,
            )
        )

    # Order chronologically by event time, breaking ties by reception time
    candidates.sort(key=lambda o: (o.event_at, o.received_at))
    return candidates


def get_latest_observation(
    observations: Sequence[Observation],
    as_of: datetime,
    instrument: str = "BTC",
    source: str | None = None,
    max_age_seconds: float = 60.0,
) -> ObservationResult:
    """
    Gets the latest valid observation for instrument available at or before as_of.
    If the newest observation is older than max_age_seconds, returns status='HISTORY_MISSING'.
    """
    filtered = filter_and_order_observations(observations, as_of, instrument=instrument, source=source)
    if not filtered:
        return ObservationResult(price=None, status="NO_OBSERVATIONS")

    latest = filtered[-1]
    as_of_utc = pd.to_datetime(as_of, utc=True).to_pydatetime()
    age = (as_of_utc - latest.event_at).total_seconds()

    if age > max_age_seconds:
        return ObservationResult(
            price=None,
            status="HISTORY_MISSING",
            event_at=latest.event_at,
            received_at=latest.received_at,
            source=latest.source,
            age_seconds=age,
        )

    return ObservationResult(
        price=latest.price,
        status="VALID",
        event_at=latest.event_at,
        received_at=latest.received_at,
        source=latest.source,
        age_seconds=age,
    )


def get_underlying_state(
    observations: Sequence[Observation],
    as_of: datetime,
    instrument: str = "BTC",
    max_age_seconds: float = 60.0,
) -> UnderlyingState:
    """
    Extracts distinct named fields for Binance price and Oracle price at as_of.
    """
    binance_res = get_latest_observation(
        observations, as_of, instrument=instrument, source="BINANCE", max_age_seconds=max_age_seconds
    )
    oracle_res = get_latest_observation(
        observations, as_of, instrument=instrument, source="ORACLE", max_age_seconds=max_age_seconds
    )

    # General latest regardless of source
    any_res = get_latest_observation(
        observations, as_of, instrument=instrument, source=None, max_age_seconds=max_age_seconds
    )

    status = "VALID" if any_res.is_valid else any_res.status
    return UnderlyingState(
        instrument=instrument.upper(),
        binance_price=binance_res.price if binance_res.is_valid else None,
        oracle_price=oracle_res.price if oracle_res.is_valid else None,
        latest_price=any_res.price if any_res.is_valid else None,
        latest_source=any_res.source if any_res.is_valid else None,
        as_of=pd.to_datetime(as_of, utc=True).to_pydatetime(),
        status=status,
        age_seconds=any_res.age_seconds,
    )


def compute_underlying_return(
    observations: Sequence[Observation],
    as_of: datetime,
    horizon_seconds: float,
    instrument: str = "BTC",
    source: str | None = None,
    tolerance_seconds: float | None = None,
    max_age_seconds: float = 60.0,
) -> tuple[float | None, bool]:
    """
    Computes log return log(S_t / S_{t - horizon_seconds}) strictly backward as-of from as_of.
    Requires real high-frequency observation at reference horizon within tolerance_seconds.
    Enforces strictly backward reference search: event_at <= target_time and received_at <= target_time.
    Returns (ret, has_reference_flag).
    """
    if tolerance_seconds is None:
        tolerance_seconds = max(5.0, horizon_seconds * 0.35)

    current_res = get_latest_observation(
        observations, as_of, instrument=instrument, source=source, max_age_seconds=max_age_seconds
    )
    if not current_res.is_valid or current_res.price is None:
        return None, False

    as_of_utc = pd.to_datetime(as_of, utc=True).to_pydatetime()
    target_time = as_of_utc - timedelta(seconds=horizon_seconds)

    # Strictly backward as-of search: filter observations that occurred and were received on or before target_time
    candidates = filter_and_order_observations(
        observations, as_of=target_time,
        instrument=instrument, source=source,
    )
    if not candidates:
        return None, False

    # Find the candidate within [target_time - tolerance, target_time]
    t_min = target_time - timedelta(seconds=tolerance_seconds)
    t_max = target_time

    valid_refs = [c for c in candidates if t_min <= c.event_at <= t_max and c.received_at <= target_time]
    if not valid_refs:
        return None, False

    # Pick the most recent observation on or before target_time
    best_ref = max(valid_refs, key=lambda c: (c.event_at, c.received_at))
    if best_ref.price <= 0.0:
        return None, False

    ret = float(np.log(current_res.price / best_ref.price))
    return ret, True


async def load_observations_for_window(
    db: AsyncSession,
    instrument: str,
    start_time: datetime,
    end_time: datetime,
    max_received_at: datetime | None = None,
    sources: list[str] | None = None,
) -> list[Observation]:
    """
    Loads underlying observations from the database within the requested window,
    enforcing max_received_at boundary when provided.
    """
    from polyflip.db.models import UnderlyingObservation

    query = select(UnderlyingObservation).where(
        UnderlyingObservation.instrument == instrument.upper(),
        UnderlyingObservation.event_at >= start_time,
        UnderlyingObservation.event_at <= end_time,
    )
    if max_received_at is not None:
        query = query.where(UnderlyingObservation.received_at <= max_received_at)
    if sources:
        query = query.where(UnderlyingObservation.source.in_([s.upper() for s in sources]))

    query = query.order_by(UnderlyingObservation.event_at.asc(), UnderlyingObservation.received_at.asc())
    res = await db.execute(query)
    rows = res.scalars().all()

    return [
        Observation(
            instrument=r.instrument,
            price=float(r.price),
            source=r.source,
            event_at=r.event_at,
            received_at=r.received_at,
            extra_data=r.extra_data,
        )
        for r in rows
    ]


class ObservationRepository:
    """
    Data access repository for underlying price observations.
    Provides idempotent writes with conflict suppression on
    (instrument, source, event_at, received_at) and strictly causal queries.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, observation: Observation) -> bool:
        """
        Idempotently inserts a single observation into underlying_observations.
        Returns True if inserted, False if duplicate skipped.
        """
        inserted_count = await self.save_batch([observation])
        return inserted_count > 0

    async def save_batch(self, observations: Sequence[Observation]) -> int:
        """
        Idempotently inserts multiple observations in a single batch.
        Uses ON CONFLICT (instrument, source, event_at, received_at) DO NOTHING.
        Works across both PostgreSQL and SQLite.
        Returns the number of rows inserted.
        """
        if not observations:
            return 0

        from polyflip.db.models import UnderlyingObservation

        values = []
        for o in observations:
            e_at = pd.to_datetime(o.event_at, utc=True).to_pydatetime()
            r_at = pd.to_datetime(o.received_at, utc=True).to_pydatetime()
            values.append(
                {
                    "instrument": o.instrument.upper(),
                    "source": o.source.upper(),
                    "price": float(o.price),
                    "event_at": e_at,
                    "received_at": r_at,
                    "extra_data": o.extra_data,
                }
            )

        bind = self.session.bind
        if bind is None and hasattr(self.session, "get_bind"):
            try:
                bind = self.session.get_bind()
            except Exception:
                bind = None
        dialect_name = getattr(getattr(bind, "dialect", None), "name", "postgresql")

        if dialect_name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            stmt = sqlite_insert(UnderlyingObservation).values(values)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["instrument", "source", "event_at", "received_at"]
            )
        else:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(UnderlyingObservation).values(values)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["instrument", "source", "event_at", "received_at"]
            )

        result = await self.session.execute(stmt)
        await self.session.commit()
        if result.rowcount is not None and result.rowcount >= 0:
            return result.rowcount
        return len(values)

    async def get_latest_observation(
        self,
        instrument: str = "BTC",
        source: str | None = None,
        as_of: datetime | None = None,
        max_age_seconds: float = 60.0,
    ) -> ObservationResult:
        """
        Queries the latest causal observation on or before as_of.
        Enforces event_at <= as_of AND received_at <= as_of.
        """
        from polyflip.db.models import UnderlyingObservation

        if as_of is None:
            as_of = datetime.now(timezone.utc)
        as_of_utc = pd.to_datetime(as_of, utc=True).to_pydatetime()

        query = select(UnderlyingObservation).where(
            UnderlyingObservation.instrument == instrument.upper(),
            UnderlyingObservation.event_at <= as_of_utc,
            UnderlyingObservation.received_at <= as_of_utc,
        )
        if source is not None:
            query = query.where(UnderlyingObservation.source == source.upper())

        query = query.order_by(
            UnderlyingObservation.event_at.desc(),
            UnderlyingObservation.received_at.desc(),
        ).limit(1)

        res = await self.session.execute(query)
        row = res.scalar_one_or_none()

        if row is None:
            return ObservationResult(price=None, status="NO_OBSERVATIONS")

        row_event_utc = pd.to_datetime(row.event_at, utc=True).to_pydatetime()
        row_recv_utc = pd.to_datetime(row.received_at, utc=True).to_pydatetime()
        age = (as_of_utc - row_event_utc).total_seconds()

        if age > max_age_seconds:
            return ObservationResult(
                price=None,
                status="HISTORY_MISSING",
                event_at=row_event_utc,
                received_at=row_recv_utc,
                source=row.source,
                age_seconds=age,
            )

        return ObservationResult(
            price=float(row.price),
            status="VALID",
            event_at=row_event_utc,
            received_at=row_recv_utc,
            source=row.source,
            age_seconds=age,
        )

    async def get_underlying_state(
        self,
        instrument: str = "BTC",
        as_of: datetime | None = None,
        max_age_seconds: float = 60.0,
    ) -> UnderlyingState:
        """
        Reconstructs the UnderlyingState containing distinct named fields for
        Binance price and Oracle price at as_of.
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)
        as_of_utc = pd.to_datetime(as_of, utc=True).to_pydatetime()

        binance_res = await self.get_latest_observation(
            instrument=instrument, source="BINANCE", as_of=as_of_utc, max_age_seconds=max_age_seconds
        )
        oracle_res = await self.get_latest_observation(
            instrument=instrument, source="ORACLE", as_of=as_of_utc, max_age_seconds=max_age_seconds
        )
        any_res = await self.get_latest_observation(
            instrument=instrument, source=None, as_of=as_of_utc, max_age_seconds=max_age_seconds
        )

        status = "VALID" if any_res.is_valid else any_res.status
        return UnderlyingState(
            instrument=instrument.upper(),
            binance_price=binance_res.price if binance_res.is_valid else None,
            oracle_price=oracle_res.price if oracle_res.is_valid else None,
            latest_price=any_res.price if any_res.is_valid else None,
            latest_source=any_res.source if any_res.is_valid else None,
            as_of=as_of_utc,
            status=status,
            age_seconds=any_res.age_seconds,
        )

    async def load_window(
        self,
        instrument: str,
        start_time: datetime,
        end_time: datetime,
        max_received_at: datetime | None = None,
        sources: list[str] | None = None,
    ) -> list[Observation]:
        return await load_observations_for_window(
            self.session, instrument, start_time, end_time, max_received_at=max_received_at, sources=sources
        )

    async def count(
        self,
        instrument: str | None = None,
        source: str | None = None,
    ) -> int:
        from polyflip.db.models import UnderlyingObservation

        query = select(func.count(UnderlyingObservation.id))
        if instrument is not None:
            query = query.where(UnderlyingObservation.instrument == instrument.upper())
        if source is not None:
            query = query.where(UnderlyingObservation.source == source.upper())
        res = await self.session.execute(query)
        return res.scalar() or 0


class ObservationWriter:
    """
    High-frequency buffered ObservationWriter for streaming and batch writes.
    Buffers incoming observations in memory and writes them periodically or
    on batch threshold to the database via ObservationRepository.
    Tracks health telemetry, tick counters, latencies, and connection errors.
    """

    def __init__(
        self,
        session_factory=None,
        buffer_capacity: int = 2000,
        batch_size: int = 50,
        flush_interval_sec: float = 1.0,
    ):
        self.session_factory = session_factory
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec

        self._buffer: list[Observation] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._running = False

        # Telemetry & metrics
        self._ticks_received = 0
        self._ticks_written = 0
        self._ticks_deduplicated = 0
        self._flush_count = 0
        self._error_count = 0
        self._last_error: str | None = None
        self._last_flush_at: datetime | None = None
        self._last_tick_time: dict[str, datetime] = {}
        self._latest_prices: dict[str, float] = {}

    def record(self, observation: Observation) -> None:
        """
        Enqueues an observation into memory buffer synchronously.
        """
        key = f"{observation.instrument.upper()}:{observation.source.upper()}"
        self._last_tick_time[key] = observation.received_at
        self._latest_prices[key] = observation.price
        self._ticks_received += 1

        if len(self._buffer) >= self.buffer_capacity:
            drop_count = max(1, self.buffer_capacity // 10)
            self._buffer = self._buffer[drop_count:]
            logger.warning("observation_writer_buffer_overflow_dropped", dropped=drop_count)

        self._buffer.append(observation)

    def record_tick(
        self,
        instrument: str,
        price: float,
        source: str,
        event_at: datetime | None = None,
        received_at: datetime | None = None,
        extra_data: dict[str, Any] | None = None,
        source_event_id: str | None = None,
    ) -> Observation:
        """
        Convenience builder to record a single tick into the buffer.
        """
        now = datetime.now(timezone.utc)
        obs = Observation(
            instrument=instrument.upper(),
            price=float(price),
            source=source.upper(),
            event_at=event_at if event_at is not None else now,
            received_at=received_at if received_at is not None else now,
            extra_data=extra_data,
            source_event_id=source_event_id,
        )
        self.record(obs)
        return obs

    async def flush(self, session: AsyncSession | None = None) -> int:
        """
        Flushes all buffered observations to the database using ObservationRepository.
        """
        async with self._lock:
            if not self._buffer:
                return 0
            batch = self._buffer
            self._buffer = []

        try:
            if session is not None:
                repo = ObservationRepository(session)
                written = await repo.save_batch(batch)
            elif self.session_factory is not None:
                async with self.session_factory() as s:
                    repo = ObservationRepository(s)
                    written = await repo.save_batch(batch)
            else:
                async with self._lock:
                    self._buffer = batch + self._buffer
                return 0

            self._ticks_written += written
            self._ticks_deduplicated += max(0, len(batch) - written)
            self._flush_count += 1
            self._last_flush_at = datetime.now(timezone.utc)
            self._last_error = None
            return written

        except Exception as exc:
            self._error_count += 1
            self._last_error = str(exc)
            logger.error("observation_writer_flush_error", error=str(exc), batch_size=len(batch))
            async with self._lock:
                remaining_space = max(0, self.buffer_capacity - len(self._buffer))
                self._buffer = batch[:remaining_space] + self._buffer
            return 0

    async def start(self) -> None:
        """
        Starts the background periodic flush task.
        """
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("observation_writer_started", flush_interval=self.flush_interval_sec)

    async def stop(self) -> None:
        """
        Stops background flush task and performs final flush.
        """
        if not self._running:
            return
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush()
        logger.info("observation_writer_stopped", ticks_written=self._ticks_written)

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_sec)
                if self._buffer:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("observation_writer_loop_error", error=str(exc))

    def get_latest_cached_price(self, instrument: str, source: str | None = None) -> float | None:
        """
        Returns freshest in-memory price without hitting database.
        """
        if source is not None:
            return self._latest_prices.get(f"{instrument.upper()}:{source.upper()}")
        for s in ("BINANCE", "ORACLE", "CHAINLINK"):
            val = self._latest_prices.get(f"{instrument.upper()}:{s}")
            if val is not None:
                return val
        for k, v in self._latest_prices.items():
            if k.startswith(f"{instrument.upper()}:"):
                return v
        return None

    def get_health(self, max_stale_seconds: float = 60.0) -> dict[str, Any]:
        """
        Returns structured healthcheck telemetry.
        """
        now = datetime.now(timezone.utc)
        latest_tick_age = None
        if self._last_tick_time:
            newest_tick = max(self._last_tick_time.values())
            latest_tick_age = (now - newest_tick).total_seconds()

        if not self._running and self._ticks_received == 0:
            status = "IDLE"
            healthy = True
        elif self._error_count > 5 and (self._last_flush_at is None or (now - self._last_flush_at).total_seconds() > max_stale_seconds):
            status = "ERROR"
            healthy = False
        elif latest_tick_age is not None and latest_tick_age > max_stale_seconds:
            status = "DEGRADED"
            healthy = False
        else:
            status = "HEALTHY"
            healthy = True

        return {
            "status": status,
            "healthy": healthy,
            "is_running": self._running,
            "buffer_size": len(self._buffer),
            "ticks_received": self._ticks_received,
            "ticks_written": self._ticks_written,
            "ticks_deduplicated": self._ticks_deduplicated,
            "flush_count": self._flush_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "last_flush_at": self._last_flush_at.isoformat() if self._last_flush_at else None,
            "latest_tick_age_seconds": latest_tick_age,
            "sources": {k: v.isoformat() for k, v in self._last_tick_time.items()},
            "latest_prices": dict(self._latest_prices),
        }


_GLOBAL_OBSERVATION_WRITER: ObservationWriter | None = None


def get_observation_writer(session_factory=None) -> ObservationWriter:
    global _GLOBAL_OBSERVATION_WRITER
    if _GLOBAL_OBSERVATION_WRITER is None:
        if session_factory is None:
            from polyflip.db.connection import async_session
            session_factory = async_session
        _GLOBAL_OBSERVATION_WRITER = ObservationWriter(session_factory=session_factory)
    return _GLOBAL_OBSERVATION_WRITER


def set_observation_writer(writer: ObservationWriter | None) -> None:
    global _GLOBAL_OBSERVATION_WRITER
    _GLOBAL_OBSERVATION_WRITER = writer

