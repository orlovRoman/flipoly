"""
polyflip/crypto/underlying_observations.py

High-frequency price observations collector and query adapter for underlying crypto assets (BTC).
Item 2.1: Stores (instrument, price, source, event_at, received_at).
Guarantees strict temporal causality (event_at <= decision_at AND received_at <= decision_at),
out-of-order normalization, duplicate tolerance, and explicit gap detection (HISTORY_MISSING).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence, Any, Literal
import numpy as np
import pandas as pd
from sqlalchemy import select
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

    def __post_init__(self):
        if self.price <= 0.0 or not np.isfinite(self.price):
            raise ValueError(f"Observation price must be positive finite float, got {self.price}")


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
