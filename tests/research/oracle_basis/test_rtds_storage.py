"""RTDS storage semantics on SQLite: keeper wins, collisions registered, journal kept."""

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from polyflip.collector.rtds_service import (
    archive_file_path,
    days_to_archive,
    retention_cutoff_day,
    utc_day_of_ms,
)
from polyflip.db.models import (
    Base,
    RTDSConflict,
    RTDSJournalArchive,
    RTDSObservation,
    RTDSRawJournal,
    RTDSStreamSession,
    RTDSDailyVolume,
)

TABLES = (
    "rtds_stream_sessions",
    "rtds_observations",
    "rtds_raw_journal",
    "rtds_conflicts",
    "rtds_daily_volume",
    "rtds_journal_archives",
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite://")
    metadata = [t for t in Base.metadata.sorted_tables if t.name in TABLES]
    Base.metadata.create_all(engine, tables=metadata)
    factory = sessionmaker(bind=engine)
    handle = factory()
    yield handle
    handle.close()


def _dt(ms):
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def test_keeper_first_received_wins_and_conflict_registered(session):
    session.add(RTDSStreamSession(session_id="s1", started_at=_dt(1000)))
    session.add(
        RTDSObservation(
            session_id="s1",
            source="RTDS_TWAP60",
            topic="crypto_prices_twap_sixty",
            symbol="BTC/USD",
            asset="BTC",
            currency="USD",
            window_s=60,
            raw_value_text="65000500000000000000000",
            raw_e18=65000500000000000000000,
            value_source="FULL_ACCURACY",
            observed_at=_dt(2000),
            received_at=_dt(2100),
        )
    )
    session.commit()
    # Same keeper key, later receipt, different value: must be rejected...
    session.add(
        RTDSObservation(
            session_id="s1",
            source="RTDS_TWAP60",
            topic="crypto_prices_twap_sixty",
            symbol="BTC/USD",
            asset="BTC",
            currency="USD",
            window_s=60,
            raw_value_text="65000600000000000000000",
            raw_e18=65000600000000000000000,
            value_source="FULL_ACCURACY",
            observed_at=_dt(2000),
            received_at=_dt(2200),
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    keeper = session.query(RTDSObservation).one()
    assert (
        keeper.raw_value_text == "65000500000000000000000"
    )  # exact text on every backend
    # Numeric(38,0) is exact on Postgres; SQLite binds numerics as float, so
    # exactness here is approximate — the TEXT column is the audit source.
    assert float(keeper.raw_e18) == pytest.approx(6.50005e22, rel=1e-9)
    session.add(
        RTDSConflict(
            source="RTDS_TWAP60",
            symbol="BTC/USD",
            window_s=60,
            observed_at=_dt(2000),
            keeper_raw_e18=keeper.raw_e18,
            keeper_received_at=keeper.received_at,
            rejected_raw_e18=65000600000000000000000,
            rejected_value_text="65000600000000000000000",
            rejected_received_at=_dt(2200),
            session_id="s1",
        )
    )
    session.commit()
    assert session.query(RTDSConflict).count() == 1
    # Same rejection twice is idempotent.
    session.add(
        RTDSConflict(
            source="RTDS_TWAP60",
            symbol="BTC/USD",
            window_s=60,
            observed_at=_dt(2000),
            keeper_raw_e18=keeper.raw_e18,
            keeper_received_at=keeper.received_at,
            rejected_raw_e18=65000600000000000000000,
            rejected_value_text="65000600000000000000000",
            rejected_received_at=_dt(2200),
            session_id="s1",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_raw_journal_row(session):
    session.add(
        RTDSRawJournal(
            session_id="s1",
            received_at=_dt(2100),
            topic="crypto_prices",
            raw_text='{"topic":"crypto_prices"}',
            sha256="ab" * 32,
        )
    )
    session.commit()
    assert session.query(RTDSRawJournal).count() == 1


def test_retention_helpers_are_explicit():
    assert utc_day_of_ms(1_700_000_000_000) == date(2023, 11, 14)
    today = date(2026, 9, 11)
    assert retention_cutoff_day(today, 30) == date(2026, 8, 12)
    assert archive_file_path("./backups/rtds", date(2026, 9, 1)).endswith(
        "rtds_raw_journal_2026-09-01.parquet"
    )
    journal = {date(2026, 8, 1), date(2026, 8, 12), date(2026, 9, 10)}
    assert days_to_archive(journal, set(), date(2026, 8, 12)) == [date(2026, 8, 1)]
    assert days_to_archive(journal, {date(2026, 8, 1)}, date(2026, 8, 12)) == []
    with pytest.raises(ValueError):
        retention_cutoff_day(today, 0)


def test_daily_volume_and_archive_registry(session):
    session.add(
        RTDSDailyVolume(
            day=date(2026, 9, 10),
            topic="crypto_prices",
            messages=100,
            bytes=5000,
            events=100,
        )
    )
    session.commit()
    row = session.query(RTDSDailyVolume).one()
    row.messages = int(row.messages) + 50
    session.commit()
    assert session.query(RTDSDailyVolume).one().messages == 150
    session.add(
        RTDSJournalArchive(
            day=date(2026, 8, 1),
            path="./backups/rtds/rtds_raw_journal_2026-08-01.parquet",
            sha256="cd" * 32,
            rows=432000,
        )
    )
    session.commit()
    assert session.query(RTDSJournalArchive).count() == 1
