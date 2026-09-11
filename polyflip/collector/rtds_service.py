"""RTDS collector service: standalone process, no scheduler involvement.

Connects to wss://ws-live-data.polymarket.com, persists keeper observations to
rtds_observations (first received wins), every source message to
rtds_raw_journal, keeper-key collisions to rtds_conflicts, and connection
history to rtds_stream_sessions. Run: python -m polyflip.collector.rtds_service
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import signal
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from polyflip.collector.rtds_client import (
    CanonicalEvent,
    RTDSClient,
    build_subscriptions,
    parse_rtds_message,
)

logger = structlog.get_logger(__name__)

ALIVE_FILE = "/tmp/rtds_collector_alive"


def _env_list(name: str, default: str) -> list[str]:
    return [
        part.strip()
        for part in os.environ.get(name, default).split(",")
        if part.strip()
    ]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def _coerce_day(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)).date()


def utc_day_of_ms(ms: int) -> date:
    """UTC calendar day of a millisecond timestamp (retention bucketing)."""
    return _to_dt(int(ms)).date()


def retention_cutoff_day(today: date, retention_days: int) -> date:
    """Days strictly older than the cutoff are archived and purged."""
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    return today - timedelta(days=int(retention_days))


def archive_file_path(archive_dir: str, day: date) -> str:
    return os.path.join(archive_dir, f"rtds_raw_journal_{day.isoformat()}.parquet")


def days_to_archive(
    journal_days: set[date], archived_days: set[date], cutoff: date
) -> list[date]:
    """Journal days eligible for rotation: older than cutoff, not yet archived."""
    return sorted(d for d in journal_days if d < cutoff and d not in archived_days)


class RTDSService:
    """Buffers RTDS frames and flushes them to Postgres with conflict registry."""

    def __init__(self, session_factory=None):
        from polyflip.db.connection import async_session as default_factory

        self.session_factory = session_factory or default_factory
        self.session_id = uuid.uuid4().hex
        raw_spot = _env_list("RTDS_SPOT_SYMBOLS", "")
        # Empty RTDS_SPOT_SYMBOLS subscribes to crypto_prices WITHOUT filters and
        # allowlists client-side, so unconfirmed filter support (e.g. dogeusdt)
        # can never break BTC/ETH/SOL/XRP. A non-empty value restores filtered
        # subscription and doubles as the allowlist.
        self.spot_filtered = [s.lower() for s in raw_spot]
        allow = _env_list(
            "RTDS_SPOT_ALLOWLIST", "btcusdt,ethusdt,solusdt,xrpusdt,dogeusdt"
        )
        self.spot_allowlist = {s.upper() for s in (self.spot_filtered or allow)}
        self.chainlink_symbols = _env_list(
            "RTDS_CHAINLINK_SYMBOLS", "btc/usd,eth/usd,sol/usd,xrp/usd"
        )
        self.twap_symbols = _env_list(
            "RTDS_TWAP_SYMBOLS", "btc/usd,eth/usd,sol/usd,xrp/usd"
        )
        self.flush_sec = _env_float("RTDS_FLUSH_SEC", 2.0)
        self.buffer_cap = _env_int("RTDS_BUFFER_CAP", 5000)
        self.retention_days = _env_int("RTDS_JOURNAL_RETENTION_DAYS", 30)
        self.archive_dir = os.environ.get("RTDS_ARCHIVE_DIR", "./backups/rtds")
        self._volume: dict[tuple[date, str], list[int]] = {}
        self._last_retention_ms = 0
        self._events: list[CanonicalEvent] = []
        self._journal: list[dict[str, Any]] = []
        self._gaps: list[dict[str, int]] = []
        self._confirmed: set[tuple[str, str]] = set()
        self._running = False
        self._session_dirty = True
        self._start_ms = 0
        self._ended_ms = 0
        self.counters = {
            "messages": 0,
            "events": 0,
            "keepers_written": 0,
            "conflicts": 0,
            "journal_written": 0,
            "parse_errors": 0,
            "unknown_frames": 0,
            "filtered_out": 0,
        }
        self._client: RTDSClient | None = None

    async def on_message(self, raw_text: str, received_ms: int) -> None:
        from polyflip.collector.rtds_client import RTDSError as _ParseError

        self.counters["messages"] += 1
        try:
            events = parse_rtds_message(raw_text, received_ms)
        except _ParseError as exc:
            self.counters["parse_errors"] += 1
            logger.warning("rtds_parse_error", error=str(exc))
            return
        if not events:
            self.counters["unknown_frames"] += 1
            return
        kept = []
        for event in events:
            self._confirmed.add((event.topic, event.symbol))
            # Unfiltered spot carries every pair: persist only allowlisted
            # assets, count the rest as filtered (never as errors or prices).
            if (
                event.topic == "crypto_prices"
                and event.symbol not in self.spot_allowlist
            ):
                self.counters["filtered_out"] += 1
                continue
            kept.append(event)
        if not kept:
            return
        self.counters["events"] += len(kept)
        day = utc_day_of_ms(received_ms)
        raw_bytes = len(raw_text.encode("utf-8"))
        # One event per frame on the subscribed topics; bytes are attributed
        # per kept event, which equals per message here.
        for event in kept:
            bucket = self._volume.setdefault((day, event.topic), [0, 0, 0])
            bucket[0] += 1
            bucket[1] += raw_bytes
            bucket[2] += 1
        if len(self._events) + len(kept) > self.buffer_cap:
            drop = len(self._events) + len(kept) - self.buffer_cap
            self._events = self._events[drop:]
            self._journal = self._journal[drop:]
            logger.warning("rtds_buffer_overflow_dropped", dropped=drop)
        self._events.extend(kept)
        for event in kept:
            self._journal.append(
                {
                    "received_ms": event.received_ms,
                    "topic": event.topic,
                    "raw_text": raw_text,
                    "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                }
            )

    async def on_gap(self, from_ms: int, to_ms: int) -> None:
        self._gaps.append({"from_ms": from_ms, "to_ms": to_ms})
        self._session_dirty = True
        logger.warning("rtds_disconnect_gap", from_ms=from_ms, to_ms=to_ms)

    async def _flush(self) -> None:
        from polyflip.db.models import (
            RTDSConflict,
            RTDSDailyVolume,
            RTDSObservation,
            RTDSRawJournal,
            RTDSStreamSession,
        )

        if (
            not self._events
            and not self._journal
            and not self._volume
            and not self._session_dirty
        ):
            return
        events, self._events = self._events, []
        journal, self._journal = self._journal, []
        volume, self._volume = self._volume, {}
        async with self.session_factory() as session:
            try:
                for event in events:
                    keeper = RTDSObservation(
                        session_id=self.session_id,
                        source=event.source,
                        topic=event.topic,
                        symbol=event.symbol,
                        asset=event.asset,
                        currency=event.currency,
                        window_s=event.window_s,
                        raw_value_text=event.value_text,
                        raw_e18=event.raw_e18,
                        value_source=event.value_source,
                        observed_at=_to_dt(event.observed_ms),
                        received_at=_to_dt(event.received_ms),
                        extra_data=(event.extra or None),
                    )
                    try:
                        async with session.begin_nested():
                            session.add(keeper)
                            await session.flush()
                        self.counters["keepers_written"] += 1
                    except IntegrityError:
                        existing = (
                            await session.execute(
                                select(RTDSObservation).where(
                                    RTDSObservation.source == event.source,
                                    RTDSObservation.symbol == event.symbol,
                                    RTDSObservation.window_s == event.window_s,
                                    RTDSObservation.observed_at
                                    == _to_dt(event.observed_ms),
                                )
                            )
                        ).scalar_one()
                        session.add(
                            RTDSConflict(
                                source=event.source,
                                symbol=event.symbol,
                                window_s=event.window_s,
                                observed_at=_to_dt(event.observed_ms),
                                keeper_raw_e18=existing.raw_e18,
                                keeper_received_at=existing.received_at,
                                rejected_raw_e18=event.raw_e18,
                                rejected_value_text=event.value_text,
                                rejected_received_at=_to_dt(event.received_ms),
                                session_id=self.session_id,
                            )
                        )
                        try:
                            async with session.begin_nested():
                                await session.flush()
                            self.counters["conflicts"] += 1
                        except IntegrityError:
                            pass  # same rejection already registered
                for entry in journal:
                    session.add(
                        RTDSRawJournal(
                            session_id=self.session_id,
                            received_at=_to_dt(entry["received_ms"]),
                            topic=entry["topic"],
                            raw_text=entry["raw_text"],
                            sha256=entry["sha256"],
                        )
                    )
                self.counters["journal_written"] += len(journal)
                for (day, topic), (msgs, nbytes, evts) in volume.items():
                    row = (
                        await session.execute(
                            select(RTDSDailyVolume).where(
                                RTDSDailyVolume.day == day,
                                RTDSDailyVolume.topic == topic,
                            )
                        )
                    ).scalar_one_or_none()
                    if row is None:
                        session.add(
                            RTDSDailyVolume(
                                day=day,
                                topic=topic,
                                messages=msgs,
                                bytes=nbytes,
                                events=evts,
                            )
                        )
                    else:
                        row.messages = int(row.messages or 0) + msgs
                        row.bytes = int(row.bytes or 0) + nbytes
                        row.events = int(row.events or 0) + evts
                if self._session_dirty:
                    await session.merge(
                        RTDSStreamSession(
                            session_id=self.session_id,
                            started_at=_to_dt(self._start_ms),
                            ended_at=_to_dt(self._ended_ms) if self._ended_ms else None,
                            topics=[
                                s["topic"] for s in getattr(self, "_subscriptions", [])
                            ],
                            symbols={
                                "spot": sorted(self.spot_allowlist),
                                "spot_filtered": self.spot_filtered,
                                "chainlink": self.chainlink_symbols,
                                "twap": self.twap_symbols,
                            },
                            reconnect_count=(
                                self._client.reconnect_count if self._client else 0
                            ),
                            disconnect_windows=self._gaps or None,
                        )
                    )
                    self._session_dirty = False
                await session.commit()
            except Exception as exc:
                await session.rollback()
                self._events = events + self._events
                self._journal = journal + self._journal
                for key, delta in volume.items():
                    bucket = self._volume.setdefault(key, [0, 0, 0])
                    bucket[0] += delta[0]
                    bucket[1] += delta[1]
                    bucket[2] += delta[2]
                logger.error("rtds_flush_error", error=str(exc))

    async def archive_and_purge(self) -> dict[str, Any]:
        """Rotate journal days older than retention to Parquet, then delete.

        Volume aggregates in rtds_daily_volume survive rotation; the archive
        registry records path, sha256, and row counts for audit.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from polyflip.db.models import RTDSJournalArchive, RTDSRawJournal

        today = datetime.now(timezone.utc).date()
        cutoff = retention_cutoff_day(today, self.retention_days)
        result: dict[str, Any] = {"cutoff": cutoff.isoformat(), "archived": []}
        os.makedirs(self.archive_dir, exist_ok=True)
        async with self.session_factory() as session:
            day_rows = (
                await session.execute(
                    select(sa_func.date(RTDSRawJournal.received_at)).distinct()
                )
            ).all()
            journal_days = {_coerce_day(r[0]) for r in day_rows if r[0] is not None}
            archived = {
                r[0]
                for r in (await session.execute(select(RTDSJournalArchive.day))).all()
            }
            for day in days_to_archive(journal_days, archived, cutoff):
                path = archive_file_path(self.archive_dir, day)
                schema = pa.schema(
                    [
                        ("session_id", pa.string()),
                        ("received_at", pa.string()),
                        ("topic", pa.string()),
                        ("raw_text", pa.string()),
                        ("sha256", pa.string()),
                    ]
                )
                writer = pq.ParquetWriter(path, schema)
                rows = 0
                try:
                    stream = await session.stream(
                        select(RTDSRawJournal)
                        .where(
                            RTDSRawJournal.received_at >= _day_start(day),
                            RTDSRawJournal.received_at
                            < _day_start(day + timedelta(days=1)),
                        )
                        .order_by(RTDSRawJournal.id)
                        .execution_options(yield_per=5000)
                    )
                    batch: list[dict[str, Any]] = []
                    async for part in stream.partitions():
                        for row in part:
                            item = row[0]
                            batch.append(
                                {
                                    "session_id": item.session_id,
                                    "received_at": item.received_at.isoformat(),
                                    "topic": item.topic,
                                    "raw_text": item.raw_text,
                                    "sha256": item.sha256,
                                }
                            )
                            if len(batch) >= 5000:
                                writer.write_table(
                                    pa.Table.from_pylist(batch, schema=schema)
                                )
                                rows += len(batch)
                                batch = []
                    if batch:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        rows += len(batch)
                finally:
                    writer.close()
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                session.add(
                    RTDSJournalArchive(
                        day=day, path=path, sha256=digest.hexdigest(), rows=rows
                    )
                )
                await session.execute(
                    RTDSRawJournal.__table__.delete().where(
                        RTDSRawJournal.received_at >= _day_start(day),
                        RTDSRawJournal.received_at
                        < _day_start(day + timedelta(days=1)),
                    )
                )
                await session.commit()
                result["archived"].append(
                    {"day": day.isoformat(), "rows": rows, "path": path}
                )
                logger.info("rtds_journal_archived", day=day.isoformat(), rows=rows)
        return result

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.flush_sec)
            except asyncio.CancelledError:
                break
            if self._running:
                await self._flush()

    def desired_pairs(self) -> set[tuple[str, str]]:
        """Every (topic, symbol) considered supported; confirmed only on arrival."""
        from polyflip.collector.rtds_client import normalize_symbol

        desired = {("crypto_prices", sym) for sym in self.spot_allowlist}
        for raw in self.chainlink_symbols:
            try:
                sym, _, _ = normalize_symbol(raw)
            except Exception:
                continue
            desired.add(("crypto_prices_chainlink", sym))
        for raw in self.twap_symbols:
            try:
                sym, _, _ = normalize_symbol(raw)
            except Exception:
                continue
            desired.add(("crypto_prices_twap_thirty", sym))
            desired.add(("crypto_prices_twap_sixty", sym))
        return desired

    def health(self) -> dict[str, Any]:
        """Confirmed vs UNCONFIRMED topics/symbols. Unconfirmed (e.g. DOGE)
        stays NO_DATA downstream — never a zero price, never an error."""
        unconfirmed = sorted(self.desired_pairs() - self._confirmed)
        return {
            "session": self.session_id,
            "connected": self._client._connected if self._client else False,
            "reconnects": self._client.reconnect_count if self._client else 0,
            "gaps": len(self._gaps),
            "counters": dict(self.counters),
            "confirmed_pairs": len(self._confirmed),
            "unconfirmed_pairs": [f"{t}|{s}" for t, s in unconfirmed],
        }

    async def run(self) -> None:
        import time

        self._running = True
        self._start_ms = int(time.time() * 1000)
        self._subscriptions = build_subscriptions(
            self.spot_filtered or None, self.chainlink_symbols, self.twap_symbols
        )
        self._client = RTDSClient(
            subscriptions=self._subscriptions,
            on_message=self.on_message,
            on_gap=self.on_gap,
            url=os.environ.get("RTDS_WS_URL", "wss://ws-live-data.polymarket.com"),
        )
        # Wrap gap callback to also mark session dirty via reconnect count refresh.
        original_gap = self.on_gap

        async def gap_and_dirty(from_ms: int, to_ms: int) -> None:
            await original_gap(from_ms, to_ms)

        self._client.on_gap = gap_and_dirty
        client_task = asyncio.create_task(self._client.run())
        flush_task = asyncio.create_task(self._flush_loop())
        iteration = 0
        try:
            while self._running:
                try:
                    await asyncio.sleep(15.0)
                except asyncio.CancelledError:
                    break
                try:
                    with open(ALIVE_FILE, "w", encoding="utf-8") as handle:
                        handle.write(datetime.now(timezone.utc).isoformat())
                except OSError:
                    pass
                iteration += 1
                if iteration % 4 == 0:
                    logger.info("rtds_collector_telemetry", health=self.health())
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                if now_ms - self._last_retention_ms > 3_600_000:
                    self._last_retention_ms = now_ms
                    try:
                        rotated = await self.archive_and_purge()
                        if rotated["archived"]:
                            logger.info("rtds_retention_rotated", result=rotated)
                    except Exception as exc:
                        logger.error("rtds_retention_error", error=str(exc))
        finally:
            self._running = False
            self._ended_ms = int(time.time() * 1000)
            self._client.stop()
            client_task.cancel()
            flush_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass
            try:
                await flush_task
            except asyncio.CancelledError:
                pass
            self._session_dirty = True
            await self._flush()
            logger.info(
                "rtds_collector_stopped",
                session=self.session_id,
                counters=self.counters,
            )


async def main() -> None:
    import structlog as _structlog

    _structlog.configure(
        processors=[
            _structlog.processors.TimeStamper(fmt="iso"),
            _structlog.processors.JSONRenderer(),
        ]
    )
    service = RTDSService()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass
    service_task = asyncio.create_task(service.run())
    logger.info("rtds_collector_standalone_running", session=service.session_id)
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    service._running = False
    await service_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
