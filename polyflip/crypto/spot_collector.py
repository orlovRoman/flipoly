"""
polyflip/crypto/spot_collector.py

Continuous real-time spot tick ingestion service for underlying assets (BTC, ETH, SOL, etc.).
Ingests:
1. BINANCE spot ticks via persistent WebSocket stream (miniTicker / ticker) with auto-reconnect.
2. BINANCE REST adaptive fallback poller for continuous high-frequency coverage during reconnects.
3. ORACLE / CHAINLINK ticks from Polymarket's canonical strikes and underlying price feeds.

All ticks are buffered and idempotently committed to underlying_observations in polyflip_db
using ObservationWriter and ObservationRepository with healthcheck telemetry and watchdog supervision.
"""
from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiohttp
import httpx
import structlog

from polyflip.collector.client import PolymarketClient, _canonical_strike, _canonical_strike_provenance
from polyflip.config import settings
from polyflip.crypto.underlying_observations import (
    Observation,
    ObservationWriter,
    get_observation_writer,
)

logger = structlog.get_logger(__name__)

# Canonical mapping between asset ticker and Binance pair
ASSET_TO_BINANCE_SYMBOL: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "BNB": "BNBUSDT",
}
BINANCE_SYMBOL_TO_ASSET: dict[str, str] = {v: k for k, v in ASSET_TO_BINANCE_SYMBOL.items()}

BINANCE_WS_BASE = "wss://stream.binance.com:9443"
BINANCE_WS_BACKUP = "wss://data-stream.binance.vision"
BINANCE_REST_BASE = "https://data-api.binance.vision"
BINANCE_REST_BACKUP = "https://api.binance.com"


class BinanceSpotStream:
    """
    Persistent WebSocket listener for Binance spot ticker / miniTicker streams.
    Handles automatic reconnection with exponential backoff and jitter.
    """

    def __init__(
        self,
        assets: Sequence[str],
        writer: ObservationWriter,
        ws_url: str | None = None,
    ):
        self.assets = [a.upper() for a in assets]
        self.writer = writer
        self.ws_url = ws_url
        self._running = False
        self._connected = False
        self._last_msg_at: datetime | None = None
        self._reconnect_count = 0
        self._last_error: str | None = None

    def _build_stream_url(self, use_backup: bool = False) -> str:
        if self.ws_url:
            return self.ws_url
        stream_names = []
        for a in self.assets:
            sym = ASSET_TO_BINANCE_SYMBOL.get(a)
            if sym:
                stream_names.append(f"{sym.lower()}@miniTicker")
        if not stream_names:
            stream_names = ["btcusdt@miniTicker"]
        combined = "/".join(stream_names)
        base = BINANCE_WS_BACKUP if use_backup else BINANCE_WS_BASE
        return f"{base}/stream?streams={combined}"

    async def run(self) -> None:
        self._running = True
        attempt = 0

        while self._running:
            use_backup = (attempt > 0 and attempt % 2 == 1)
            url = self._build_stream_url(use_backup=use_backup)
            try:
                logger.info("binance_ws_connecting", url=url, attempt=attempt, use_backup=use_backup)
                timeout = aiohttp.ClientTimeout(total=None, connect=10.0, sock_read=30.0)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.ws_connect(url, heartbeat=15.0) as ws:
                        self._connected = True
                        self._reconnect_count += 1
                        attempt = 0
                        logger.info("binance_ws_connected", url=url)

                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._handle_message(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logger.warning("binance_ws_stream_closed", msg_type=msg.type)
                                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                attempt += 1
                self._last_error = str(exc)
                delay = min(15.0, 1.0 * (1.5 ** min(attempt, 8)))
                logger.warning(
                    "binance_ws_connection_failed",
                    attempt=attempt,
                    delay_sec=delay,
                    error=str(exc),
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break
            finally:
                self._connected = False

    def _handle_message(self, raw_data: str) -> None:
        try:
            payload = json.loads(raw_data)
            data = payload.get("data", payload)
            symbol = str(data.get("s", "")).upper()
            asset = BINANCE_SYMBOL_TO_ASSET.get(symbol)
            if not asset or asset not in self.assets:
                return

            price_str = data.get("c") or data.get("p")
            if price_str is None:
                return
            price = float(price_str)
            if price <= 0:
                return

            event_time_ms = data.get("E") or data.get("T")
            now_utc = datetime.now(timezone.utc)
            if event_time_ms:
                event_at = datetime.fromtimestamp(event_time_ms / 1000.0, tz=timezone.utc)
            else:
                event_at = now_utc

            self._last_msg_at = now_utc
            self.writer.record_tick(
                instrument=asset,
                price=price,
                source="BINANCE",
                event_at=event_at,
                received_at=now_utc,
                extra_data={"stream": "ws_miniTicker", "symbol": symbol},
            )
        except Exception as exc:
            logger.debug("binance_ws_message_parse_error", error=str(exc))

    def stop(self) -> None:
        self._running = False
        self._connected = False


class BinanceRestPoller:
    """
    Adaptive REST poller for Binance spot prices.
    Polls frequently (every 1.5-2.0s) if the WebSocket stream is disconnected or stale;
    throttles to lower frequency (every 5-10s) when WebSocket is streaming healthy ticks.
    Guarantees continuous tick recording even during network or firewall interruptions.
    """

    def __init__(
        self,
        assets: Sequence[str],
        writer: ObservationWriter,
        stream_ref: BinanceSpotStream | None = None,
        base_url: str = BINANCE_REST_BASE,
    ):
        self.assets = [a.upper() for a in assets]
        self.writer = writer
        self.stream_ref = stream_ref
        self.base_url = base_url
        self._running = False
        self._last_poll_at: datetime | None = None
        self._last_error: str | None = None

    async def run(self) -> None:
        self._running = True
        symbols = [ASSET_TO_BINANCE_SYMBOL[a] for a in self.assets if a in ASSET_TO_BINANCE_SYMBOL]
        client_timeout = httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=5.0)

        async with httpx.AsyncClient(timeout=client_timeout) as client:
            while self._running:
                try:
                    # Adaptive sleep based on WebSocket freshness
                    now = datetime.now(timezone.utc)
                    ws_healthy = False
                    if self.stream_ref and self.stream_ref._connected and self.stream_ref._last_msg_at:
                        age = (now - self.stream_ref._last_msg_at).total_seconds()
                        if age < 3.0:
                            ws_healthy = True

                    poll_delay = 5.0 if ws_healthy else 1.5
                    await asyncio.sleep(poll_delay)
                    if not self._running:
                        break

                    await self._poll_once(client, symbols)
                    self._last_poll_at = datetime.now(timezone.utc)
                    self._last_error = None
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self._last_error = str(exc)
                    logger.warning("binance_rest_poll_failed", error=str(exc))
                    await asyncio.sleep(2.0)

    async def _poll_once(self, client: httpx.AsyncClient, symbols: list[str]) -> None:
        now_utc = datetime.now(timezone.utc)
        endpoint = f"{self.base_url}/api/v3/ticker/price"
        backup_endpoint = f"{BINANCE_REST_BACKUP}/api/v3/ticker/price"

        data = None
        for ep in (endpoint, backup_endpoint):
            try:
                resp = await client.get(ep)
                if resp.status_code == 200:
                    data = resp.json()
                    break
            except Exception as e:
                logger.debug("binance_rest_poll_endpoint_error", endpoint=ep, error=str(e))
                continue

        if not data or not isinstance(data, list):
            return

        target_symbols = set(symbols)
        for item in data:
            sym = item.get("symbol")
            if sym in target_symbols:
                asset = BINANCE_SYMBOL_TO_ASSET.get(sym)
                if not asset:
                    continue
                try:
                    price = float(item["price"])
                except (KeyError, ValueError, TypeError):
                    continue
                if price <= 0:
                    continue

                self.writer.record_tick(
                    instrument=asset,
                    price=price,
                    source="BINANCE",
                    event_at=now_utc,
                    received_at=now_utc,
                    extra_data={"transport": "REST_POLL", "symbol": sym},
                )

    def stop(self) -> None:
        self._running = False


class PolymarketOracleCollector:
    """
    Polls Polymarket active 15m crypto markets and extracts real-time
    opening strikes and underlying prices (resolved via Chainlink).
    Emits ORACLE and CHAINLINK observations to accumulate historical strike reference data.
    """

    def __init__(
        self,
        assets: Sequence[str],
        writer: ObservationWriter,
        poll_interval_sec: float = 10.0,
    ):
        self.assets = [a.upper() for a in assets]
        self.writer = writer
        self.poll_interval_sec = poll_interval_sec
        self._running = False
        self._last_poll_at: datetime | None = None
        self._last_error: str | None = None
        self._recorded_strikes: dict[str, tuple[float, datetime]] = {}

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._poll_oracle_ticks()
                self._last_poll_at = datetime.now(timezone.utc)
                self._last_error = None
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("polymarket_oracle_poll_failed", error=str(exc))

            try:
                await asyncio.sleep(self.poll_interval_sec)
            except asyncio.CancelledError:
                break

    async def _poll_oracle_ticks(self) -> None:
        client = PolymarketClient()
        now = datetime.now(timezone.utc)
        try:
            markets = await client.get_active_15m_markets(self.assets)
            for m in markets:
                asset = m.get("asset")
                if not asset or asset not in self.assets:
                    continue

                prov = m.get("strike_provenance")
                strike_val = getattr(prov, "strike_value", None) if prov else m.get("underlying_price")
                if strike_val is None or strike_val <= 0:
                    continue

                market_id = str(m.get("market_id", ""))
                cache_key = f"{asset}:{market_id}"
                event_at = getattr(prov, "strike_effective_at", None) or now
                strike_f = float(strike_val)

                # Deduplicate: if this market's strike was already recorded and hasn't changed, skip
                if cache_key in self._recorded_strikes:
                    prev_val, prev_event = self._recorded_strikes[cache_key]
                    if abs(prev_val - strike_f) < 1e-6 and prev_event == event_at:
                        continue

                self._recorded_strikes[cache_key] = (strike_f, event_at)
                strike_src = getattr(prov, "strike_source", "UNKNOWN") if prov else "UNKNOWN"
                received_at = getattr(prov, "strike_received_at", None) or now

                # Record as ORACLE
                self.writer.record_tick(
                    instrument=asset,
                    price=strike_f,
                    source="ORACLE",
                    event_at=event_at,
                    received_at=received_at,
                    extra_data={
                        "market_id": market_id,
                        "condition_id": m.get("condition_id"),
                        "strike_source": strike_src,
                        "question": m.get("question"),
                    },
                )
                # Also record as CHAINLINK for explicit resolution attribution
                self.writer.record_tick(
                    instrument=asset,
                    price=strike_f,
                    source="CHAINLINK",
                    event_at=event_at,
                    received_at=received_at,
                    extra_data={
                        "market_id": market_id,
                        "condition_id": m.get("condition_id"),
                        "strike_source": strike_src,
                    },
                )

            if len(self._recorded_strikes) > 500:
                active_keys = {f"{m.get('asset')}:{m.get('market_id')}" for m in markets}
                self._recorded_strikes = {
                    k: v for k, v in self._recorded_strikes.items() if k in active_keys
                }
        finally:
            await client.close()

    def stop(self) -> None:
        self._running = False


class SpotCollectorDaemon:
    """
    Supervisor orchestrator for real-time spot tick ingestion.
    Coordinates:
    - ObservationWriter background batch flusher
    - BinanceSpotStream (WebSocket)
    - BinanceRestPoller (adaptive REST fallback)
    - PolymarketOracleCollector (Chainlink strike updates)
    - Healthcheck telemetry and watchdog heartbeats
    """

    def __init__(
        self,
        assets: Sequence[str] | None = None,
        session_factory=None,
        flush_interval_sec: float = 1.0,
        batch_size: int = 50,
    ):
        raw_assets = assets or getattr(settings, "asset_list", ["BTC", "ETH", "SOL", "XRP", "DOGE"])
        self.assets = [a.upper() for a in raw_assets]
        self.writer = get_observation_writer(session_factory=session_factory)
        self.writer.flush_interval_sec = flush_interval_sec
        self.writer.batch_size = batch_size

        self.ws_stream = BinanceSpotStream(assets=self.assets, writer=self.writer)
        self.rest_poller = BinanceRestPoller(
            assets=self.assets, writer=self.writer, stream_ref=self.ws_stream
        )
        self.oracle_collector = PolymarketOracleCollector(
            assets=self.assets, writer=self.writer, poll_interval_sec=10.0
        )

        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._start_time: datetime | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = datetime.now(timezone.utc)

        # Start writer background flusher
        await self.writer.start()

        # Spawn ingestion tasks
        self._tasks = [
            asyncio.create_task(self.ws_stream.run(), name="spot_binance_ws"),
            asyncio.create_task(self.rest_poller.run(), name="spot_binance_rest"),
            asyncio.create_task(self.oracle_collector.run(), name="spot_polymarket_oracle"),
            asyncio.create_task(self._watchdog_loop(), name="spot_watchdog"),
        ]

        logger.info(
            "spot_collector_daemon_started",
            assets=self.assets,
            flush_interval=self.writer.flush_interval_sec,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        self.ws_stream.stop()
        self.rest_poller.stop()
        self.oracle_collector.stop()

        for t in self._tasks:
            t.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Final writer stop and flush
        await self.writer.stop()
        logger.info("spot_collector_daemon_stopped")

    async def run(self) -> None:
        """Runs the daemon indefinitely until cancelled."""
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _watchdog_loop(self) -> None:
        """Watchdog loop writing heartbeat markers and periodic telemetry logging."""
        heartbeat_file = Path("/tmp/spot_collector_alive")
        iteration = 0

        while self._running:
            try:
                # Write heartbeat timestamp
                heartbeat_file.write_text(
                    datetime.now(timezone.utc).isoformat(), encoding="utf-8"
                )
            except Exception:
                pass

            iteration += 1
            if iteration % 4 == 0:  # Every ~60s (4 * 15s)
                health = self.get_health()
                logger.info("spot_collector_telemetry", health=health)

            try:
                await asyncio.sleep(15.0)
            except asyncio.CancelledError:
                break

    def get_health(self) -> dict[str, Any]:
        """Returns aggregated telemetry status of all subcomponents."""
        writer_health = self.writer.get_health()
        now = datetime.now(timezone.utc)
        uptime_seconds = (now - self._start_time).total_seconds() if self._start_time else 0.0

        is_healthy = (
            self._running
            and writer_health.get("healthy", False)
            and (self.ws_stream._connected or self.rest_poller._last_poll_at is not None)
        )

        return {
            "status": "HEALTHY" if is_healthy else ("IDLE" if not self._running else "DEGRADED"),
            "healthy": is_healthy,
            "is_running": self._running,
            "uptime_seconds": uptime_seconds,
            "assets": self.assets,
            "binance_ws": {
                "connected": self.ws_stream._connected,
                "reconnect_count": self.ws_stream._reconnect_count,
                "last_msg_at": self.ws_stream._last_msg_at.isoformat() if self.ws_stream._last_msg_at else None,
                "last_error": self.ws_stream._last_error,
            },
            "binance_rest": {
                "last_poll_at": self.rest_poller._last_poll_at.isoformat() if self.rest_poller._last_poll_at else None,
                "last_error": self.rest_poller._last_error,
            },
            "oracle_collector": {
                "last_poll_at": self.oracle_collector._last_poll_at.isoformat() if self.oracle_collector._last_poll_at else None,
                "last_error": self.oracle_collector._last_error,
            },
            "writer": writer_health,
        }


# Global singleton daemon
_GLOBAL_SPOT_DAEMON: SpotCollectorDaemon | None = None


def get_spot_collector_daemon(assets: Sequence[str] | None = None, session_factory=None) -> SpotCollectorDaemon:
    global _GLOBAL_SPOT_DAEMON
    if _GLOBAL_SPOT_DAEMON is None:
        _GLOBAL_SPOT_DAEMON = SpotCollectorDaemon(assets=assets, session_factory=session_factory)
    return _GLOBAL_SPOT_DAEMON


async def main():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
    from polyflip.db.connection import async_session

    daemon = get_spot_collector_daemon(session_factory=async_session)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass

    daemon_task = asyncio.create_task(daemon.run())
    logger.info("spot_collector_standalone_running")

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass

    logger.info("spot_collector_stopping")
    await daemon.stop()
    daemon_task.cancel()
    try:
        await daemon_task
    except asyncio.CancelledError:
        pass
    logger.info("spot_collector_stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
