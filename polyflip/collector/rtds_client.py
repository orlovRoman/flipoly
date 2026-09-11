"""RTDS streaming client over wss://ws-live-data.polymarket.com.

Single connection, application-level PING heartbeat every 5 seconds, explicit
resubscribe after every reconnect. RTDS has no replay: disconnect windows are
reported to the caller as gaps and must be recorded, never backfilled.

Wire reference: https://docs.polymarket.com/market-data/realtime-data and
https://docs.polymarket.com/market-data/chainlink-twap.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
PING_INTERVAL_SEC = 5.0
RECONNECT_BASE_DELAY_SEC = 1.0
RECONNECT_MAX_DELAY_SEC = 30.0

# Wire topic -> (storage source, twap window seconds or 0).
TOPIC_SOURCES: dict[str, tuple[str, int]] = {
    "crypto_prices": ("RTDS_BINANCE", 0),
    "crypto_prices_chainlink": ("RTDS_CHAINLINK_SPOT", 0),
    "crypto_prices_twap_thirty": ("RTDS_TWAP30", 30),
    "crypto_prices_twap_sixty": ("RTDS_TWAP60", 60),
}


class RTDSError(ValueError):
    """Malformed RTDS message or client misuse."""


@dataclass(frozen=True)
class CanonicalEvent:
    """One parsed RTDS price update with exact integer value."""

    topic: str
    source: str
    symbol: str  # normalized: BTCUSDT | BTC/USD
    asset: str
    currency: str  # USDT | USD
    window_s: int
    value_text: str  # exact wire representation
    raw_e18: int  # exact integer E18 value
    value_source: str  # FULL_ACCURACY (TWAP E18 only) | DECIMAL_REPR (exact wire
    # decimal, not E18-canonical) | NUMERIC_REPR (native-float fallback)
    observed_ms: int  # payload.timestamp
    received_ms: int  # outer timestamp, or local receipt clock
    received_source: str  # WIRE | LOCAL_CLOCK
    extra: dict[str, Any] = field(default_factory=dict)


def _wire_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RTDSError(f"invalid {name}: {value!r}")
    return value


def _exact_e18(value: Any) -> tuple[str, int]:
    """Exact E18 integer from a decimal wire value without binary-float arithmetic.

    Callers parse JSON with parse_float=Decimal, so decimal strings, ints, and
    JSON numbers all arrive with their exact decimal representation. A native
    Python float is accepted only defensively via repr(). The caller assigns
    value_source: DECIMAL_REPR for exact decimals, NUMERIC_REPR for floats.
    NOTE: Chainlink full_accuracy_value is already an E18 integer and must go
    through _e18_int(), not here.
    """
    if isinstance(value, bool) or value is None:
        raise RTDSError(f"invalid value: {value!r}")
    if isinstance(value, Decimal):
        text = format(value, "f")
    elif isinstance(value, str):
        text = value.strip()
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        text = repr(value)
    else:
        raise RTDSError(f"invalid value: {value!r}")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise RTDSError(f"invalid decimal value: {text!r}") from exc
    if not amount.is_finite() or amount <= 0:
        raise RTDSError(f"value must be finite and positive: {text!r}")
    return text, int(amount * 10**18)


def _e18_int(text: Any) -> tuple[str, int]:
    """Exact E18 integer from Chainlink full_accuracy_value (already E18 units)."""
    clean = str(text).strip()
    if not _looks_like_int(clean):
        raise RTDSError(f"invalid full_accuracy_value: {text!r}")
    amount = int(clean)
    if amount <= 0:
        raise RTDSError(f"value must be positive: {clean!r}")
    return clean, amount


def _looks_like_int(text: str) -> bool:
    return bool(re.fullmatch(r"-?[0-9]+", str(text).strip()))


def normalize_symbol(raw: Any) -> tuple[str, str, str]:
    text = str(raw).strip()
    lower = text.lower()
    if lower.endswith("usdt") and "/" not in lower:
        asset = text[:-4].upper()
        if not asset.isalnum() or not asset:
            raise RTDSError(f"invalid symbol: {raw!r}")
        return f"{asset}USDT", asset, "USDT"
    if lower.endswith("/usd"):
        asset = text[:-4].upper()
        if not asset.isalnum() or not asset:
            raise RTDSError(f"invalid symbol: {raw!r}")
        return f"{asset}/USD", asset, "USD"
    raise RTDSError(f"unsupported symbol: {raw!r}")


_normalize_symbol = normalize_symbol  # backward-compatible alias


def classify_frame(raw_text: str) -> str:
    """Classify a raw frame without full parsing: empty | snapshot | update |
    unknown | malformed. Used for honest counters on frames that yield no events."""
    if not raw_text.strip():
        return "empty"
    try:
        message = json.loads(raw_text)
    except json.JSONDecodeError:
        return "malformed"
    if not isinstance(message, dict):
        return "malformed"
    if message.get("topic") not in TOPIC_SOURCES:
        return "unknown"
    payload = message.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return "snapshot"
    if message.get("type") == "update":
        return "update"
    return "unknown"


def parse_rtds_message(raw_text: str, received_ms: int) -> list[CanonicalEvent]:
    """Parse one RTDS text frame into canonical events (pure, no I/O).

    Empty keepalive frames, batched snapshot payloads (`payload.data`), unknown
    topics, and non-update types return [] — the caller distinguishes them with
    classify_frame() for honest counters. Snapshots are skipped deliberately:
    symbol-free batches cannot be attributed without inference, and the 1 Hz
    live singles make them redundant; warmup is covered by freshness gates.
    Malformed known-topic updates raise. JSON numbers use parse_float=Decimal.
    TWAP topics require an integer full_accuracy_value; anything else raises
    instead of substituting the display price.
    """
    if not raw_text.strip():
        return []
    try:
        message = json.loads(raw_text, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        raise RTDSError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(message, dict):
        raise RTDSError("RTDS frame must be a JSON object")
    topic = message.get("topic")
    if topic not in TOPIC_SOURCES:
        return []
    if message.get("type") != "update":
        return []
    source, window_s = TOPIC_SOURCES[topic]
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise RTDSError(f"{topic}: payload must be an object")
    if isinstance(payload.get("data"), list):
        # Symbol-free batch: skip without inference (see docstring).
        return []
    symbol, asset, currency = _normalize_symbol(payload.get("symbol"))
    observed_ms = _wire_int(payload.get("timestamp"), "payload.timestamp")
    outer_ts = message.get("timestamp")
    if isinstance(outer_ts, int) and not isinstance(outer_ts, bool) and outer_ts >= 0:
        wire_received, received_source = outer_ts, "WIRE"
    else:
        wire_received, received_source = (
            _wire_int(received_ms, "received_ms"),
            "LOCAL_CLOCK",
        )
    raw_value = payload.get("value")
    full_accuracy = payload.get("full_accuracy_value")
    extra: dict[str, Any] = {}
    if full_accuracy is not None and _looks_like_int(full_accuracy):
        # Integer full_accuracy_value: the exact E18 source (TWAP, Chainlink spot).
        value_text, raw_e18 = _e18_int(full_accuracy)
        value_source = "FULL_ACCURACY"
    else:
        if window_s:
            # Official TWAP without an integer full_accuracy_value: missing
            # data, never the display price substituted silently.
            raise RTDSError(f"{topic}: twap update without full_accuracy_value")
        if raw_value is None:
            raise RTDSError(f"{topic}: payload has no value")
        # Spot: exact decimal representation of what the wire carried — precise,
        # but NOT equivalent to an exact E18 source (server-side rounding).
        # A decimal full_accuracy_value is ignored in favour of `value`.
        value_text, raw_e18 = _exact_e18(raw_value)
        value_source = (
            "NUMERIC_REPR" if isinstance(raw_value, float) else "DECIMAL_REPR"
        )
        if full_accuracy is not None:
            extra["full_accuracy_ignored"] = str(full_accuracy)[:64]
    if window_s and "window_s" in payload:
        extra["wire_window_s"] = payload["window_s"]
    return [
        CanonicalEvent(
            topic=str(topic),
            source=source,
            symbol=symbol,
            asset=asset,
            currency=currency,
            window_s=window_s,
            value_text=value_text,
            raw_e18=raw_e18,
            value_source=value_source,
            observed_ms=observed_ms,
            received_ms=wire_received,
            received_source=received_source,
            extra=extra,
        )
    ]


def build_subscriptions(
    spot_symbols: list[str] | None,
    chainlink_symbols: list[str] | None,
    twap_symbols: list[str] | None,
) -> list[dict[str, Any]]:
    """RTDS subscribe frame entries for the four protocol topics.

    A None list subscribes WITHOUT filters (the server sends every pair for
    that topic; the service allowlists client-side). Unfiltered is the default:
    per-symbol filter support beyond the documented pairs is unconfirmed, and
    one rejected filter must never break the whole stream.
    """
    subscriptions: list[dict[str, Any]] = []
    if spot_symbols is None:
        subscriptions.append({"topic": "crypto_prices", "type": "update"})
    else:
        spot = [s.strip().lower() for s in spot_symbols if s.strip()]
        if spot:
            subscriptions.append(
                {"topic": "crypto_prices", "type": "update", "filters": ",".join(spot)}
            )
    for symbol in chainlink_symbols or []:
        clean = symbol.strip().lower()
        if clean:
            subscriptions.append(
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": '{"symbol":"' + clean + '"}',
                }
            )
    if chainlink_symbols is None:
        subscriptions.append({"topic": "crypto_prices_chainlink", "type": "*"})
    for window, topic in (
        (30, "crypto_prices_twap_thirty"),
        (60, "crypto_prices_twap_sixty"),
    ):
        if twap_symbols is None:
            subscriptions.append({"topic": topic, "type": "update"})
            continue
        for symbol in twap_symbols:
            clean = symbol.strip().lower()
            if clean:
                subscriptions.append(
                    {
                        "topic": topic,
                        "type": "update",
                        "filters": '{"symbol":"' + clean + '"}',
                    }
                )
    if not subscriptions:
        raise RTDSError("no RTDS subscriptions configured")
    return subscriptions


def _now_ms() -> int:
    return int(time.time() * 1000)


class RTDSClient:
    """Single-connection RTDS client with heartbeat, backoff, and resubscribe.

    Transport only: delivers raw text frames plus disconnect gap windows to
    callbacks. Parsing and persistence live in the service layer.
    """

    def __init__(
        self,
        subscriptions: list[dict[str, Any]],
        on_message: Callable[[str, int], Awaitable[None]],
        on_gap: Callable[[int, int], Awaitable[None]] | None = None,
        url: str = RTDS_WS_URL,
        ping_interval_sec: float = PING_INTERVAL_SEC,
        connect_factory: Callable[[], Awaitable[Any]] | None = None,
        connect_timeout_sec: float = 10.0,
        idle_timeout_sec: float = 120.0,
    ):
        if not subscriptions:
            raise RTDSError("RTDSClient needs at least one subscription")
        self.subscriptions = subscriptions
        self.on_message = on_message
        self.on_gap = on_gap
        self.url = url
        self.ping_interval_sec = ping_interval_sec
        self.connect_timeout_sec = connect_timeout_sec
        self.idle_timeout_sec = idle_timeout_sec
        self._connect_factory = connect_factory or self._default_connect
        self._running = False
        self._connected = False
        self.reconnect_count = 0
        self.messages_received = 0
        self.last_message_ms: int | None = None
        self._last_frame_ms: int | None = None
        self.last_error: str | None = None

    async def _default_connect(
        self,
    ):  # pragma: no cover - exercised in smoke, not unit tests
        import aiohttp

        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=self.connect_timeout_sec
        )
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            return await session.ws_connect(self.url, heartbeat=None)
        except Exception:
            await session.close()
            raise

    async def _ping_loop(self, ws) -> None:
        try:
            while self._running and self._connected:
                await asyncio.sleep(self.ping_interval_sec)
                if self._running and self._connected:
                    await ws.send_str("PING")
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # ping failures surface via the read loop
            logger.debug("rtds_ping_error", error=str(exc))

    async def _watchdog_loop(self, ws) -> None:
        """Close silent connections so the read loop can reconnect with a gap.

        Any frame (message or PONG) proves liveness. Without this, a half-open
        TCP connection would hang the read loop forever with no gap recorded.
        The same close unblocks reads promptly on stop().
        """
        if self.idle_timeout_sec > 0:
            interval = min(1.0, max(0.05, self.idle_timeout_sec / 4.0))
        else:
            interval = 1.0
        try:
            while self._running and self._connected:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                if (
                    self.idle_timeout_sec > 0
                    and self._last_frame_ms is not None
                    and _now_ms() - self._last_frame_ms > self.idle_timeout_sec * 1000
                ):
                    logger.warning(
                        "rtds_idle_timeout",
                        idle_sec=round((_now_ms() - self._last_frame_ms) / 1000.0, 1),
                    )
                    break
            try:
                await ws.close()
            except Exception:
                pass
        except asyncio.CancelledError:
            pass

    async def run(self) -> None:
        self._running = True
        attempt = 0
        disconnect_ms: int | None = None
        while self._running:
            try:
                ws = await self._connect_factory()
                try:
                    await ws.send_str(
                        json.dumps(
                            {"action": "subscribe", "subscriptions": self.subscriptions}
                        )
                    )
                    self._connected = True
                    attempt = 0
                    now_ms = _now_ms()
                    if disconnect_ms is not None:
                        self.reconnect_count += 1
                        if self.on_gap is not None:
                            await self.on_gap(disconnect_ms, now_ms)
                        disconnect_ms = None
                    logger.info("rtds_connected", url=self.url)
                    self._last_frame_ms = _now_ms()
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    watchdog_task = asyncio.create_task(self._watchdog_loop(ws))
                    try:
                        async for msg in ws:
                            if not self._running:
                                break
                            data = msg.data if not isinstance(msg, str) else msg
                            if not isinstance(data, str):
                                continue
                            self._last_frame_ms = _now_ms()
                            if data == "PONG":
                                continue
                            self.messages_received += 1
                            self.last_message_ms = _now_ms()
                            await self.on_message(data, self.last_message_ms)
                    finally:
                        ping_task.cancel()
                        watchdog_task.cancel()
                    # A read loop that ends while still running is a dropped
                    # connection (real transports never exhaust spontaneously):
                    # open a gap window so it is recorded, never skipped.
                    if self._running and disconnect_ms is None:
                        disconnect_ms = _now_ms()
                finally:
                    self._connected = False
                    try:
                        await ws.close()
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self.last_error = str(exc)
                attempt += 1
                if disconnect_ms is None:
                    disconnect_ms = _now_ms()
                delay = min(
                    RECONNECT_MAX_DELAY_SEC,
                    RECONNECT_BASE_DELAY_SEC * (2.0 ** min(attempt, 5)),
                )
                logger.warning(
                    "rtds_connection_failed",
                    attempt=attempt,
                    delay_sec=delay,
                    error=str(exc),
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break
        self._connected = False

    def stop(self) -> None:
        self._running = False
