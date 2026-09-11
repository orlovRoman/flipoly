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
    value_source: str  # FULL_ACCURACY | NUMERIC_REPR
    observed_ms: int  # payload.timestamp
    received_ms: int  # outer timestamp, or local receipt clock
    received_source: str  # WIRE | LOCAL_CLOCK
    extra: dict[str, Any] = field(default_factory=dict)


def _wire_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RTDSError(f"invalid {name}: {value!r}")
    return value


def _exact_e18(value: Any) -> tuple[str, int, str]:
    """Exact E18 integer from a decimal wire value without binary-float arithmetic.

    Callers parse JSON with parse_float=Decimal, so decimal strings, ints, and
    JSON numbers all arrive exact (FULL_ACCURACY). A native Python float is
    accepted only defensively via repr() and flagged NUMERIC_REPR.
    NOTE: Chainlink full_accuracy_value is already an E18 integer and must go
    through _e18_int(), not here.
    """
    if isinstance(value, bool) or value is None:
        raise RTDSError(f"invalid value: {value!r}")
    if isinstance(value, Decimal):
        text, source = format(value, "f"), "FULL_ACCURACY"
    elif isinstance(value, str):
        text = value.strip()
        source = "FULL_ACCURACY"
    elif isinstance(value, int):
        text = str(value)
        source = "FULL_ACCURACY"
    elif isinstance(value, float):
        text = repr(value)
        source = "NUMERIC_REPR"
    else:
        raise RTDSError(f"invalid value: {value!r}")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise RTDSError(f"invalid decimal value: {text!r}") from exc
    if not amount.is_finite() or amount <= 0:
        raise RTDSError(f"value must be finite and positive: {text!r}")
    return text, int(amount * 10**18), source


def _e18_int(text: Any) -> tuple[str, int]:
    """Exact E18 integer from Chainlink full_accuracy_value (already E18 units)."""
    clean = str(text).strip()
    if not re.fullmatch(r"-?[0-9]+", clean):
        raise RTDSError(f"invalid full_accuracy_value: {text!r}")
    amount = int(clean)
    if amount <= 0:
        raise RTDSError(f"value must be positive: {clean!r}")
    return clean, amount


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


def parse_rtds_message(raw_text: str, received_ms: int) -> list[CanonicalEvent]:
    """Parse one RTDS text frame into canonical events (pure, no I/O).

    Unknown topics and non-update types return [] so the caller can count them
    without dropping the connection. Malformed known-topic updates raise.
    JSON numbers are parsed with parse_float=Decimal, preserving the exact wire
    decimal representation. TWAP topics require full_accuracy_value; a missing
    field raises instead of silently substituting the display price.
    """
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
    if "full_accuracy_value" in payload and payload["full_accuracy_value"] is not None:
        value_text, raw_e18 = _e18_int(payload["full_accuracy_value"])
        value_source = "FULL_ACCURACY"
    elif window_s:
        # Official TWAP without its exact E18 field: never substitute the
        # display price silently; the gap is recorded upstream as missing data.
        raise RTDSError(f"{topic}: twap update without full_accuracy_value")
    else:
        if raw_value is None:
            raise RTDSError(f"{topic}: payload has no value")
        value_text, raw_e18, value_source = _exact_e18(raw_value)
    extra: dict[str, Any] = {}
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
    chainlink_symbols: list[str],
    twap_symbols: list[str],
) -> list[dict[str, Any]]:
    """RTDS subscribe frame entries for the four protocol topics.

    spot_symbols=None subscribes to crypto_prices WITHOUT filters (the server
    sends every pair; the service allowlists client-side). This avoids depending
    on unconfirmed per-symbol filter support (e.g. dogeusdt) for the stream.
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
    for symbol in chainlink_symbols:
        clean = symbol.strip().lower()
        if clean:
            subscriptions.append(
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": '{"symbol":"' + clean + '"}',
                }
            )
    for window, topic in (
        (30, "crypto_prices_twap_thirty"),
        (60, "crypto_prices_twap_sixty"),
    ):
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
    ):
        if not subscriptions:
            raise RTDSError("RTDSClient needs at least one subscription")
        self.subscriptions = subscriptions
        self.on_message = on_message
        self.on_gap = on_gap
        self.url = url
        self.ping_interval_sec = ping_interval_sec
        self._connect_factory = connect_factory or self._default_connect
        self._running = False
        self._connected = False
        self.reconnect_count = 0
        self.messages_received = 0
        self.last_message_ms: int | None = None
        self.last_error: str | None = None

    async def _default_connect(
        self,
    ):  # pragma: no cover - exercised in smoke, not unit tests
        import aiohttp

        session = aiohttp.ClientSession()
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
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for msg in ws:
                            if not self._running:
                                break
                            data = msg.data if not isinstance(msg, str) else msg
                            if not isinstance(data, str):
                                continue
                            if data == "PONG":
                                continue
                            self.messages_received += 1
                            self.last_message_ms = _now_ms()
                            await self.on_message(data, self.last_message_ms)
                    finally:
                        ping_task.cancel()
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
