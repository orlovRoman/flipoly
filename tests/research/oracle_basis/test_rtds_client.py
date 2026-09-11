"""RTDS wire parsing and reconnect/resubscribe behavior (no network)."""

import asyncio
import json

import pytest

from polyflip.collector.rtds_client import (
    RTDSError,
    RTDSClient,
    build_subscriptions,
    parse_rtds_message,
)


def test_twap_prefers_full_accuracy_value():
    raw = json.dumps(
        {
            "topic": "crypto_prices_twap_thirty",
            "type": "update",
            "timestamp": 1785178800123,
            "payload": {
                "symbol": "btc/usd",
                "value": 65000.5,
                "full_accuracy_value": "65000500000000000000000",
                "timestamp": 1785178800000,
                "window_s": 30,
            },
        }
    )
    (event,) = parse_rtds_message(raw, 1785178800999)
    assert event.source == "RTDS_TWAP30"
    assert (
        event.symbol == "BTC/USD" and event.asset == "BTC" and event.currency == "USD"
    )
    assert event.window_s == 30
    assert event.value_text == "65000500000000000000000"
    assert event.raw_e18 == 65000500000000000000000
    assert event.value_source == "FULL_ACCURACY"
    assert event.observed_ms == 1785178800000
    assert (event.received_ms, event.received_source) == (1785178800123, "WIRE")


def test_sixty_second_window_and_local_receipt_fallback():
    raw = json.dumps(
        {
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "payload": {
                "symbol": "eth/usd",
                "value": "3420.15",
                "full_accuracy_value": "3420150000000000000000",
                "timestamp": 1000,
            },
        }
    )
    (event,) = parse_rtds_message(raw, 1500)
    assert event.source == "RTDS_TWAP60" and event.window_s == 60
    assert event.raw_e18 == 342015 * 10**16
    assert (event.received_ms, event.received_source) == (1500, "LOCAL_CLOCK")


def test_binance_spot_numeric_value_is_exact_decimal():
    # JSON numbers are parsed with parse_float=Decimal: the wire representation
    # stays exact, no binary-float step.
    raw = json.dumps(
        {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 1782753357257,
            "payload": {
                "symbol": "btcusdt",
                "timestamp": 1782753357213,
                "value": 67234.5,
            },
        }
    )
    (event,) = parse_rtds_message(raw, 0)
    assert event.source == "RTDS_BINANCE" and event.symbol == "BTCUSDT"
    # Exact wire decimal, but NOT an E18-canonical source (server-side rounding).
    assert event.value_source == "DECIMAL_REPR"
    assert event.value_text == "67234.5"
    assert event.raw_e18 == 672345 * 10**17


def test_native_float_fallback_is_flagged():
    from polyflip.collector.rtds_client import _exact_e18

    text, raw_e18 = _exact_e18(0.5)
    assert text == "0.5"
    assert raw_e18 == 5 * 10**17


def test_twap_without_full_accuracy_raises():
    raw = json.dumps(
        {
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "timestamp": 2,
            "payload": {"symbol": "btc/usd", "timestamp": 1, "value": "65000.5"},
        }
    )
    with pytest.raises(RTDSError):
        parse_rtds_message(raw, 0)


def test_chainlink_spot_and_ignored_frames():
    raw = json.dumps(
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 7,
            "payload": {"symbol": "eth/usd", "timestamp": 6, "value": "3420.15"},
        }
    )
    (event,) = parse_rtds_message(raw, 0)
    assert event.source == "RTDS_CHAINLINK_SPOT" and event.currency == "USD"
    assert (
        parse_rtds_message(json.dumps({"topic": "equity_prices", "type": "update"}), 0)
        == []
    )
    assert (
        parse_rtds_message(
            json.dumps({"topic": "crypto_prices", "type": "subscribe"}), 0
        )
        == []
    )
    with pytest.raises(RTDSError):
        parse_rtds_message("not json", 0)
    with pytest.raises(RTDSError):
        parse_rtds_message(
            json.dumps({"topic": "crypto_prices", "type": "update", "payload": {}}), 0
        )


def test_subscription_frame_formats():
    subs = build_subscriptions(["btcusdt", "ethusdt"], ["btc/usd"], ["btc/usd"])
    by_topic = {s["topic"]: s for s in subs}
    assert by_topic["crypto_prices"]["filters"] == "btcusdt,ethusdt"
    # Exact compact JSON form required by RTDS: no spaces.
    assert by_topic["crypto_prices_chainlink"]["filters"] == '{"symbol":"btc/usd"}'
    assert by_topic["crypto_prices_twap_thirty"]["filters"] == '{"symbol":"btc/usd"}'
    assert by_topic["crypto_prices_twap_sixty"]["filters"] == '{"symbol":"btc/usd"}'
    unfiltered = build_subscriptions(None, ["btc/usd"], ["btc/usd"])
    spot_entry = next(s for s in unfiltered if s["topic"] == "crypto_prices")
    assert (
        "filters" not in spot_entry
    )  # server sends all pairs; allowlist is client-side
    with pytest.raises(RTDSError):
        build_subscriptions([], [], [])


class ScriptedWS:
    def __init__(self, script):
        self.script = script
        self.sent: list[str] = []
        self.closed = False

    async def send_str(self, data):
        self.sent.append(data)

    def __aiter__(self):
        async def gen():
            for item in self.script:
                if isinstance(item, Exception):
                    raise item
                yield item

        return gen()

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_reconnect_resubscribes_and_reports_gap():
    msg = json.dumps(
        {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 10,
            "payload": {"symbol": "btcusdt", "timestamp": 9, "value": "100"},
        }
    )
    created: list[ScriptedWS] = []
    scripts = [[msg, "PONG"], ConnectionError("boom"), [msg]]

    async def factory():
        script = scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        ws = ScriptedWS(script)
        created.append(ws)
        return ws

    received: list[str] = []
    gaps: list[tuple[int, int]] = []
    client = RTDSClient(
        subscriptions=[
            {"topic": "crypto_prices", "type": "update", "filters": "btcusdt"}
        ],
        on_message=lambda raw, at: _collect(client, received, raw),
        on_gap=lambda a, b: _gap(gaps, a, b),
        connect_factory=factory,
    )
    await client.run()
    assert len(received) == 2  # PONG heartbeat ack never delivered
    assert len(created) == 2
    subscribes = [json.loads(s) for ws in created for s in ws.sent if "subscribe" in s]
    assert len(subscribes) == 2  # resubscribed after reconnect
    assert len(gaps) == 1 and gaps[0][1] >= gaps[0][0]
    assert client.reconnect_count == 1


async def _collect(client, received, raw):
    received.append(raw)
    if len(received) >= 2:
        client.stop()


async def _gap(gaps, a, b):
    gaps.append((a, b))


class HangingWS:
    """Yields messages, then hangs until close() — like a half-open socket."""

    def __init__(self, messages):
        import asyncio

        self.messages = messages
        self.sent: list[str] = []
        self.closed = False
        self._release = asyncio.Event()

    async def send_str(self, data):
        self.sent.append(data)

    def __aiter__(self):
        async def gen():
            for item in self.messages:
                yield item
            await self._release.wait()

        return gen()

    async def close(self):
        self.closed = True
        self._release.set()


@pytest.mark.asyncio
async def test_idle_timeout_reconnects_with_gap():
    msg = json.dumps(
        {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 10,
            "payload": {"symbol": "btcusdt", "timestamp": 9, "value": "100"},
        }
    )
    calls: list[int] = []

    async def factory():
        calls.append(1)
        return HangingWS([msg])

    received: list[str] = []
    gaps: list[tuple[int, int]] = []
    client = RTDSClient(
        subscriptions=[{"topic": "crypto_prices", "type": "update"}],
        on_message=lambda raw, at: _collect(client, received, raw),
        on_gap=lambda a, b: _gap(gaps, a, b),
        connect_factory=factory,
        idle_timeout_sec=0.2,
    )
    await asyncio.wait_for(client.run(), timeout=10)
    assert len(received) == 2  # one per connection, then stop
    assert len(calls) == 2  # silent connection was dropped and redialed
    assert len(gaps) == 1 and gaps[0][1] >= gaps[0][0]
    assert client.reconnect_count == 1


def test_empty_keepalive_and_snapshot_batch_yield_no_events():
    from polyflip.collector.rtds_client import classify_frame

    assert parse_rtds_message("", 1000) == []
    assert parse_rtds_message("   ", 1000) == []
    assert classify_frame("") == "empty"
    batch = json.dumps(
        {
            "topic": "crypto_prices_twap_thirty",
            "type": "update",
            "payload": {
                "data": [
                    {
                        "full_accuracy_value": "77833738834286831730688",
                        "timestamp": 1789134201000,
                        "value": 77833.73883428684,
                    }
                ]
            },
        }
    )
    # Symbol-free batches are skipped without inference, not misattributed.
    assert parse_rtds_message(batch, 2000) == []
    assert classify_frame(batch) == "snapshot"
    assert classify_frame(json.dumps({"topic": "nope"})) == "unknown"
    assert classify_frame("{oops") == "malformed"


def test_spot_decimal_full_accuracy_falls_back_to_value():
    # Observed wire shape: Binance spot carries full_accuracy_value as a
    # DECIMAL string. It is ignored in favour of `value`, flagged in extra.
    raw = json.dumps(
        {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 1789134261127,
            "payload": {
                "full_accuracy_value": "77760.00000000",
                "symbol": "btcusdt",
                "timestamp": 1789134261000,
                "value": 77760,
            },
        }
    )
    (event,) = parse_rtds_message(raw, 0)
    assert event.value_source == "DECIMAL_REPR"
    assert event.raw_e18 == 77760 * 10**18
    assert event.extra["full_accuracy_ignored"] == "77760.00000000"


def test_chainlink_spot_e18_full_accuracy():
    raw = json.dumps(
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": 1789134261496,
            "payload": {
                "full_accuracy_value": "77736690319809985000000",
                "symbol": "btc/usd",
                "timestamp": 1789134260000,
                "value": 77736.69031980999,
            },
        }
    )
    (event,) = parse_rtds_message(raw, 0)
    assert event.source == "RTDS_CHAINLINK_SPOT"
    assert event.value_source == "FULL_ACCURACY"
    assert event.raw_e18 == 77736690319809985000000
