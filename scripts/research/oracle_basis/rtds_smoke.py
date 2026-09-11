"""RTDS WebSocket smoke test: all four topics, heartbeats, reconnect (read-only).

Phase 1 subscribes to the full protocol set (unfiltered spot + Chainlink spot +
TWAP 30/60) for --duration-sec, covering several 5-second heartbeat intervals.
Phase 2 drops the connection and reconnects, verifying resubscribe works.
No database, no persistence. Empty topics are reported as NO_DATA, never as
zero prices. Exit 0 requires a live connection with at least one message;
exit 2 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# Project root on sys.path (repo convention for standalone scripts).
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RTDS connectivity smoke test.")
    parser.add_argument(
        "--url",
        default=os.environ.get("RTDS_WS_URL", "wss://ws-live-data.polymarket.com"),
    )
    parser.add_argument("--duration-sec", type=float, default=45.0)
    parser.add_argument("--max-messages", type=int, default=500)
    parser.add_argument("--skip-reconnect-test", action="store_true")
    return parser


async def _ping_loop(ws, stop) -> None:
    try:
        while not stop.is_set():
            await asyncio.sleep(5.0)
            await ws.send_str("PING")
    except asyncio.CancelledError:
        pass


async def _collect(ws, budget: dict, started: float) -> None:
    import aiohttp

    from polyflip.collector.rtds_client import parse_rtds_message

    async for msg in ws:
        if not isinstance(msg, aiohttp.WSMessage) or not isinstance(msg.data, str):
            continue
        if msg.data == "PONG":
            budget["pongs"] += 1
            continue
        now_ms = int(time.time() * 1000)
        budget["messages"] += 1
        try:
            events = parse_rtds_message(msg.data, now_ms)
        except Exception as exc:  # noqa: BLE001 - smoke must report, not crash
            budget["parse_errors"].append(str(exc)[:200])
            continue
        for event in events:
            key = f"{event.topic}|{event.symbol}"
            budget["events"] += 1
            budget["per_pair"][key] = budget["per_pair"].get(key, 0) + 1
            if len(budget["samples"]) < 3:
                budget["samples"].append(msg.data[:300])
        if (
            budget["messages"] >= budget["cap"]
            or time.time() - started >= budget["duration"]
        ):
            return


async def _phase(url: str, subscriptions: list, duration: float, cap: int) -> dict:
    import aiohttp

    budget = {
        "messages": 0,
        "events": 0,
        "pongs": 0,
        "parse_errors": [],
        "per_pair": {},
        "samples": [],
        "cap": cap,
        "duration": duration,
    }
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, heartbeat=None) as ws:
            await ws.send_str(
                json.dumps({"action": "subscribe", "subscriptions": subscriptions})
            )
            stop = asyncio.Event()
            ping_task = asyncio.create_task(_ping_loop(ws, stop))
            try:
                await _collect(ws, budget, time.time())
            finally:
                stop.set()
                ping_task.cancel()
    return budget


async def smoke(url: str, duration: float, cap: int, reconnect_test: bool) -> dict:
    from polyflip.collector.rtds_client import build_subscriptions

    subscriptions = build_subscriptions(
        None,
        ["btc/usd", "eth/usd", "sol/usd", "xrp/usd"],
        ["btc/usd", "eth/usd", "sol/usd", "xrp/usd"],
    )
    phase1 = await _phase(url, subscriptions, duration, cap)
    report = {
        "topics_subscribed": sorted({s["topic"] for s in subscriptions}),
        "heartbeat_intervals_covered": round(duration / 5.0, 1),
        "phase1": _summarize(phase1),
        "reconnect": "SKIPPED",
    }
    if reconnect_test:
        # Fresh connection proves resubscribe works without server cooperation.
        phase2 = await _phase(url, subscriptions, min(duration, 20.0), 10)
        report["reconnect"] = (
            "OK" if phase2["messages"] > 0 else "RECONNECTED_NO_MESSAGES"
        )
        report["phase2"] = _summarize(phase2)
    return report


def _summarize(budget: dict) -> dict:
    topics = sorted({k.split("|")[0] for k in budget["per_pair"]})
    return {
        "messages": budget["messages"],
        "events": budget["events"],
        "pongs": budget["pongs"],
        "topics_seen": topics,
        "topics_empty_no_data": sorted(
            {
                "crypto_prices",
                "crypto_prices_chainlink",
                "crypto_prices_twap_thirty",
                "crypto_prices_twap_sixty",
            }
            - set(topics)
        ),
        "pairs": budget["per_pair"],
        "parse_errors": budget["parse_errors"][:5],
        "samples": budget["samples"],
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = asyncio.run(
            smoke(
                args.url,
                args.duration_sec,
                args.max_messages,
                not args.skip_reconnect_test,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"status": "SMOKE_FAILED", "error": f"{type(exc).__name__}: {exc}"}
            )
        )
        return 2
    ok = report["phase1"]["messages"] > 0
    print(
        json.dumps(
            {"status": "SMOKE_OK" if ok else "SMOKE_NO_DATA", **report}, indent=2
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
