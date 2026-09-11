"""RTDS service allowlist/confirmed tracking without a database."""

import json

import pytest

from polyflip.collector.rtds_service import RTDSService


def _spot(symbol, value):
    return json.dumps(
        {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 2000,
            "payload": {"symbol": symbol, "timestamp": 1000, "value": value},
        }
    )


@pytest.mark.asyncio
async def test_allowlist_drops_unwanted_pairs_without_errors(monkeypatch):
    monkeypatch.setenv("RTDS_SPOT_SYMBOLS", "")
    monkeypatch.setenv("RTDS_SPOT_ALLOWLIST", "btcusdt,dogeusdt")
    monkeypatch.setenv("RTDS_CHAINLINK_SYMBOLS", "")
    monkeypatch.setenv("RTDS_CHAINLINK_ALLOWLIST", "btc/usd")
    monkeypatch.setenv("RTDS_TWAP_SYMBOLS", "")
    monkeypatch.setenv("RTDS_TWAP_ALLOWLIST", "btc/usd")
    service = RTDSService(session_factory=object())
    assert service.spot_filters is None  # unfiltered subscription
    assert ("crypto_prices", "DOGEUSDT") in service.desired_pairs()

    await service.on_message(_spot("dogeusdt", "1.5"), 2000)
    await service.on_message(_spot("bnbusdt", "600.0"), 2000)
    assert service.counters["events"] == 1
    assert service.counters["filtered_out"] == 1
    assert service.counters["parse_errors"] == 0
    assert len(service._events) == 1 and len(service._journal) == 1
    # Confirmation tracks arrival even for filtered pairs; health reports
    # unconfirmed desired pairs for NO_DATA (never zero prices).
    health = service.health()
    assert ("crypto_prices", "BNBUSDT") in service._confirmed
    assert "crypto_prices|BTCUSDT" in health["unconfirmed_pairs"]
    assert "crypto_prices|DOGEUSDT" not in health["unconfirmed_pairs"]
