"""RTDS stream validation, gap/reconnect accounting, as-of selection."""

import pytest

from polyflip.collector.rtds_collector import (
    RTDSError,
    StreamTracker,
    select_asof,
    validate_observation,
)


def raw(**over):
    base = {
        "source": "BINANCE",
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "price": "100.5",
        "observed_at": 1_000,
        "received_at": 1_200,
    }
    base.update(over)
    return base


def test_validate_ok_and_normalizes():
    obs = validate_observation(raw())
    assert (obs.source, obs.asset, obs.symbol, obs.currency) == (
        "BINANCE",
        "BTC",
        "BTCUSDT",
        "USDT",
    )
    assert obs.price_e18 == 100 * 10**18 + 5 * 10**17
    assert obs.received_at_ms >= obs.observed_at_ms


def test_validate_rejects_float_usd_mix_and_causality():
    with pytest.raises(TypeError):
        validate_observation(raw(price=100.5))
    with pytest.raises(RTDSError):
        validate_observation(raw(symbol="BTCUSD", currency="USDT"))
    with pytest.raises(RTDSError):
        validate_observation(raw(observed_at=2_000, received_at=1_000))
    with pytest.raises(RTDSError):
        validate_observation(raw(source="NOPE"))


def test_tracker_gaps_out_of_order_and_stale():
    tracker = StreamTracker(source="BINANCE", symbol="BTCUSDT", gap_threshold_ms=2000)
    assert (
        tracker.add(validate_observation(raw(observed_at=1_000, received_at=1_000)))
        == "FIRST"
    )
    assert (
        tracker.add(validate_observation(raw(observed_at=2_000, received_at=2_000)))
        == "OK"
    )
    assert (
        tracker.add(validate_observation(raw(observed_at=3_000, received_at=6_000)))
        == "GAP"
    )
    assert (
        tracker.add(validate_observation(raw(observed_at=1_500, received_at=1_500)))
        == "OUT_OF_ORDER"
    )
    summary = tracker.summary()
    assert summary["gaps_over_threshold"] == 1
    assert summary["max_gap_ms"] == 4000
    assert summary["out_of_order"] == 1
    assert tracker.stale_age_ms(9_000) == 3_000


def test_future_packet_cannot_change_past_decision():
    past = [
        validate_observation(raw(observed_at=1_000, received_at=1_000, price="100")),
        validate_observation(raw(observed_at=2_000, received_at=2_000, price="101")),
    ]
    decision_before = select_asof(past, 2_500)
    # A late packet about the past arrives after the decision was taken.
    late = validate_observation(raw(observed_at=1_500, received_at=9_000, price="999"))
    decision_after = select_asof(past, 2_500)
    assert decision_before == decision_after
    assert decision_after is not None and decision_after.price_e18 == 101 * 10**18
    assert select_asof([*past, late], 9_500) == late
    assert select_asof(past, 500) is None
