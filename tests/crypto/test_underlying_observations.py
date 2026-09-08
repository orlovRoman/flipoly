"""
tests/crypto/test_underlying_observations.py

Unit tests for high-frequency underlying price observations (Step 2.1).
Verifies:
1. Duplicate observations do not alter the last available price.
2. Out-of-order messages are sorted deterministically.
3. Strict temporal causality: observations after decision_at (event_at or received_at) are never used.
4. Gaps in observations yield status='HISTORY_MISSING' instead of stale data.
5. Oracle price and Binance price are tracked as distinct named fields.
6. 30s return requires actual observations within tolerance, not minute candle fabrications.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
import numpy as np

from polyflip.crypto.underlying_observations import (
    Observation,
    get_latest_observation,
    get_underlying_state,
    compute_underlying_return,
    filter_and_order_observations,
)


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_duplicate_observations_do_not_alter_latest_price(base_time: datetime):
    """Duplicate observations must be deduplicated and preserve the exact latest price."""
    t0 = base_time
    obs_list = [
        Observation("BTC", 50000.0, "BINANCE", t0, t0),
        Observation("BTC", 50010.0, "BINANCE", t0 + timedelta(seconds=10), t0 + timedelta(seconds=10)),
        # Exact duplicates of the previous observation
        Observation("BTC", 50010.0, "BINANCE", t0 + timedelta(seconds=10), t0 + timedelta(seconds=10)),
        Observation("BTC", 50010.0, "BINANCE", t0 + timedelta(seconds=10), t0 + timedelta(seconds=10)),
    ]

    res = get_latest_observation(obs_list, as_of=t0 + timedelta(seconds=15), instrument="BTC")
    assert res.is_valid
    assert res.price == 50010.0

    filtered = filter_and_order_observations(obs_list, as_of=t0 + timedelta(seconds=15))
    assert len(filtered) == 2


def test_out_of_order_messages_sorted_deterministically(base_time: datetime):
    """Messages arriving out-of-order must be sorted chronologically by (event_at, received_at)."""
    t0 = base_time
    # Arrived in shuffled order: t+20, t+0, t+10
    obs_list = [
        Observation("BTC", 50020.0, "BINANCE", t0 + timedelta(seconds=20), t0 + timedelta(seconds=22)),
        Observation("BTC", 50000.0, "BINANCE", t0, t0 + timedelta(seconds=1)),
        Observation("BTC", 50010.0, "BINANCE", t0 + timedelta(seconds=10), t0 + timedelta(seconds=12)),
    ]

    filtered = filter_and_order_observations(obs_list, as_of=t0 + timedelta(seconds=25))
    assert [o.price for o in filtered] == [50000.0, 50010.0, 50020.0]

    # Check latest as of t+15 (should be 50010.0, not seeing 50020.0)
    res_15 = get_latest_observation(obs_list, as_of=t0 + timedelta(seconds=15), instrument="BTC")
    assert res_15.is_valid
    assert res_15.price == 50010.0


def test_strict_temporal_causality_excludes_future_events_and_receptions(base_time: datetime):
    """
    Observations with event_at > decision_at OR received_at > decision_at must be strictly excluded.
    """
    decision_at = base_time + timedelta(minutes=5)

    obs_list = [
        # Valid causal observation
        Observation("BTC", 50000.0, "BINANCE", decision_at - timedelta(seconds=5), decision_at - timedelta(seconds=4)),
        # Event happened in past, but was received AFTER decision_at (network delay / backfill)
        Observation("BTC", 50050.0, "BINANCE", decision_at - timedelta(seconds=2), decision_at + timedelta(seconds=3)),
        # Event happened in future
        Observation("BTC", 50100.0, "BINANCE", decision_at + timedelta(seconds=1), decision_at + timedelta(seconds=2)),
    ]

    res = get_latest_observation(obs_list, as_of=decision_at, instrument="BTC")
    assert res.is_valid
    assert res.price == 50000.0
    assert res.event_at == decision_at - timedelta(seconds=5)


def test_gap_in_observations_yields_history_missing(base_time: datetime):
    """If latest observation is older than max_age_seconds, status must be 'HISTORY_MISSING'."""
    t0 = base_time
    obs_list = [
        Observation("BTC", 50000.0, "BINANCE", t0, t0),
    ]

    # 30 seconds after observation: within default 60s max_age
    res_ok = get_latest_observation(obs_list, as_of=t0 + timedelta(seconds=30), max_age_seconds=60.0)
    assert res_ok.status == "VALID"
    assert res_ok.price == 50000.0

    # 120 seconds after observation: gap detected
    res_gap = get_latest_observation(obs_list, as_of=t0 + timedelta(seconds=120), max_age_seconds=60.0)
    assert res_gap.status == "HISTORY_MISSING"
    assert res_gap.price is None
    assert res_gap.age_seconds == 120.0


def test_oracle_and_binance_prices_are_distinct_named_fields(base_time: datetime):
    """Oracle price and Binance price must be kept distinct in UnderlyingState."""
    t0 = base_time
    obs_list = [
        Observation("BTC", 50000.0, "BINANCE", t0, t0),
        Observation("BTC", 50015.0, "ORACLE", t0 + timedelta(seconds=2), t0 + timedelta(seconds=2)),
    ]

    state = get_underlying_state(obs_list, as_of=t0 + timedelta(seconds=5), instrument="BTC")
    assert state.status == "VALID"
    assert state.binance_price == 50000.0
    assert state.oracle_price == 50015.0
    assert state.latest_price == 50015.0
    assert state.latest_source == "ORACLE"


def test_30s_return_requires_actual_observation_within_tolerance(base_time: datetime):
    """
    compute_underlying_return computes log(S_t / S_{t-30s}) using real observations.
    If reference observation is missing or outside tolerance, returns (None, False).
    """
    t0 = base_time
    # Observations at t=0, t=30s
    obs_list = [
        Observation("BTC", 50000.0, "BINANCE", t0, t0),
        Observation("BTC", 50500.0, "BINANCE", t0 + timedelta(seconds=30), t0 + timedelta(seconds=30)),
    ]

    # At t=30s, looking back 30s to t=0
    ret, has_ref = compute_underlying_return(
        obs_list, as_of=t0 + timedelta(seconds=30), horizon_seconds=30.0, tolerance_seconds=5.0
    )
    assert has_ref is True
    assert ret is not None
    assert ret == pytest.approx(float(np.log(50500.0 / 50000.0)), rel=1e-6)

    # If the observation at t=0 is missing (only t=30s exists)
    obs_missing = [Observation("BTC", 50500.0, "BINANCE", t0 + timedelta(seconds=30), t0 + timedelta(seconds=30))]
    ret_bad, has_ref_bad = compute_underlying_return(
        obs_missing, as_of=t0 + timedelta(seconds=30), horizon_seconds=30.0, tolerance_seconds=5.0
    )
    assert has_ref_bad is False
    assert ret_bad is None

def test_underlying_return_historical_lag_causality(base_time: datetime):
    """
    Item 13: Test that historical reference observation strictly requires received_at <= target_time.
    Candidate A: event_at = t - 29s (closer to t-30s), received_at = t - 25s (> t - 30s).
    Candidate B: event_at = t - 25s (farther from t-30s, or event_at = t - 34s), received_at = t - 34s (<= t - 30s).
    Without reception filtering, Candidate A would win on proximity/recency.
    With strict reception causality, Candidate A is discarded, and Candidate B is chosen.
    """
    t0 = base_time
    as_of = t0 + timedelta(seconds=60)
    horizon = 30.0
    # target_time for reference tick is as_of - 30s = t0 + 30s

    obs_list = [
        # Current tick at as_of (t=60s)
        Observation("BTC", 60000.0, "BINANCE", as_of, as_of),

        # Valid candidate: event_at = t0 + 26s, received_at = t0 + 26s (<= t0 + 30s)
        Observation("BTC", 50000.0, "BINANCE", t0 + timedelta(seconds=26), t0 + timedelta(seconds=26)),

        # Leaked candidate: event_at = t0 + 29s (closer to t0+30s), but received_at = t0 + 35s (> t0 + 30s)
        Observation("BTC", 55000.0, "BINANCE", t0 + timedelta(seconds=29), t0 + timedelta(seconds=35)),
    ]

    ret, has_ref = compute_underlying_return(
        obs_list, as_of=as_of, horizon_seconds=horizon, tolerance_seconds=10.0
    )

    assert has_ref is True
    assert ret is not None
    # Must use 50000 (valid causal), not 55000 (received late)
    expected_ret = float(np.log(60000.0 / 50000.0))
    assert ret == pytest.approx(expected_ret, rel=1e-6)


def test_underlying_return_late_received_current_tick(base_time: datetime):
    """
    Item 14: Test that a current tick occurring before as_of but received after as_of
    is strictly discarded and does not affect the current price or return.
    """
    t0 = base_time
    as_of = t0 + timedelta(seconds=60)
    horizon = 30.0

    obs_list = [
        # Historical reference tick at t=30s
        Observation("BTC", 50000.0, "BINANCE", t0 + timedelta(seconds=30), t0 + timedelta(seconds=30)),

        # Valid current tick at t=58s, received at t=58s (<= as_of)
        Observation("BTC", 60000.0, "BINANCE", t0 + timedelta(seconds=58), t0 + timedelta(seconds=58)),

        # Leaked current tick: event_at = t0 + 59s, but received_at = t0 + 62s (> as_of)
        Observation("BTC", 70000.0, "BINANCE", t0 + timedelta(seconds=59), t0 + timedelta(seconds=62)),
    ]

    ret, has_ref = compute_underlying_return(
        obs_list, as_of=as_of, horizon_seconds=horizon, tolerance_seconds=10.0
    )

    assert has_ref is True
    assert ret is not None
    # Must use 60000 (valid causal), not 70000 (received late)
    expected_ret = float(np.log(60000.0 / 50000.0))
    assert ret == pytest.approx(expected_ret, rel=1e-6)

