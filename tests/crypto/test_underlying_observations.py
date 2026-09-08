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

def test_underlying_return_strict_causality_received_at(base_time: datetime):
    """
    Test that compute_underlying_return strictly filters out observations 
    where received_at > target_time, even if event_at <= target_time.
    """
    t0 = base_time
    target_time = t0 + timedelta(seconds=30)
    
    obs_list = [
        # Reference observation (t0)
        Observation("BTC", 50000.0, "BINANCE", t0, t0),
        
        # Valid observation at t=30s
        Observation("BTC", 50500.0, "BINANCE", target_time, target_time),
        
        # Leaked observation: event happened before target_time (t=28s) 
        # but received after target_time (t=32s). Must be discarded.
        Observation("BTC", 51000.0, "BINANCE", target_time - timedelta(seconds=2), target_time + timedelta(seconds=2)),
    ]
    
    ret, has_ref = compute_underlying_return(
        obs_list, as_of=target_time, horizon_seconds=30.0, tolerance_seconds=5.0
    )
    
    assert has_ref is True
    assert ret is not None
    # The return should be calculated using the 50500 observation, not 51000.
    # log(50500 / 50000)
    expected_ret = float(np.log(50500.0 / 50000.0))
    assert ret == pytest.approx(expected_ret, rel=1e-6)
