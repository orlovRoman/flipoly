"""Proxy TWAP, reconstruction gap, flip/strike errors, moneyness."""

import pytest

from polyflip.collector.rtds_collector import E18
from polyflip.research.oracle_basis.proxy_twap import (
    InsufficientCoverageError,
    classify_vs_strike,
    flip_error,
    gap_abs_bps,
    gap_bps,
    moneyness_bps,
    proxy_direction,
    strike_error,
    summarize_gap_bps,
    twap_e18,
)


def test_twap_holds_last_pre_window_value():
    # 100 for [0,10s), 200 for [10s,60s): (100*10 + 200*50)/60 = 183.333...
    got = twap_e18([(0, 100 * E18), (10_000, 200 * E18)], 0, 60_000)
    assert got == (100 * E18 * 10_000 + 200 * E18 * 50_000 + 30_000) // 60_000


def test_twap_missing_start_is_missing_not_zero():
    with pytest.raises(InsufficientCoverageError):
        twap_e18([(5_000, 100 * E18)], 0, 60_000)
    with pytest.raises(InsufficientCoverageError):
        twap_e18([], 0, 60_000)


def test_gap_bps_definition():
    assert gap_bps(101 * E18, 100 * E18) == 100
    assert gap_bps(99 * E18, 100 * E18) == -100
    assert gap_bps(100 * E18, 100 * E18) == 0
    assert gap_abs_bps(99 * E18, 100 * E18) == 100
    # 0.5 bps rounds up (half away from zero for the magnitude).
    assert gap_bps(100 * E18 + E18 // 200, 100 * E18) == 1


def test_gap_summary_quantiles():
    summary = summarize_gap_bps([1, 2, 3, 4, 100])
    assert summary["count"] == 5
    assert summary["median"] == 3
    assert summary["min"] == 1 and summary["max"] == 100
    assert summary["p90"] == 100 and summary["p95"] == 100 and summary["p99"] == 100
    assert summarize_gap_bps([])["count"] == 0


def test_flip_error_and_tie():
    assert flip_error(100 * E18, 101 * E18, True) == {
        "proxy_direction": "UP",
        "flip_error": False,
    }
    assert flip_error(100 * E18, 101 * E18, False) == {
        "proxy_direction": "UP",
        "flip_error": True,
    }
    assert flip_error(100 * E18, 100 * E18, True) == {
        "proxy_direction": "TIE",
        "flip_error": None,
    }
    assert proxy_direction(101 * E18, 100 * E18) == "DOWN"


def test_common_strike_error_without_draw():
    assert strike_error(101 * E18, 100 * E18, True, None)["strike_error"] is False
    assert strike_error(101 * E18, 100 * E18, False, None)["strike_error"] is True
    # Verified BTC TWAP rule: equality settles UP.
    assert classify_vs_strike(100 * E18, 100 * E18, True) is True
    assert classify_vs_strike(100 * E18, 100 * E18, False) is False
    assert classify_vs_strike(100 * E18, 100 * E18, None) is None
    assert strike_error(100 * E18, 100 * E18, True, None)["strike_error"] is None


def test_moneyness_is_not_reconstruction_gap():
    assert moneyness_bps(101 * E18, 100 * E18) == 100
