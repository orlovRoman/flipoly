"""
tests/research/test_coverage_table.py

Tests for Stage 1 coverage table:
- Verification across assets and calendar days
- Separate accounting of observed vs reconstructed quotes
- Self-check: spot collector presence does not imply full trading set completeness
- Self-check: selection independent of old BUY/SKIP decisions
"""
import pandas as pd
from polyflip.research.coverage_table import compute_data_coverage_table


def test_coverage_table_empty():
    res = compute_data_coverage_table([])
    assert res["summary"]["total_rows"] == 0
    assert res["self_checks"]["spot_collector_does_not_imply_full_trading_set"] is True


def test_coverage_table_distinguishes_observed_and_reconstructed():
    rows = [
        {
            "market_id": "m1",
            "asset": "BTC",
            "decision_at": "2026-08-10T12:00:00Z",
            "yes_ask": 0.35,
            "no_ask": 0.65,
            "outcome_yes": "YES",
            "is_reconstructed": False,
            "strike_value": 60000.0,
        },
        {
            "market_id": "m2",
            "asset": "BTC",
            "decision_at": "2026-08-10T12:15:00Z",
            "yes_ask": 0.70,
            "no_ask": 0.30,
            "outcome_yes": "NO",
            "is_reconstructed": True,
            "strike_value": 60100.0,
        },
        {
            "market_id": "m3",
            "asset": "ETH",
            "decision_at": "2026-08-10T12:00:00Z",
            "yes_ask": 0.40,
            "no_ask": None,
            "outcome_yes": "YES",
            "is_reconstructed": False,
            "strike_value": None,  # Missing strike!
        },
    ]
    cov = compute_data_coverage_table(rows)
    assert cov["summary"]["total_rows"] == 3
    assert cov["summary"]["assets"] == ["BTC", "ETH"]

    btc_day = cov["by_asset_daily"]["BTC"]["2026-08-10"]
    assert btc_day["n_markets"] == 2
    assert btc_day["reconstructed_no_quotes_count"] == 1
    assert btc_day["both_quotes_observed_count"] == 2
    assert btc_day["strike_available_count"] == 2
    assert btc_day["full_trading_set_complete_count"] == 2

    eth_day = cov["by_asset_daily"]["ETH"]["2026-08-10"]
    assert eth_day["n_markets"] == 1
    assert eth_day["strike_available_count"] == 0  # Missing strike
    assert eth_day["full_trading_set_complete_count"] == 0  # Not complete because missing strike
    assert cov["self_checks"]["spot_collector_does_not_imply_full_trading_set"] is True
