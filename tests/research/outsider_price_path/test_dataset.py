"""
tests/research/outsider_price_path/test_dataset.py

Unit tests for dataset building and market processing (Stages 1, 2, 4):
- Immutable protocol hash verification
- T-5 decision snapshot selection & delay threshold (max 15s)
- Symmetric outsider detection & parity handling
- Real quote filter [0.01, 0.40]
- Unresolved market handling
- Line-item economics & fee calculation
"""
from datetime import datetime, timezone, timedelta
import pytest

from polyflip.research.outsider_price_path.dataset import (
    DEFAULT_PROTOCOL,
    ResearchProtocol,
    MarketMetadata,
    process_single_market,
    assign_ask_bin,
)


def _dt(sec: int) -> datetime:
    return datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def test_protocol_hash_invariance_and_sensitivity():
    """Verify protocol hash is deterministic and sensitive to any parameter change."""
    p1 = ResearchProtocol()
    p2 = ResearchProtocol()
    assert p1.protocol_hash == p2.protocol_hash
    assert len(p1.protocol_hash) == 64

    # Changing any parameter alters the hash
    p_mod = ResearchProtocol(ask_max=0.45)
    assert p1.protocol_hash != p_mod.protocol_hash


def test_decision_snapshot_delay_boundary():
    """
    Test T-5 boundary:
    Market expiry = 12:15:00 (sec=900). T-5 = 12:10:00 (sec=600).
    First snapshot after boundary at 605s (delay 5s <= 15s) -> OK.
    First snapshot at 620s (delay 20s > 15s) -> DELAY_TOO_LARGE.
    """
    meta = MarketMetadata(
        market_id="1001",
        asset="BTC",
        expiry=_dt(900),
        start_time=_dt(0),
        final_outcome="YES",
    )

    # Valid snapshot at 605s (delay 5s)
    snaps_valid = [
        {"recorded_at": _dt(i * 50).isoformat(), "mid_price": "0.30", "best_ask": "0.31", "best_bid": "0.29"}
        for i in range(12)
    ]
    snaps_valid.append({"recorded_at": _dt(605).isoformat(), "mid_price": "0.25", "best_ask": "0.26", "best_bid": "0.24"})
    
    rec_valid = process_single_market(meta, snaps_valid)
    assert rec_valid.selection_status == "OK"
    assert rec_valid.delay_sec == 5.0
    assert rec_valid.ask == 0.26

    # Delayed snapshot at 620s (delay 20s > 15s)
    snaps_delayed = [
        {"recorded_at": _dt(i * 50).isoformat(), "mid_price": "0.30", "best_ask": "0.31", "best_bid": "0.29"}
        for i in range(12)
    ]
    snaps_delayed.append({"recorded_at": _dt(620).isoformat(), "mid_price": "0.25", "best_ask": "0.26", "best_bid": "0.24"})
    
    rec_delayed = process_single_market(meta, snaps_delayed)
    assert rec_delayed.selection_status == "DELAY_TOO_LARGE"


def test_parity_handling():
    """Test mid == 0.50 triggers PARITY skip."""
    meta = MarketMetadata(
        market_id="1002",
        asset="ETH",
        expiry=_dt(900),
        start_time=_dt(0),
        final_outcome="YES",
    )
    snaps = [
        {"recorded_at": _dt(i * 50).isoformat(), "mid_price": "0.50", "best_ask": "0.51", "best_bid": "0.49"}
        for i in range(12)
    ]
    snaps.append({"recorded_at": _dt(605).isoformat(), "mid_price": "0.50", "best_ask": "0.50", "best_bid": "0.50"})
    rec = process_single_market(meta, snaps)
    assert rec.selection_status == "PARITY"


def test_ask_price_filter():
    """Test ask outside [0.01, 0.40] is filtered."""
    meta = MarketMetadata(
        market_id="1003",
        asset="SOL",
        expiry=_dt(900),
        start_time=_dt(0),
        final_outcome="YES",
    )
    # Mid 0.45 (< 0.50) but ask is 0.46 (> 0.40)
    snaps = [
        {"recorded_at": _dt(i * 50).isoformat(), "mid_price": "0.45", "best_ask": "0.46", "best_bid": "0.44"}
        for i in range(12)
    ]
    snaps.append({"recorded_at": _dt(605).isoformat(), "mid_price": "0.45", "best_ask": "0.46", "best_bid": "0.44"})
    rec = process_single_market(meta, snaps)
    assert rec.selection_status == "PRICE_FILTER"


def test_unresolved_market():
    """Test non-YES/NO resolution is excluded with UNRESOLVED."""
    meta = MarketMetadata(
        market_id="1004",
        asset="BTC",
        expiry=_dt(900),
        start_time=_dt(0),
        final_outcome="PENDING",
    )
    snaps = [{"recorded_at": _dt(605).isoformat(), "mid_price": "0.30", "best_ask": "0.31"}]
    rec = process_single_market(meta, snaps)
    assert rec.selection_status == "UNRESOLVED"


def test_economics_calculation():
    """Verify $1 budget line-item economics."""
    meta = MarketMetadata(
        market_id="1005",
        asset="BTC",
        expiry=_dt(900),
        start_time=_dt(0),
        final_outcome="YES",  # Win
    )
    snaps = [
        {"recorded_at": _dt(i * 50).isoformat(), "mid_price": "0.20", "best_ask": "0.20", "best_bid": "0.20"}
        for i in range(12)
    ]
    snaps.append({"recorded_at": _dt(605).isoformat(), "mid_price": "0.20", "best_ask": "0.20", "best_bid": "0.20"})
    rec = process_single_market(meta, snaps)
    assert rec.selection_status == "OK"
    assert rec.target == 1
    # budget $1.00, ask 0.20 -> shares = 5.0
    assert abs(rec.shares - 5.0) < 1e-5
    # gross_pnl = 5.0 * (1.0 - 0.20) = +4.00 USDC
    assert abs(rec.gross_pnl - 4.00) < 1e-5
    # fee_02pct = 1.0 * 0.002 = 0.002 USDC
    assert abs(rec.fee_02pct - 0.002) < 1e-5
    # net_pnl_02pct = 4.00 - 0.002 = +3.998 USDC
    assert abs(rec.net_pnl_02pct - 3.998) < 1e-5
    # scenario fee 0.1% -> 3.999 USDC
    assert abs(rec.net_pnl_01pct - 3.999) < 1e-5
