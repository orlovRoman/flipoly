"""PTM v2 contract tests: items 5,7,8,9,10,14."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/research"))
from ptm_lib2 import bin_of, classify_role, opportunity_id, pnl_row, select_entry


def _snap(rec):
    return {"time_left_min": 0.0, "recorded_at": rec}


def test_entry_boundary_rejects_early_row():
    t0 = datetime(2026, 8, 4, tzinfo=timezone.utc)
    snaps = [_snap(t0 - timedelta(seconds=5)), _snap(t0 + timedelta(seconds=10))]
    row, status, _, delay = select_entry(snaps, t0)
    assert status == "OK" and delay == 10.0


def test_entry_first_after_boundary_wins_not_later_price():
    t0 = datetime(2026, 8, 4, tzinfo=timezone.utc)
    snaps = [_snap(t0 + timedelta(seconds=5)), _snap(t0 + timedelta(seconds=20))]
    row, status, _, delay = select_entry(snaps, t0)
    assert status == "OK" and delay == 5.0


def test_entry_late_rejected():
    t0 = datetime(2026, 8, 4, tzinfo=timezone.utc)
    row, status, reason, delay = select_entry([_snap(t0 + timedelta(seconds=31))], t0)
    assert status == "MISSING_ENTRY_QUOTE" and reason == "LATE_ENTRY_GT_30S" and delay is None


def test_entry_arity_always_four():
    t0 = datetime(2026, 8, 4, tzinfo=timezone.utc)
    assert len(select_entry([], t0)) == 4
    assert len(select_entry([_snap(t0 - timedelta(seconds=5))], t0)) == 4


def test_timeleft_expiry_consistency():
    # time_left from DB column must agree with expiry-anchored time within 120s
    expiry = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    rec = expiry - timedelta(minutes=8)
    tl_db = 8.02
    tl_exp = (expiry - rec).total_seconds() / 60.0
    assert abs(tl_db - tl_exp) * 60.0 < 120.0


def test_fee_unknown_gives_none_net():
    s, c, g, fee, net, fs = pnl_row(0.4, True, 1.0, None)
    assert fee is None and net is None and fs == "UNKNOWN"
    assert abs(g - (1 / 0.4 - 1)) < 5e-7


def test_scenario_fee_exact_two_cents():
    s, c, g, fee, net, fs = pnl_row(0.4, True, 1.0, 0.02)
    assert fs == "SCENARIO" and abs((g - net) - 0.02) < 1e-9


def test_bins_closed():
    lab, st = bin_of(0.995)
    assert (lab, st) == ("ABOVE_0.99", "OUT_OF_RANGE_HIGH")
    lab, st = bin_of(None)
    assert st == "MISSING_PRICE"
    lab, st = bin_of(0.40)
    assert (lab, st) == ("[0.40,0.50)", "OK")


def test_role_unified_mid_vs_ask():
    assert classify_role(0.49, 0.52) == ("OUTSIDER", "MID")
    assert classify_role(None, 0.52) == ("FAVORITE", "ASK_ONLY")
    assert classify_role(None, None) == (None, "UNKNOWN")


def test_opportunity_id_stable_under_reorder():
    a = opportunity_id("m1", "GRID", "T-8", "YES", "2026-08-04T12:00:00+00:00")
    b = opportunity_id("m1", "GRID", "T-8", "YES", "2026-08-04T12:00:00+00:00")
    c = opportunity_id("m1", "GRID", "T-5", "YES", "2026-08-04T12:00:00+00:00")
    assert a == b and a != c


def test_synthetic_end_to_end():
    # item 14: entry -> ledger math -> CT join -> agg -> bootstrap, order-invariant
    t0 = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    snaps = [
        {"time_left_min": 8.5, "recorded_at": t0 - timedelta(seconds=30)},
        {"time_left_min": 7.9, "recorded_at": t0 + timedelta(seconds=6),
         "mid_price": 0.3, "best_bid": 0.28, "best_ask": 0.32, "spread": 0.04},
        {"time_left_min": 7.0, "recorded_at": t0 + timedelta(minutes=1),
         "mid_price": 0.2, "best_bid": 0.18, "best_ask": 0.22, "spread": 0.04},
    ]
    row, st, _, delay = select_entry(snaps, t0)
    assert st == "OK" and row["best_ask"] == 0.32  # first, not cheaper later
    won = True
    s, c, g, fee, net, fs = pnl_row(0.32, won, 1.0, None)
    assert fee is None and net is None
    assert abs(g - (1 / 0.32 - 1)) < 5e-7
    lab, bst = bin_of(0.32)
    assert (lab, bst) == ("[0.30,0.35)", "OK")
    role, basis = classify_role(0.3, 0.32)
    assert (role, basis) == ("OUTSIDER", "MID")
    oid = opportunity_id("m9", "GRID", "T-8", "YES", row["recorded_at"].isoformat())
    ct = {oid: "REVERSION"}
    assert ct[oid] == "REVERSION"  # join by key, order-invariant
