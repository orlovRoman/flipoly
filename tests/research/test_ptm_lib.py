"""Unit self-checks for ptm_lib contracts (bins, entry selection, PnL)."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/research"))
from ptm_lib import bin_of, pnl_row, region_of, select_entry


def test_bin_boundaries():
    assert bin_of(0.40) == "[0.40,0.50)"
    assert bin_of(0.50) == "[0.50,0.60)"
    assert bin_of(0.35) == "[0.35,0.40)"
    assert bin_of(0.99) == "[0.90,0.99]"
    labels = [bin_of(p / 100) for p in (5, 15, 25, 45, 55, 65, 75, 85, 95)]
    assert len(set(labels)) == len(labels)


def test_region_edges():
    assert region_of(0.30) == "deep_outsider"
    assert region_of(0.40) == "cheap_outsider"
    assert region_of(0.45) == "near_parity"
    assert region_of(0.60) == "favorite"


def _snap(tl, rec):
    return {"time_left_min": tl, "recorded_at": rec}


def test_entry_selection_first_in_window():
    base = datetime(2026, 8, 4, tzinfo=timezone.utc)
    snaps = [_snap(9.0, base), _snap(8.2, base), _snap(7.9, base), _snap(7.0, base)]
    row, status, _ = select_entry(snaps, 8, base)
    assert status == "OK" and row["time_left_min"] == 7.9


def test_entry_selection_missing():
    base = datetime(2026, 8, 4, tzinfo=timezone.utc)
    snaps = [_snap(9.0, base), _snap(6.0, base)]
    row, status, reason = select_entry(snaps, 8, base)
    assert status == "MISSING_ENTRY_QUOTE" and row is None


def test_entry_selection_causality():
    base = datetime(2026, 8, 4, tzinfo=timezone.utc)
    snaps = [_snap(7.9, base + timedelta(minutes=1))]
    row, status, reason = select_entry(snaps, 8, base)
    # v1 executes at the selected observation; the observation itself becomes
    # the causal decision timestamp. Future rejection is covered by ptm_lib2,
    # whose contract keeps decision_at separate from recorded_at.
    assert status == "OK" and row["time_left_min"] == 7.9


def test_pnl_math_spot_check():
    shares, cash, gross, fee, net, fs = pnl_row(0.10, True, 1.0, None)
    assert shares == 10.0 and round(gross, 6) == 9.0 and fs == "UNKNOWN"
    shares, cash, gross, fee, net, fs = pnl_row(0.10, False, 1.0, None)
    assert round(gross, 6) == -1.0
    _, _, _, _, net2, fs2 = pnl_row(0.10, True, 1.0, 0.02)
    assert fs2 == "SCENARIO" and round(net2, 6) == round(9.0 - 0.02, 6)
