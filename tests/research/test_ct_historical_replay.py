"""
tests/research/test_ct_historical_replay.py

Tests asserting Items 6 and 7:
Item 6: All 647 opportunity IDs from the locked common opportunity ledger match exactly.
Item 7: Historical economics reproduce +81.6477 USDC with zero line-item discrepancies.
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest

from polyflip.research.ct_replay import replay_historical_ct_ledger
from polyflip.trading.ct_policy import get_btc_ct_t5_v1_spec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_historical_yes_only_replay_matches_all_647_ids():
    """Item 6 Self-Check: exactly 647 opportunities, 100% ID match, 0 missing, 0 unexpected."""
    ledger_path = REPO_ROOT / "artifacts" / "research" / "common_opportunity_ledger.json"
    assert ledger_path.exists(), f"Ledger file missing at {ledger_path}"

    res = replay_historical_ct_ledger(ledger_path)

    assert res.total_ledger_rows >= 2636
    assert res.expected_count == 647
    assert res.selected_count == 647
    assert res.matched_ids_count == 647
    assert res.id_match_exact is True
    assert len(res.missing_ids) == 0
    assert len(res.unexpected_ids) == 0


def test_historical_economics_reproduces_81_6477_usdc():
    """Item 7 Self-Check: net PnL matches +81.6477 USDC with zero line-item discrepancies."""
    ledger_path = REPO_ROOT / "artifacts" / "research" / "common_opportunity_ledger.json"
    assert ledger_path.exists(), f"Ledger file missing at {ledger_path}"

    res = replay_historical_ct_ledger(ledger_path, tolerance=1e-4)

    assert math.isclose(res.reproduced_net_pnl, 81.6477, abs_tol=1e-4)
    assert math.isclose(res.ledger_net_pnl, 81.6477, abs_tol=1e-4)
    assert res.pnl_absolute_difference < 1e-6
    assert res.max_line_item_diff < 1e-6
    assert len(res.line_item_discrepancies) == 0, f"Discrepancies found: {res.line_item_discrepancies}"
    assert res.win_count == 160
    assert math.isclose(res.win_rate, 0.2473, abs_tol=1e-3)
