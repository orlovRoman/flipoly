"""Unit tests for frozen target/side/execution semantics (spec sections 4,5,21)."""
import math

import pytest

from canonical import (
    apply_residual,
    candidate_side,
    clip01,
    exec_cost,
    favorite_flip,
    favorite_side,
    market_flip,
    price_band,
    settle_net,
    trade_decision,
)


def test_truth_table_four_combos():
    assert favorite_flip("YES", "YES") == 0
    assert favorite_flip("YES", "NO") == 1
    assert favorite_flip("NO", "NO") == 0
    assert favorite_flip("NO", "YES") == 1


def test_pending_unresolved_never_zero():
    for bad in ("PENDING", "UNRESOLVED", None, "yes", ""):
        with pytest.raises(ValueError):
            favorite_flip("YES", bad)


def test_ambiguous_favorite():
    assert favorite_side(0.5) == "AMBIGUOUS"
    assert favorite_side(0.5001) == "YES"
    assert favorite_side(0.4999) == "NO"
    with pytest.raises(ValueError):
        favorite_side(0.0)
    with pytest.raises(ValueError):
        favorite_side(1.5)


def test_single_inversion_matches_outsider():
    assert candidate_side("YES") == "NO"
    assert candidate_side("NO") == "YES"
    # p_flip is never P(YES): for fav YES, P(candidate NO wins) == p_flip
    # by construction of the model output; inversion happens exactly once here.
    with pytest.raises(ValueError):
        candidate_side("AMBIGUOUS")


def test_market_control_provenance():
    p, prov = market_flip(0.3, "NO")
    assert p == pytest.approx(0.3) and prov == "MID_OBSERVED"
    p, prov = market_flip(0.7, "YES")
    assert p == pytest.approx(0.3) and prov == "MID_SYNTHETIC"


def test_execution_win_loss_both_sides():
    ask = 0.4
    cost = exec_cost(ask)
    assert cost == pytest.approx(0.4 + 0.07 * 0.4 * 0.6 + 0.005 * 0.4)
    win = settle_net("YES", True, ask)
    assert win == pytest.approx(1.0 / cost - 1.0)
    win_no = settle_net("NO", True, ask)
    assert win_no == pytest.approx(win)  # symmetric
    assert settle_net("YES", False, ask) == -1.0
    assert settle_net("NO", False, ask) == -1.0
    assert settle_net("YES", None, ask) == 0.0


def test_l3_zero_residual_identity():
    for m in (0.05, 0.3, 0.5, 0.9):
        assert apply_residual(m, 0.0) == pytest.approx(clip01(m), abs=1e-9)


def test_trade_decision_threshold():
    ask = 0.3
    edge_buy = exec_cost(ask) + 0.031
    side, edge = trade_decision(edge_buy, ask, min_edge=0.03)
    assert side == "BUY" and edge == pytest.approx(0.031)
    side, edge = trade_decision(exec_cost(ask) + 0.029, ask, min_edge=0.03)
    assert side == "SKIP"
    # ask and probability never mixed: edge is p minus cost, both in prob units
    assert math.isfinite(edge)


def test_price_bands():
    assert price_band(0.07) == "0.05-0.1"
    assert price_band(0.15) == "0.1-0.2"
    assert price_band(0.25) == "0.2-0.3"
    assert price_band(0.4) == "0.3-0.5"
    assert price_band(0.03) is None
    assert price_band(0.6) is None
