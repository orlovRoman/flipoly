from polyflip.research.canonical_models.economics import (
    breakeven_fee_rate, execute, scenario_pnls,
)
from polyflip.research.canonical_models.policies import (
    Opportunity, decide, ev_net, pick_outsider_side,
)


def test_win_loss_partial_reconcile():
    w = execute(1.0, 0.2, 0.0, True)
    assert abs(w.net_pnl - 4.0) < 1e-9  # 5 shares * $1 payout + 0 remainder - $1 budget
    l = execute(1.0, 0.2, 0.0, False)
    assert abs(l.net_pnl - (-1.0)) < 1e-9
    p = execute(1.0, 0.2, 0.0, True, size_limit=2.0)
    assert p.status == "PARTIAL" and abs(p.shares - 2.0) < 1e-9


def test_spread_not_double_charged_gas_separate():
    # entry at ask already pays the spread; economics must not subtract it again
    a = execute(1.0, 0.25, 0.0, True)
    assert abs(a.cost - 1.0) < 1e-9  # full budget into shares at ask
    g = execute(1.0, 0.25, 0.0, True, gas=0.01)
    assert abs((a.net_pnl or 0) - (g.net_pnl or 0) - 0.01) < 1e-9


def test_unknown_fee_null_and_blocked_not_zero():
    u = execute(1.0, 0.2, None, True)
    assert u.net_pnl is None and u.status == "UNKNOWN_FEE"
    b = execute(1.0, None, 0.0, True)
    assert b.net_pnl is None and b.status == "BLOCKED_DATA"
    assert b.payout == 0 and b.shares == 0


def test_fee_scenarios_breakeven_and_top_only_flag():
    sc = scenario_pnls(1.0, 0.2, True)
    assert set(sc) == {0.0, 0.001, 0.002}
    assert sc[0.0] == 4.0 and sc[0.002] < sc[0.0]
    # breakeven: p/ask - 1
    assert abs((breakeven_fee_rate(0.8, 0.2) or 0) - 3.0) < 1e-9
    f = execute(1.0, 0.2, 0.0, True)
    assert f.data_status == "TOP_ONLY_ASSUMED" and f.fee_status == "CONFIRMED"
    u = execute(1.0, 0.2, None, True)
    assert u.fee_status == "UNKNOWN" and u.gross_pnl == 4.0


def test_side_selection_parity_and_no_lookahead():
    opp = Opportunity("m", 0.3, 0.7, 0.3, 0.7, 0.3, 0.8, "NO_DATA")
    side, _ = pick_outsider_side(opp)
    assert side == "UP"  # lower mid, even though DOWN won ex-post
    par = Opportunity("m", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert decide("M1", par, 0.02).reason == "SKIP_PARITY"
    # EV gate uses the pre-registered threshold, identical for all model policies
    assert ev_net(0.8, 0.3) > 0.02
    d = decide("M1", opp, 10.0)
    assert d.trade is False and d.reason == "EV_BELOW_THRESHOLD"
