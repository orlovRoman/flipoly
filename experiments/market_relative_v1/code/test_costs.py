"""T09-T11 self-tests: fee table, VWAP walk, 3-level PnL symmetry."""
import math

import pytest

import costs as C


def test_version():
    assert C.COST_VERSION == "v1.0.0"
    assert C.FEE_RATE == 0.07


def test_fee_control_example():
    assert abs(C.fee(100, 0.30) - 1.47) < 1e-9


@pytest.mark.parametrize("price,expected_share_fee", [
    (0.01, 0.07 * 0.01 * 0.99),
    (0.10, 0.07 * 0.10 * 0.90),
    (0.30, 0.07 * 0.30 * 0.70),
    (0.50, 0.07 * 0.50 * 0.50),
    (0.70, 0.07 * 0.70 * 0.30),
    (0.90, 0.07 * 0.90 * 0.10),
    (0.99, 0.07 * 0.99 * 0.01),
])
def test_fee_table(price, expected_share_fee):
    assert abs(C.fee_per_share(price) - expected_share_fee) < 1e-12


def test_fee_symmetry_and_bounds():
    for p in (0.05, 0.2, 0.4, 0.49):
        assert abs(C.fee_per_share(p) - C.fee_per_share(1 - p)) < 1e-12
    assert C.fee_per_share(0.5) == pytest.approx(0.0175)
    assert C.fee(10, 0.0) == 0.0
    assert C.fee(10, 1.0) == 0.0


def test_walk_single_level():
    r = C.walk_asks([(0.5, 100.0)], budget=1.0)
    assert r["shares"] == pytest.approx(2.0)
    assert r["vwap"] == pytest.approx(0.5)
    assert r["spent"] == pytest.approx(1.0)
    assert r["shortfall"] is False
    assert r["levels_used"] == 1


def test_walk_multi_level():
    r = C.walk_asks([(0.5, 1.0), (0.6, 10.0)], budget=1.0)
    assert r["shares"] == pytest.approx(1.0 + 0.5 / 0.6)
    assert r["vwap"] == pytest.approx(1.0 / r["shares"])
    assert r["spent"] == pytest.approx(1.0)
    assert r["shortfall"] is False
    assert r["levels_used"] == 2


def test_walk_insufficient():
    r = C.walk_asks([(0.5, 1.0)], budget=1.0)
    assert r["spent"] == pytest.approx(0.5)
    assert r["shortfall"] is True


def test_walk_empty_and_wide_spread():
    r = C.walk_asks([], budget=1.0)
    assert r["shares"] == 0.0 and math.isnan(r["vwap"]) and r["shortfall"] is True
    r = C.walk_asks([(0.9, 100.0)], budget=1.0)
    assert r["vwap"] == pytest.approx(0.9) and r["shortfall"] is False


def test_walk_unordered_normalized():
    r = C.walk_asks([(0.6, 10.0), (0.5, 1.0)], budget=1.0)
    assert r["levels_used"] == 2
    assert r["vwap"] == pytest.approx(1.0 / (1.0 + 0.5 / 0.6))


def test_pnl_four_cases_symmetry():
    """Contract-level symmetry holds per share; $1 stakes buy different share
    counts at mirrored prices, so USDC raws differ by design."""
    p = 0.6
    yw = C.trade_pnl(1.0, p, p)
    yl = C.trade_pnl(0.0, p, p)
    nw = C.trade_pnl(1.0, 1 - p, 1 - p)
    nl = C.trade_pnl(0.0, 1 - p, 1 - p)
    assert yw["raw"] / yw["shares"] == pytest.approx(-(nl["raw"] / nl["shares"]))
    assert yl["raw"] / yl["shares"] == pytest.approx(-(nw["raw"] / nw["shares"]))
    assert yw["shares"] == pytest.approx(1.0 / p)
    assert nl["shares"] == pytest.approx(1.0 / (1 - p))


def test_pnl_identity_no_double_count():
    t = C.trade_pnl(1.0, 0.55, 0.57)
    assert t["canon_net"] == pytest.approx(t["raw"] + t["slippage"] - t["fee07"])
    assert t["slippage"] <= 0.0


def test_pnl_yes_win_numbers():
    t = C.trade_pnl(1.0, 0.5, 0.5)
    assert t["shares"] == pytest.approx(2.0)
    assert t["raw"] == pytest.approx(1.0)
    assert t["fee07"] == pytest.approx(2.0 * 0.07 * 0.25)
    assert t["canon_net"] == pytest.approx(1.0 - 2.0 * 0.07 * 0.25)
