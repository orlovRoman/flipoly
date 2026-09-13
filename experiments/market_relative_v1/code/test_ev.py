"""T19-T21 self-tests: EV symmetry, Wilson, signal gating, economics."""
import numpy as np
import pytest

import costs as C
import ev as E


def test_wilson_control():
    lo, hi = E.wilson_bounds(50, 100)
    assert 0.0 < lo < 0.5 < hi < 1.0
    lo0, hi0 = E.wilson_bounds(0, 100)
    assert lo0 == 0.0 and hi0 < 0.05
    lo1, hi1 = E.wilson_bounds(100, 100)
    assert hi1 == 1.0 and lo1 > 0.95


def test_wilson_onesided_z():
    assert abs(E.Z_95_ONE_SIDED - 1.6448536269514722) < 1e-12


def test_bins_merge_small():
    rng = np.random.RandomState(0)
    p = np.concatenate([rng.uniform(0, 1, 1900), np.full(50, 0.999)])
    y = (rng.uniform(0, 1, 1950) < p).astype(float)
    edges = E.build_bins(p, y)
    counts, _ = np.histogram(p, bins=edges)
    assert bool((counts >= 200).all())
    assert edges[0] == 0.0 and edges[-1] == 1.0


def test_ev_symmetry():
    pf, a = 0.62, 0.55
    dy = E.decide_row(pf, a, 1 - a + 0.02, np.array([0.0, 1.0]),
                      [{"lo": pf - 0.01, "hi": pf + 0.01, "n": 500, "k": 300,
                        "lo_edge": 0.0, "hi_edge": 1.0}])
    assert dy["point_yes"] == pytest.approx(pf - a - C.fee_per_share(a))
    dn = E.decide_row(1 - pf, 1 - a + 0.02, a, np.array([0.0, 1.0]),
                      [{"lo": 1 - pf - 0.01, "hi": 1 - pf + 0.01, "n": 500, "k": 300,
                        "lo_edge": 0.0, "hi_edge": 1.0}])
    assert dn["point_no"] == pytest.approx(pf - a - C.fee_per_share(a))


def test_negative_ev_abstains():
    edges = np.array([0.0, 1.0])
    table = [{"lo": 0.4, "hi": 0.6, "n": 500, "k": 250, "lo_edge": 0.0, "hi_edge": 1.0}]
    r = E.decide_row(0.5, 0.55, 0.55, edges, table)
    assert r["side"] is None and r["reason"] == "ABSTAIN_NEGATIVE_EDGE"


def test_uncertainty_gates_point():
    edges = np.array([0.0, 1.0])
    table = [{"lo": 0.50, "hi": 0.90, "n": 500, "k": 350, "lo_edge": 0.0, "hi_edge": 1.0}]
    r = E.decide_row(0.70, 0.55, 0.60, edges, table)
    assert r["point_yes"] > 0
    assert r["lower_yes"] <= r["point_yes"]
    assert r["side"] in ("YES", None)


def test_wide_interval_abstains():
    edges = np.array([0.0, 1.0])
    table = [{"lo": 0.05, "hi": 0.95, "n": 500, "k": 250, "lo_edge": 0.0, "hi_edge": 1.0}]
    r = E.decide_row(0.90, 0.40, 0.40, edges, table)
    assert r["point_yes"] > 0
    assert r["side"] is None


def test_realized_four_cases():
    yw = E.realized_hist("YES", 0.6, 0.62, 0.40, True)
    yl = E.realized_hist("YES", 0.6, 0.62, 0.40, False)
    assert yw["raw"] > 0 > yl["raw"]
    assert yw["canon_net"] == pytest.approx(
        yw["executable"] - yw["fee07"] - yw["slip_extra"])
    nw = E.realized_hist("NO", 0.6, 0.62, 0.40, False)
    assert abs(nw["raw"] / nw["shares"] - (1.0 - 0.4)) < 1e-9
    f = E.realized_forward("YES", 0.6, 0.62, True)
    assert f["canon_net"] == pytest.approx(f["executable"] - f["fee07"])
    assert "slip_extra" not in f


def test_side_mix_favorite_and_outsider():
    edges = np.array([0.0, 1.0])
    hi_table = [{"lo": 0.97, "hi": 0.995, "n": 500, "k": 490, "lo_edge": 0.0, "hi_edge": 1.0}]
    fav = E.decide_row(0.98, 0.95, 0.06, edges, hi_table)
    assert fav["side"] == "YES"
    lo_table = [{"lo": 0.01, "hi": 0.07, "n": 500, "k": 20, "lo_edge": 0.0, "hi_edge": 1.0}]
    out = E.decide_row(0.04, 0.05, 0.96, edges, lo_table)
    assert out["side"] is None
