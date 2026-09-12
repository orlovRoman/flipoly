from polyflip.research.canonical_models.dataset import common_rows


def base(mid, **kw):
    r = {"market_id": mid, "z": 0.1, "time_left_sec": 300.0,
         "sigma_per_sec": 0.001, "ret_1m": 0.0, "ret_3m": 0.0,
         "ret_5m": 0.0, "n_cross_5m": 0.0, "secs_since_cross": None,
         "d_norm_1m": 0.0, "range_pos_5m": 0.5,
         "bounce_from_min": 0.0, "pullback_from_max": 0.0}
    r.update(kw)
    return r


def test_common_rows_and_leg_rejection():
    rows = [base("m1"), base("m1"),  # duplicate market
            base("m2", leg="UP", target_up=None),  # leg-level row
            base("m3", z=None)]  # missing feature
    keep, cov = common_rows(rows)
    assert cov.n_forecast_rows == 4 and cov.n_common_rows == 1
    assert keep[0]["market_id"] == "m1"
    assert sum(cov.dropped.values()) == 3
