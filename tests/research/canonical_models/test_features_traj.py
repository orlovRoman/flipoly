from polyflip.research.canonical_models.features_traj import (
    TRAJ_COLUMNS, to_row, trajectory_features,
)

# synthetic 5m ramp 100 -> 105, decision at t=0, strike 102
SERIES = [(-300 + i * 30, 100 + 5 * (i / 10)) for i in range(11)]


def test_formulas_on_synthetic():
    f = trajectory_features(SERIES, 102.0)
    assert f.ret_5m is not None and f.ret_5m > 0
    assert f.ret_1m is not None and f.ret_1m < f.ret_5m
    assert f.n_cross_5m >= 1  # ramp crosses 102
    assert f.secs_since_cross is not None
    assert 0.0 <= (f.range_pos_5m or -1) <= 1.0
    assert set(to_row(f)) == set(TRAJ_COLUMNS)


def test_recovery_uses_only_past_minimum():
    # minimum strictly before decision; bounce measured to p0=105
    f = trajectory_features([(-300, 90.0), (-60, 95.0), (0, 105.0)], 100.0)
    assert abs((f.bounce_from_min or 0) - (105 - 90) / 90) < 1e-9
    flat = trajectory_features([(0, 100.0)], 100.0)
    assert flat.range_pos_5m is None  # flat window -> undefined, not 0/0
