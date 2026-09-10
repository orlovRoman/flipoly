import math
from datetime import datetime, timedelta, timezone

from polyflip.research.canonical_models.features_base import base_features

DEC = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
END = DEC + timedelta(minutes=5)


def test_sign_and_scale():
    # underlying above strike -> positive log_distance and z
    f = base_features(105.0, 100.0, 0.001, DEC, END)
    assert f.status == "OK"
    assert f.log_distance and f.log_distance > 0 and f.z and f.z > 0
    # symmetric below-strike case flips sign with same magnitude
    g = base_features(100.0**2 / 105.0, 100.0, 0.001, DEC, END)
    assert abs(g.z + f.z) < 1e-9


def test_zero_vol_and_missing_have_no_infs():
    f = base_features(105.0, 100.0, 0.0, DEC, END)
    assert f.status == "ZERO_VOL" and f.z is None
    m = base_features(None, 100.0, 0.001, DEC, END)
    assert m.status == "MISSING" and m.z is None
    assert not math.isinf(f.z or 0)
