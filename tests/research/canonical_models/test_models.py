import math

from polyflip.research.canonical_models.dataset import M1_COLUMNS, M2_COLUMNS
from polyflip.research.canonical_models.models import (
    LogRegModel, MarketOffsetModel, m0_phi, market_proba,
)


def synth(n=60, seed=0):
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        z = rng.uniform(-2, 2)
        rows.append({
            "market_id": f"m{i}", "p_up_market": 0.2 + 0.6 * rng.random(),
            "z": z, "time_left_sec": 300.0, "sigma_per_sec": 0.001,
            "ret_1m": rng.uniform(-0.01, 0.01), "ret_3m": 0.0, "ret_5m": 0.0,
            "n_cross_5m": 0.0, "secs_since_cross": None, "d_norm_1m": 0.0,
            "range_pos_5m": 0.5, "bounce_from_min": 0.0, "pullback_from_max": 0.0,
            "target_up": 1 if z + rng.uniform(-1, 1) > 0 else 0,
        })
    return rows


def test_market_control_range_no_calibration():
    assert market_proba(0.35) == 0.35
    assert market_proba(None) is None
    assert market_proba(1.5) is None


def test_m0_phi_properties():
    assert m0_phi(0.0) == 0.5
    assert m0_phi(1.0) > m0_phi(0.0) > m0_phi(-1.0)
    assert 0.0 < (m0_phi(10.0) or 0) < 1.0
    assert m0_phi(None) is None
    assert m0_phi(float("inf")) is None


def test_m1_save_load_identical():
    rows = synth()
    m = LogRegModel(M1_COLUMNS).fit(rows)
    blob = m.dumps()
    m2 = LogRegModel.loads(blob)
    assert m.predict_proba_up(rows) == m2.predict_proba_up(rows)


def test_m3_uses_m2_columns():
    from polyflip.research.canonical_models.models import LgbmModel
    assert tuple(LgbmModel(M2_COLUMNS).columns) == tuple(M2_COLUMNS)


def test_m4_zero_correction_equals_market_is_offset_model():
    rows = synth()
    m = MarketOffsetModel(M2_COLUMNS).fit(rows)
    zeroed = m.zeroed()
    for r, p in zip(rows, zeroed.predict_proba_up(rows)):
        assert abs(p - r["p_up_market"]) < 1e-9
    # fitted model keeps market as offset: prediction moves with market
    r2 = dict(rows[0])
    r2["p_up_market"] = min(r2["p_up_market"] + 0.1, 0.95)
    assert m.predict_proba_up([r2])[0] > m.predict_proba_up([rows[0]])[0]


def test_probs_sum_to_one_side_choice_is_policy():
    rows = synth(5)
    m = LogRegModel(M1_COLUMNS).fit(synth())
    for p in m.predict_proba_up(rows):
        assert 0.0 < p < 1.0 and abs((p + (1 - p)) - 1.0) < 1e-12
        assert math.isfinite(p)
