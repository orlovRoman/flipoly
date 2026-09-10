from polyflip.research.canonical_models import ct_feature
from polyflip.research.canonical_models.ct_feature import CTResult, compute_ct


def test_missing_history_is_no_data():
    r = compute_ct([], {"source": "clob"})
    assert r.regime == "NO_DATA" and r.signal is None


def test_unwired_impl_never_invents_regime():
    ct_feature._Impl = None
    r = compute_ct([0.1, 0.2, 0.3], {"source": "clob"})
    assert r.regime == "NO_DATA"


def test_golden_reproduces_once_wired():
    def golden(hist, meta):
        return CTResult(1.0, "TREND_UP", "ct-v1-test", {}, len(hist), "test")
    ct_feature.register_ct(golden)
    try:
        r = compute_ct([0.5, 0.6, 0.7], {"source": "test"})
        assert (r.signal, r.regime, r.n_obs) == (1.0, "TREND_UP", 3)
        assert compute_ct(None) == compute_ct([])
    finally:
        ct_feature._Impl = None
