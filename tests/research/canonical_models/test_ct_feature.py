from datetime import datetime, timedelta, timezone

from polyflip.research.canonical_models.ct_feature import (
    CT_SPEC_ID, compute_ct, ct_allows_entry,
)
from polyflip.trading.ct_policy import get_btc_ct_t5_v1_spec

DEC = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)


def _saw(n=30):
    return [(DEC - timedelta(seconds=900 - i * 30),
             0.5 + (0.2 if i % 2 else -0.2)) for i in range(n)]


def _trend(n=30):
    return [(DEC - timedelta(seconds=900 - i * 30), 0.3 + 0.01 * i)
            for i in range(n)]


def test_golden_saw_is_reversion_trend_is_not():
    # Fixed spec, fixed wrapper params (min_observations=3 from spec, NOT the
    # classifier default of 4): deterministic golden.
    r = compute_ct(_saw(), {"asset": "BTC", "source": "test"}, DEC)
    assert (r.regime, r.status, r.version) == ("REVERSION", "VALID", CT_SPEC_ID)
    assert r.n_obs == 30 and r.spec_hash == get_btc_ct_t5_v1_spec().spec_hash
    assert ct_allows_entry(r) is True
    t = compute_ct(_trend(), {"asset": "BTC"}, DEC)
    assert (t.regime, t.status) == ("TREND", "VALID")
    assert ct_allows_entry(t) is False


def test_missing_out_of_scope_and_no_boundary_never_trend():
    assert compute_ct([], {"asset": "BTC"}, DEC).status == "NO_HISTORY"
    assert compute_ct(_saw(), {"asset": "ETH"}, DEC).status == "ASSET_OUT_OF_SCOPE"
    # no causal boundary -> refuse instead of inventing a timestamp
    assert compute_ct(_saw(), {"asset": "BTC"}, None).status == "NO_HISTORY"
    for res in (compute_ct([], {"asset": "BTC"}, DEC),
                compute_ct(_saw(), {"asset": "ETH"}, DEC)):
        assert res.regime == "UNCERTAIN" and ct_allows_entry(res) is False
