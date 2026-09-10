import math
import pickle

import pytest

from polyflip.research.legacy_btc_logreg.pinned_loader import (
    VersionNotPinnedError,
    flip_to_side_probs,
    load_pinned,
    predict_p_flip,
)


def _blob():
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    m = LogisticRegression()
    m.classes_ = np.array([0, 1])
    m.coef_ = np.array([[-0.2, -0.26, 0.01]])
    m.intercept_ = np.array([-0.76])
    m.n_features_in_ = 3
    return pickle.dumps(m)


def _sha(b: bytes) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(b).hexdigest()


def test_missing_or_wrong_never_substitutes():
    with pytest.raises(VersionNotPinnedError):
        load_pinned(b"", _sha(_blob()), ["a"])
    with pytest.raises(VersionNotPinnedError):
        load_pinned(_blob(), "sha256:" + "0" * 64, ["a"])
    with pytest.raises(VersionNotPinnedError):
        load_pinned(_blob(), _sha(_blob()), [])


def test_p_flip_matches_logit_and_side_mapping():
    b = _blob()
    model, order = load_pinned(b, _sha(b), ["mid_price", "spread", "time_left_min"])
    row = {"mid_price": 0.3, "spread": 0.02, "time_left_min": 5.0}
    z = -0.76 - 0.2 * 0.3 - 0.26 * 0.02 + 0.01 * 5.0
    assert predict_p_flip(model, row, order) == pytest.approx(1 / (1 + math.exp(-z)))
    # DOWN favourite (mid<0.5): outsider UP wins with p_flip
    up = flip_to_side_probs(0.3, 0.4)
    assert up == {"p_up": 0.4, "p_down": 0.6, "outsider": "UP"}
    # UP favourite (mid>0.5): outsider DOWN wins with p_flip
    down = flip_to_side_probs(0.7, 0.4)
    assert down == {"p_up": 0.6, "p_down": 0.4, "outsider": "DOWN"}
    with pytest.raises(ValueError):
        flip_to_side_probs(0.5, 0.4)
