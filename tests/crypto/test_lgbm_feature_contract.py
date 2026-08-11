import pickle
import numpy as np
import pytest

from polyflip.crypto.feature_sets import (
    CONTROL_FEATURES,
    FEATURE_SETS,
    feature_schema_hash,
    normalize_feature_set,
    parse_feature_names,
    validate_feature_schema,
)
from polyflip.crypto.trainer import _dataset_fingerprint, _model_smoke_test


def test_control_schema_is_stable_and_hashable():
    assert normalize_feature_set("AUTO") == "A"
    assert len(CONTROL_FEATURES) == len(set(CONTROL_FEATURES))
    assert validate_feature_schema(CONTROL_FEATURES) == CONTROL_FEATURES
    assert feature_schema_hash(CONTROL_FEATURES) == feature_schema_hash(tuple(CONTROL_FEATURES))


class _SchemaModel:
    n_features_in_ = 2

    def predict_proba(self, rows):
        return np.asarray([[0.4, 0.6]] * len(rows))


def test_model_smoke_test_uses_artifact_schema():
    assert _model_smoke_test(pickle.dumps(_SchemaModel()), ("ret_1", "ret_3")) is None
    error = _model_smoke_test(pickle.dumps(_SchemaModel()), ("ret_1",))
    assert "expected=1" in error


def test_invalid_schema_is_rejected():
    with pytest.raises(ValueError, match="Unknown crypto model features"):
        validate_feature_schema(("ret_1", "future_feature"))
    with pytest.raises(ValueError, match="duplicate"):
        parse_feature_names("ret_1,ret_1")


def test_dataset_fingerprint_is_independent_of_row_order():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"market_id": "m2", "market_start": "2026-01-01T00:15:00Z", "target": 0, "ret_1": 0.2},
            {"market_id": "m1", "market_start": "2026-01-01T00:00:00Z", "target": 1, "ret_1": -0.1},
        ]
    )
    assert _dataset_fingerprint(frame, ["ret_1"]) == _dataset_fingerprint(frame.iloc[::-1], ["ret_1"])


def test_lightgbm_variants_are_incremental_and_named():
    assert FEATURE_SETS["A"].version == "A-control-v1"
    assert FEATURE_SETS["B"].version == "B-direction-sequence-v1"
    assert FEATURE_SETS["C"].version == "C-candle-structure-v1"
    assert set(FEATURE_SETS["A"].features).issubset(FEATURE_SETS["B"].features)
    assert set(FEATURE_SETS["B"].features).issubset(FEATURE_SETS["C"].features)
    assert len(FEATURE_SETS["B"].features) > len(FEATURE_SETS["A"].features)
    assert len(FEATURE_SETS["C"].features) > len(FEATURE_SETS["B"].features)
    assert validate_feature_schema(FEATURE_SETS["C"].features) == FEATURE_SETS["C"].features


def test_control_features_match_builder_schema():
    from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS

    assert set(CONTROL_FEATURES).issubset(set(CRYPTO_FEATURE_COLUMNS))
