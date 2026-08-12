import gzip
import pickle

import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.oof_artifact import (
    deserialize_oof_artifact,
    serialize_oof_artifact,
)


def test_oof_artifact_round_trip_preserves_alignment_and_quotes():
    frame = pd.DataFrame({
        "market_id": ["m1", "m2"],
        "market_start": pd.to_datetime(["2026-08-01T00:00:00Z", "2026-08-01T00:15:00Z"], utc=True),
        "target": [0, 1],
        "final_outcome": ["NO", "YES"],
    })
    quotes = pd.DataFrame({
        "market_id": ["m1", "m2"],
        "best_bid": [0.39, 0.79],
        "best_ask": [0.41, 0.81],
    })
    blob = serialize_oof_artifact(
        frame,
        np.asarray([np.nan, 0.8]),
        quotes,
        feature_set="B",
        feature_schema_hash="abc",
    )
    payload = deserialize_oof_artifact(blob)
    assert payload["feature_set"] == "B"
    assert payload["feature_schema_hash"] == "abc"
    assert payload["frame"]["market_id"].tolist() == ["m1", "m2"]
    assert np.isnan(payload["oof_scores"][0])
    assert payload["oof_scores"][1] == pytest.approx(0.8)
    assert payload["quotes"]["best_ask"].tolist() == pytest.approx([0.41, 0.81])


def test_oof_artifact_rejects_misaligned_scores():
    frame = pd.DataFrame({"market_id": ["m1"], "target": [1]})
    with pytest.raises(ValueError, match="align"):
        serialize_oof_artifact(frame, [0.1, 0.2], None, feature_set="A")


def test_oof_artifact_rejects_legacy_pickle_payload():
    legacy = gzip.compress(pickle.dumps({"schema_version": 2}, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(ValueError, match="JSON|schema"):
        deserialize_oof_artifact(legacy)


def test_schema_version_matches_registry_column_default():
    from polyflip.crypto.oof_artifact import OOF_ARTIFACT_SCHEMA_VERSION
    from polyflip.db.models import ModelRegistryOOFArtifact
    default = ModelRegistryOOFArtifact.__table__.c.schema_version.server_default
    assert str(OOF_ARTIFACT_SCHEMA_VERSION) == str(default.arg)
