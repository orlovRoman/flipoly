import numpy as np
import pandas as pd

from polyflip.crypto.oof_artifact import deserialize_oof_artifact, serialize_oof_artifact


def test_oof_artifact_round_trips_raw_and_calibrated_scores():
    frame = pd.DataFrame({"market_id": ["m1", "m2"], "target": [0, 1]})
    quotes = pd.DataFrame({"market_id": ["m1", "m2"], "best_ask": [0.2, 0.8], "best_bid": [0.1, 0.7]})
    blob = serialize_oof_artifact(
        frame, np.asarray([0.25, 0.75]), quotes,
        feature_set="A", raw_scores=np.asarray([0.1, 0.9]),
    )
    payload = deserialize_oof_artifact(blob)
    np.testing.assert_allclose(payload["oof_scores"], [0.25, 0.75])
    np.testing.assert_allclose(payload["raw_oof_scores"], [0.1, 0.9])
