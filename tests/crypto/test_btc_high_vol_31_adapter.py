import pickle
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from polyflip.crypto.feature_sets import CONTROL_FEATURES
from polyflip.crypto.predictor import CryptoPredictor
from polyflip.crypto.trainer import CalibratedLightGBMModel
from polyflip.db.models import ModelRegistry


class LegacyBTCModel31:
    """Mock simulating legacy uncalibrated LightGBM artifact BTC_high_vol@31."""

    def __init__(self, proba: float = 0.62):
        self.proba = proba
        self.n_features_in_ = len(CONTROL_FEATURES)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[1.0 - self.proba, self.proba]])


def test_legacy_uncalibrated_bundle_adapter_contract():
    """Verify legacy uncalibrated artifact loading, raw_unavailable, and classes_ contract."""
    legacy = LegacyBTCModel31(proba=0.68)
    adapted = CalibratedLightGBMModel.from_legacy(
        legacy,
        ordered_feature_names=CONTROL_FEATURES,
        target="UP",
        positive_class=1,
    )

    # 1. Base estimator is None, calibrator is legacy model
    assert adapted.base_estimator is None
    assert adapted.calibrator is legacy
    assert adapted.target == "UP"
    assert adapted.positive_class == 1
    assert adapted.n_features_in_ == len(CONTROL_FEATURES)

    # 2. classes_ contract
    assert np.array_equal(adapted.classes_, np.array([0, 1]))

    # 3. predict_proba delegation
    X = np.zeros((1, len(CONTROL_FEATURES)))
    cal_res = adapted.predict_proba(X)
    assert np.isclose(cal_res[0, 1], 0.68)

    # 4. predict_raw_proba raises raw_unavailable
    with pytest.raises(RuntimeError, match="raw_unavailable"):
        adapted.predict_raw_proba(X)


def test_legacy_adapter_classes_contract_fallbacks():
    """Verify classes_ contract works when model has classes_ or defaults to [0, 1]."""
    class NoClassesModel:
        def __init__(self):
            self.n_features_in_ = 24

        def predict_proba(self, X):
            return np.array([[0.5, 0.5]])

    no_classes = NoClassesModel()
    adapted = CalibratedLightGBMModel.from_legacy(no_classes)
    assert np.array_equal(adapted.classes_, np.array([0, 1]))


class StandardModel:
    def __init__(self, proba: float):
        self.proba = proba
        self.n_features_in_ = len(CONTROL_FEATURES)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[1.0 - self.proba, self.proba]])

    def predict_raw_proba(self, X):
        return np.array([[1.0 - self.proba, self.proba]])


@pytest.mark.asyncio
async def test_legacy_btc_high_vol_31_inference_in_predictor():
    """Verify CryptoPredictor safely runs inference on adapted legacy BTC_high_vol@31 artifact."""
    legacy_model = LegacyBTCModel31(proba=0.72)
    adapted_bundle = CalibratedLightGBMModel.from_legacy(
        legacy_model,
        ordered_feature_names=CONTROL_FEATURES,
    )

    pred = CryptoPredictor()
    mock_db = AsyncMock()

    # BTC models: low_vol (v10), mid_vol (v10), high_vol (v31 legacy adapted)
    models_dict = {
        "low_vol": StandardModel(0.52),
        "mid_vol": StandardModel(0.55),
        "high_vol": adapted_bundle,
    }

    active_rows = {}
    for reg, ver in [("low_vol", 10), ("mid_vol", 10), ("high_vol", 31)]:
        r = MagicMock(spec=ModelRegistry)
        r.asset = f"BTCUSDT_{reg}"
        r.version = ver
        r.interval = "15m"
        r.ece = 0.04
        r.decision_threshold = 0.55
        r.decision_threshold_down = 0.45
        r.features = ",".join(CONTROL_FEATURES)
        r.model_blob = pickle.dumps(models_dict[reg])
        r.training_params = {
            "target_source": "POLYMARKET_FINAL_OUTCOME",
            "vol_p33": 0.50,
            "vol_p67": 1.50,
            "bundle_id": f"bundle_BTCUSDT_{reg}_{ver}",
        }
        active_rows[reg] = r

    def mock_db_exec(stmt):
        res = MagicMock()
        mock_sc = MagicMock()
        res.all.return_value = [
            MagicMock(asset="BTCUSDT_low_vol", version=10),
            MagicMock(asset="BTCUSDT_mid_vol", version=10),
            MagicMock(asset="BTCUSDT_high_vol", version=31),
        ]
        params = {}
        try:
            params = stmt.compile().params
        except Exception:
            pass
        matched_row = active_rows["high_vol"]
        for p_val in params.values():
            for reg in ["low_vol", "mid_vol", "high_vol"]:
                if p_val == f"BTCUSDT_{reg}":
                    matched_row = active_rows[reg]
                    break
        mock_sc.first.return_value = matched_row
        res.scalars.return_value = mock_sc
        res.scalar_one_or_none.return_value = None
        return res

    mock_db.execute = AsyncMock(side_effect=mock_db_exec)

    await pred.load(mock_db, "BTCUSDT")

    # High vol regime (vol_trend = 2.0 > vol_p67 = 1.50)
    high_vol_model = pred._models["BTCUSDT"]["high_vol"]
    assert high_vol_model is not None
    assert pred._model_versions["BTCUSDT"]["high_vol"] == 31

    # Calling predict_raw_proba directly raises raw_unavailable
    with pytest.raises(RuntimeError, match="raw_unavailable"):
        high_vol_model.predict_raw_proba(np.zeros((1, 24)))

    # But predict_proba succeeds
    p_cal = high_vol_model.predict_proba(np.zeros((1, 24)))[0, 1]
    assert np.isclose(p_cal, 0.72)
