import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from polyflip.crypto.trainer import CalibratedLightGBMModel
from polyflip.trading.ml_inference import ModelsCache, populate_models_cache, get_models_cache
from polyflip.db.models import ModelRegistry
import pickle


def test_calibrated_lightgbm_model_contract():
    """1.12: CalibratedLightGBMModel must expose contract properties."""
    mock_base = MagicMock()
    mock_base.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_base.classes_ = np.array([0, 1])

    mock_calib = MagicMock()
    mock_calib.predict_proba.return_value = np.array([[0.25, 0.75]])
    mock_calib.classes_ = np.array([0, 1])

    bundle = CalibratedLightGBMModel(
        raw_model=mock_base,
        calibrated_model=mock_calib,
        calibration_method="PLATT",
        ordered_feature_names=["f1", "f2", "f3"],
        target="UP",
        positive_class=1,
    )

    assert bundle.base_estimator is mock_base
    assert bundle.calibrator is mock_calib
    assert bundle.target == "UP"
    assert bundle.positive_class == 1
    assert bundle.ordered_feature_names == ("f1", "f2", "f3")
    assert np.array_equal(bundle.classes_, np.array([0, 1]))

    raw_res = bundle.predict_raw_proba(np.zeros((1, 3)))
    assert raw_res[0, 1] == 0.7

    cal_res = bundle.predict_proba(np.zeros((1, 3)))
    assert cal_res[0, 1] == 0.75


class DummyModel:
    n_features_in_ = 28
    classes_ = [0, 1]
    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])

@pytest.mark.asyncio
async def test_crypto_predictor_reads_vol_tertiles_from_training_params():
    """1.11: CryptoPredictor.load reads vol_p33 and vol_p67 from model training_params."""
    from polyflip.crypto.predictor import CryptoPredictor
    from polyflip.crypto.feature_sets import CONTROL_FEATURES

    pred = CryptoPredictor()
    mock_db = AsyncMock()

    mock_model = DummyModel()
    mock_model.n_features_in_ = len(CONTROL_FEATURES)

    row = MagicMock(spec=ModelRegistry)
    row.asset = "BTCUSDT_low_vol"
    row.version = 5
    row.interval = "15m"
    row.ece = 0.04
    row.decision_threshold = 0.55
    row.decision_threshold_down = 0.45
    row.features = ",".join(CONTROL_FEATURES)
    row.model_blob = pickle.dumps(mock_model)
    row.training_params = {
        "target_source": "POLYMARKET_FINAL_OUTCOME",
        "vol_p33": 0.4242,
        "vol_p67": 1.8888,
        "bundle_id": "test_bundle_123",
        "bundle_version": "1.1.0",
    }

    mock_scalars = MagicMock()
    mock_scalars.first.return_value = row

    async def fake_execute(stmt):
        res = MagicMock()
        res.scalars.return_value = mock_scalars
        res.scalar_one_or_none.return_value = None
        return res

    mock_db.execute = AsyncMock(side_effect=fake_execute)

    # Load symbol
    await pred.load(mock_db, "BTCUSDT")

    # Assert vol_p33 and vol_p67 were loaded from training_params!
    assert pred._vol_p33s.get("BTCUSDT") == 0.4242
    assert pred._vol_p67s.get("BTCUSDT") == 1.8888


@pytest.mark.asyncio
async def test_models_cache_disambiguation():
    """1.13: ModelsCache disambiguates models by (model_type, asset, version)."""
    mock_db = AsyncMock()

    m_logreg = DummyModel()
    m_lgbm = DummyModel()

    row_logreg = MagicMock(spec=ModelRegistry)
    row_logreg.asset = "BTC"
    row_logreg.version = 3
    row_logreg.model_type = "LogisticRegression"
    row_logreg.model_blob = pickle.dumps(m_logreg)
    row_logreg.features = "f1,f2"
    row_logreg.ece = 0.02

    row_lgbm = MagicMock(spec=ModelRegistry)
    row_lgbm.asset = "BTC"
    row_lgbm.version = 1
    row_lgbm.model_type = "LightGBM"
    row_lgbm.model_blob = pickle.dumps(m_lgbm)
    row_lgbm.features = "f1,f2,f3"
    row_lgbm.ece = 0.05

    # First execute returns active_info
    mock_active = MagicMock()
    mock_active.asset = "BTC"
    mock_active.version = 3
    mock_active.model_type = "LogisticRegression"

    mock_active2 = MagicMock()
    mock_active2.asset = "BTC"
    mock_active2.version = 1
    mock_active2.model_type = "LightGBM"

    call_count = 0
    async def fake_exec(stmt):
        nonlocal call_count
        call_count += 1
        res = MagicMock()
        if call_count == 1:
            res.all.return_value = [mock_active, mock_active2]
        else:
            mock_sc = MagicMock()
            mock_sc.all.return_value = [row_logreg, row_lgbm]
            res.scalars.return_value = mock_sc
        return res

    mock_db.execute = AsyncMock(side_effect=fake_exec)

    cache = get_models_cache()
    await populate_models_cache(mock_db)

    # Verify both entries exist without collision
    assert ("logisticregression", "BTC", 3) in cache.entries
    assert ("lightgbm", "BTC", 1) in cache.entries

    # Query by model_type
    lr_retrieved = cache.get("BTC", model_type="LogisticRegression")
    lgbm_retrieved = cache.get("BTC", model_type="LightGBM")

    assert lr_retrieved is not None
    assert lgbm_retrieved is not None
    # LogReg has precedence in legacy cache.models
    assert cache.models.get("BTC") is not None
