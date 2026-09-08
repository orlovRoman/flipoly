import pickle
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pandas as pd
import pytest

from polyflip.db.models import ModelRegistry
from polyflip.trading.ml_inference import (
    ModelsCache,
    get_models_cache,
    populate_models_cache,
    run_model_inference,
)


class MockMLModel:
    def __init__(self, n_features: int, proba: float = 0.65):
        self.n_features_in_ = n_features
        self.proba = proba
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[1.0 - self.proba, self.proba]])


def test_models_cache_dual_active_models_disjoint_features():
    """Verify ModelsCache stores and retrieves disjoint feature lists for dual active models on BTC."""
    cache = ModelsCache(
        models={},
        versions={},
        features={},
        eces={},
        entries={},
        features_by_type={},
        features_by_entry={},
    )

    m_logreg = MockMLModel(2, 0.58)
    m_lgbm = MockMLModel(4, 0.72)

    features_logreg = ["ret_1", "vol_6"]
    features_lgbm = ["ret_1", "vol_6", "rsi_14", "cvd_1"]

    # Register both models for BTC
    cache.put("BTC", m_logreg, model_type="LogisticRegression", version=3, features=features_logreg, ece=0.02)
    cache.put("BTC", m_lgbm, model_type="LightGBM", version=1, features=features_lgbm, ece=0.04)

    # 1. Feature retrieval by model_type
    retrieved_lr_feats = cache.get_features("BTC", model_type="LogisticRegression")
    retrieved_lgbm_feats = cache.get_features("BTC", model_type="LightGBM")

    assert retrieved_lr_feats == ["ret_1", "vol_6"]
    assert retrieved_lgbm_feats == ["ret_1", "vol_6", "rsi_14", "cvd_1"]

    # 2. Case-insensitivity
    assert cache.get_features("BTC", model_type="logreg") == ["ret_1", "vol_6"]
    assert cache.get_features("BTC", model_type="lgbm") == ["ret_1", "vol_6", "rsi_14", "cvd_1"]

    # 3. Version-specific retrieval
    assert cache.get_features("BTC", model_type="logreg", version=3) == ["ret_1", "vol_6"]
    assert cache.get_features("BTC", model_type="lightgbm", version=1) == ["ret_1", "vol_6", "rsi_14", "cvd_1"]

    # 4. Model object retrieval without collision
    assert cache.get("BTC", model_type="LogisticRegression") is m_logreg
    assert cache.get("BTC", model_type="LightGBM") is m_lgbm


@pytest.mark.asyncio
async def test_populate_models_cache_dual_active_btc_models():
    """Verify populate_models_cache properly populates both LogReg and LightGBM without collision."""
    mock_db = AsyncMock()

    m_lr = MockMLModel(3, 0.55)
    m_lgbm = MockMLModel(5, 0.70)

    row_lr = MagicMock(spec=ModelRegistry)
    row_lr.asset = "BTC"
    row_lr.version = 5
    row_lr.model_type = "LogisticRegression"
    row_lr.model_blob = pickle.dumps(m_lr)
    row_lr.features = "f1,f2,f3"
    row_lr.ece = 0.015

    row_lgbm = MagicMock(spec=ModelRegistry)
    row_lgbm.asset = "BTC"
    row_lgbm.version = 2
    row_lgbm.model_type = "LightGBM"
    row_lgbm.model_blob = pickle.dumps(m_lgbm)
    row_lgbm.features = "f1,f2,f3,f4,f5"
    row_lgbm.ece = 0.035

    active_meta_lr = MagicMock(asset="BTC", version=5, model_type="LogisticRegression")
    active_meta_lgbm = MagicMock(asset="BTC", version=2, model_type="LightGBM")

    call_count = 0
    async def fake_db_exec(stmt):
        nonlocal call_count
        call_count += 1
        res = MagicMock()
        if call_count == 1:
            res.all.return_value = [active_meta_lr, active_meta_lgbm]
        else:
            mock_sc = MagicMock()
            mock_sc.all.return_value = [row_lr, row_lgbm]
            res.scalars.return_value = mock_sc
        return res

    mock_db.execute = AsyncMock(side_effect=fake_db_exec)

    cache = get_models_cache()
    await populate_models_cache(mock_db)

    # Features must be correctly separated
    assert cache.get_features("BTC", model_type="LogisticRegression") == ["f1", "f2", "f3"]
    assert cache.get_features("BTC", model_type="LightGBM") == ["f1", "f2", "f3", "f4", "f5"]

    # Inference check for each
    df = pd.DataFrame([{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}])
    
    m_lr_retrieved = cache.get("BTC", model_type="LogisticRegression")
    feats_lr = cache.get_features("BTC", model_type="LogisticRegression")
    p_lr = run_model_inference(df, m_lr_retrieved, feats_lr)

    m_lgbm_retrieved = cache.get("BTC", model_type="LightGBM")
    feats_lgbm = cache.get_features("BTC", model_type="LightGBM")
    p_lgbm = run_model_inference(df, m_lgbm_retrieved, feats_lgbm)

    assert p_lr is not None and np.isclose(p_lr, 0.55)
    assert p_lgbm is not None and np.isclose(p_lgbm, 0.70)
