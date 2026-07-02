import pytest
import pickle
import numpy as np
import pandas as pd
from polyflip.models.trainer import _fit_and_serialize
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n) * 2 + 1,
    })
    # y depends on feature1
    prob = 1 / (1 + np.exp(-X['feature1']))
    y = pd.Series((np.random.rand(n) < prob).astype(int), name='target')
    groups = pd.Series(np.repeat(np.arange(n//2), 2), name='group')
    return X, y, groups

def test_calibration_ece_logged(dummy_data, capsys):
    X, y, groups = dummy_data
    _fit_and_serialize(X, y, groups)
    captured = capsys.readouterr()
    assert "calibration_check" in captured.out or "calibration_check" in captured.err

def test_final_model_is_calibrated(dummy_data):
    X, y, groups = dummy_data
    model_bytes, val_acc, baseline_acc, threshold, ece = _fit_and_serialize(X, y, groups)
    model = pickle.loads(model_bytes)
    assert isinstance(model, CalibratedClassifierCV), "Финальная модель должна быть CalibratedClassifierCV"

def test_calibrated_probabilities_sum_to_one(dummy_data):
    X, y, groups = dummy_data
    model_bytes, *_ = _fit_and_serialize(X, y, groups)
    model = pickle.loads(model_bytes)
    proba = model.predict_proba(X.head(5))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

def test_ece_after_calibration_below_threshold(dummy_data):
    X, y, groups = dummy_data
    model_bytes, *_ = _fit_and_serialize(X, y, groups)
    model = pickle.loads(model_bytes)
    
    oof_proba = model.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, oof_proba, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    
    assert ece < 0.15, f"ECE после калибровки: {ece:.4f} — слишком высокий"
