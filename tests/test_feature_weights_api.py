import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from polyflip.api.analytics import extract_coefficients_from_blob

def test_extract_coefficients_from_blob_pipeline():
    features = ["time_left_min", "mid_price", "spread"]
    X = np.array([[10, 0.5, 0.01], [5, 0.6, 0.02], [2, 0.4, 0.01]])
    y = np.array([1, 0, 1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    pipe.fit(X, y)
    blob = pickle.dumps(pipe)

    coefs = extract_coefficients_from_blob(blob, ",".join(features))
    assert len(coefs) == 3
    assert "time_left_min" in coefs
    assert "mid_price" in coefs
    assert "spread" in coefs

def test_extract_coefficients_from_blob_calibrated():
    features = ["f1", "f2"]
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    pipe.fit(X, y)

    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(pipe), cv=[([], np.arange(len(y)))])
    calibrated.fit(X, y)
    blob = pickle.dumps(calibrated)

    coefs = extract_coefficients_from_blob(blob, ",".join(features))
    assert len(coefs) == 2
    assert "f1" in coefs
    assert "f2" in coefs
