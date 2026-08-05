import pytest
import pickle
import numpy as np
import pandas as pd
from polyflip.models.trainer import _fit_and_serialize
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({'feature1': np.random.randn(n), 'feature2': np.random.randn(n) * 2 + 1})
    prob = 1 / (1 + np.exp(-X['feature1']))
    y = pd.Series((np.random.rand(n) < prob).astype(int), name='target')
    groups = pd.Series(np.repeat(np.arange(n // 2), 2), name='group')
    return (X, y, groups)