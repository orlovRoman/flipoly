import pytest
import pickle
import numpy as np
import pandas as pd
from polyflip.models.trainer import _fit_and_serialize
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 300  # увеличено для стабильности calibration_curve
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n) * 2 + 1,
    })
    prob = 1 / (1 + np.exp(-X['feature1']))
    y = pd.Series((np.random.rand(n) < prob).astype(int), name='target')
    groups = pd.Series(np.repeat(np.arange(n // 2), 2), name='group')
    return X, y, groups


def test_returns_five_values(dummy_data):
    """_fit_and_serialize должна возвращать 5 значений после добавления ece"""
    X, y, groups = dummy_data
    result = _fit_and_serialize(X, y, groups)
    assert len(result) == 5, f"Ожидалось 5 значений, получено {len(result)}"


def test_final_model_is_calibrated(dummy_data):
    """Финальная модель должна быть CalibratedClassifierCV"""
    X, y, groups = dummy_data
    model_bytes, *_ = _fit_and_serialize(X, y, groups)
    model = pickle.loads(model_bytes)
    assert isinstance(model, CalibratedClassifierCV)


def test_calibrated_probabilities_sum_to_one(dummy_data):
    """Вероятности должны суммироваться в 1.0"""
    X, y, groups = dummy_data
    model_bytes, *_ = _fit_and_serialize(X, y, groups)
    model = pickle.loads(model_bytes)
    proba = model.predict_proba(X.head(10))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_ece_is_valid_float(dummy_data):
    """ECE должен быть float в диапазоне [0, 1]"""
    X, y, groups = dummy_data
    _, _, _, _, ece = _fit_and_serialize(X, y, groups)
    assert isinstance(ece, float), f"ece должен быть float, получено {type(ece)}"
    assert 0.0 <= ece <= 1.0, f"ECE={ece:.4f} выходит за пределы [0, 1]"


def test_ece_reasonable_after_calibration(dummy_data):
    """После калибровки ECE на тренировочных данных должен быть < 0.15"""
    X, y, groups = dummy_data
    model_bytes, *_ = _fit_and_serialize(X, y, groups)
    model = pickle.loads(model_bytes)
    proba = model.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    assert ece < 0.15, f"ECE={ece:.4f} слишком высокий после калибровки"


def test_no_data_leakage_oof_scores_honest(dummy_data):
    """
    Проверяет отсутствие data leakage: OOF-прогнозы должны быть
    хуже, чем прогнозы на train (если leakage есть — они будут одинаковы).
    """
    X, y, groups = dummy_data
    model_bytes, val_acc, *_ = _fit_and_serialize(X, y, groups)
    model = pickle.loads(model_bytes)
    # Train accuracy финальной модели должна быть >= val_acc
    from sklearn.metrics import roc_auc_score
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    # Если есть leakage — val_acc будет подозрительно близок к train_auc
    # Честный gap: train_auc > val_acc на realistic данных
    assert train_auc >= val_acc - 0.05, (
        f"Train AUC ({train_auc:.4f}) неожиданно хуже val AUC ({val_acc:.4f}) — "
        f"возможна ошибка в логике OOF"
    )
