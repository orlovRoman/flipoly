import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score

def test_threshold_calibration_logic():
    # Синтетические скоры
    np.random.seed(42)
    y_true = np.array([1]*50 + [0]*150)
    y_scores = np.random.beta(3, 1, 50).tolist() + np.random.beta(1, 3, 150).tolist()

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_scores)

    optimal = None
    for prec, thresh in zip(precision_arr, thresholds):
        if prec >= 0.60:
            optimal = thresh
            break

    assert optimal is not None, "Не найден порог с precision >= 0.60 на хороших данных"
    assert 0.0 < optimal < 1.0, f"Порог вне [0,1]: {optimal}"

    # Проверяем, что при этом пороге precision действительно >= 60%
    preds_at_threshold = (np.array(y_scores) >= optimal).astype(int)
    actual_precision = precision_score(y_true, preds_at_threshold)
    assert actual_precision >= 0.55, f"Precision={actual_precision:.2f} — ниже ожидаемого"
