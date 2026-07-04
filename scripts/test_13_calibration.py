# scripts/test_13_calibration.py
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from lightgbm import LGBMClassifier

np.random.seed(42)
X = np.random.randn(500, 5)
y = (X[:, 0] + np.random.randn(500) * 0.5 > 0).astype(int)

lgbm = LGBMClassifier(n_estimators=20, verbosity=-1, random_state=42)
lgbm.fit(X[:300], y[:300])

# Тест: cv=None с FrozenEstimator должен работать без ошибок и калибровать
cal = CalibratedClassifierCV(estimator=FrozenEstimator(lgbm), method="sigmoid", cv=None)
cal.fit(X[300:400], y[300:400])   # только platt-scaling

probas = cal.predict_proba(X[400:])[:, 1]
assert probas.min() >= 0.0 and probas.max() <= 1.0, "Вероятности вне [0, 1]"

# Тест: вероятности откалиброваны (не просто 0/1)
assert probas.std() > 0.05, "Вероятности не дифференцированы — calibration не работает"
assert len(cal.calibrated_classifiers_) == 1, f"Ожидался 1 классификатор, получено {len(cal.calibrated_classifiers_)}"

print("✅ Тест 13: CalibratedClassifierCV FrozenEstimator OK")
