import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

X = pd.DataFrame(np.random.randn(10, 3), columns=['a', 'b', 'c'])
y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

X_train, y_train = X.iloc[:8], y.iloc[:8]
X_cal, y_cal = X.iloc[8:], y.iloc[8:]

base = LogisticRegression().fit(X_train, y_train)

cal = CalibratedClassifierCV(
    estimator=FrozenEstimator(base),
    method="sigmoid",
    cv=[([], np.arange(len(y_cal)))]
)
cal.fit(X_cal, y_cal)

try:
    preds = cal.predict_proba(X)
    print("Success, predictions:", preds.shape)
except Exception as e:
    print("Error during predict_proba:", e)
