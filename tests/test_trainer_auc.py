import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def test_trainer_auc_metric():
    X_raw, y_raw = make_classification(n_samples=200, n_features=4, random_state=42)
    X_df = pd.DataFrame(X_raw, columns=["f1", "f2", "f3", "f4"])
    y_s = pd.Series(y_raw)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, val in skf.split(X_df, y_s):
        pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", random_state=42))])
        pipe.fit(X_df.iloc[tr], y_s.iloc[tr])
        p = pipe.predict_proba(X_df.iloc[val])[:, 1]
        aucs.append(roc_auc_score(y_s.iloc[val], p))

    mean_auc = np.mean(aucs)
    assert mean_auc > 0.5, f"AUC={mean_auc:.3f} ниже случайного — что-то сломано"
    assert all(0.0 <= a <= 1.0 for a in aucs), "AUC вышел за границы [0,1]"
