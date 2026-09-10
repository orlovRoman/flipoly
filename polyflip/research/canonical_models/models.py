"""Steps 19-25: six forecast variants, all predicting P(UP).

- market: P(UP)=observed UP mid (no fitted calibration). P(DOWN)=1-P(UP);
  real DOWN mid kept only as a book-consistency diagnostic.
- M0: Phi(z) fixed formula.
- M1: base LogReg on [z, time_left, sigma], scaler fit on train only.
- M2: extended LogReg on M1+trajectory, fixed C.
- M3: LightGBM on same info as M2, capped complexity, early stopping on
  INNER validation (never final test).
- M4: market-offset correction logit(p)=logit(pm)+b+w.x, market coef fixed
  at 1, correction L2-shrunk to 0. Zero correction reproduces market exactly.

All models predict P(UP); P(DOWN)=1-P(UP). Side choice lives in policies.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass

import numpy as np

from .dataset import M1_COLUMNS, M2_COLUMNS

EPS = 1e-6


def _clip(p):
    return float(min(max(p, EPS), 1.0 - EPS))


def logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ---- market control (step 19) ----
def market_proba(up_mid: float | None) -> float | None:
    if up_mid is None or not (0.0 <= up_mid <= 1.0):
        return None
    return float(up_mid)


# ---- M0 (step 20) ----
def m0_phi(z: float | None) -> float | None:
    if z is None or not math.isfinite(z):
        return None
    return _clip(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


# ---- sklearn helpers ----
def _rows_to_X(rows: list[dict], columns: tuple[str, ...]) -> np.ndarray:
    X = []
    for r in rows:
        v = []
        for c in columns:
            x = r.get(c)
            if x is None:
                # secs_since_cross None -> large (no recent crossing)
                x = 1e6 if c == "secs_since_cross" else 0.0
            v.append(float(x))
        X.append(v)
    return np.asarray(X, dtype=float)


def _targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([int(r["target_up"]) for r in rows], dtype=int)


@dataclass
class SkScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def fit_scaler(X: np.ndarray) -> SkScaler:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return SkScaler(mean, scale)


class LogRegModel:
    """M1/M2 wrapper with pinned feature list + scaler (train-only)."""

    def __init__(self, columns, C: float = 1.0):
        self.columns = tuple(columns)
        self.C = float(C)
        self.scaler: SkScaler | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, rows: list[dict]):
        from sklearn.linear_model import LogisticRegression
        X = _rows_to_X(rows, self.columns)
        y = _targets(rows)
        self.scaler = fit_scaler(X)
        clf = LogisticRegression(C=self.C, max_iter=2000)
        clf.fit(self.scaler.transform(X), y)
        self.coef_ = np.asarray(clf.coef_[0], dtype=float)
        self.intercept_ = float(clf.intercept_[0])
        return self

    def predict_proba_up(self, rows: list[dict]) -> list[float]:
        assert self.scaler is not None and self.coef_ is not None
        X = self.scaler.transform(_rows_to_X(rows, self.columns))
        z = X @ self.coef_ + self.intercept_
        return [_clip(float(1.0 / (1.0 + math.exp(-v)))) for v in z]

    def dumps(self) -> bytes:
        return pickle.dumps({
            "columns": self.columns, "C": self.C,
            "mean": self.scaler.mean, "scale": self.scaler.scale,
            "coef": self.coef_, "b": self.intercept_,
        })

    @staticmethod
    def loads(blob: bytes) -> "LogRegModel":
        d = pickle.loads(blob)
        m = LogRegModel(tuple(d["columns"]), float(d["C"]))
        m.scaler = SkScaler(np.asarray(d["mean"]), np.asarray(d["scale"]))
        m.coef_ = np.asarray(d["coef"])
        m.intercept_ = float(d["b"])
        return m


class LgbmModel:
    """M3: same columns as M2, capped complexity."""

    def __init__(self, columns=M2_COLUMNS, seed: int = 42):
        self.columns = tuple(columns)
        self.seed = seed
        self.booster = None

    def fit(self, rows: list[dict], valid_rows: list[dict] | None = None):
        import lightgbm as lgb
        X = _rows_to_X(rows, self.columns)
        y = _targets(rows)
        dtrain = lgb.Dataset(X, label=y)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 15, "max_depth": 3, "min_data_in_leaf": 50,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 10.0, "verbosity": -1, "seed": self.seed,
            "num_threads": 4,
        }
        valid = None
        if valid_rows:
            valid = [lgb.Dataset(_rows_to_X(valid_rows, self.columns),
                                 label=_targets(valid_rows), reference=dtrain)]
            self.booster = lgb.train(params, dtrain, num_boost_round=300,
                                     valid_sets=valid,
                                     callbacks=[lgb.early_stopping(30, verbose=False)])
        else:
            self.booster = lgb.train(params, dtrain, num_boost_round=100)
        return self

    def predict_proba_up(self, rows: list[dict]) -> list[float]:
        assert self.booster is not None
        X = _rows_to_X(rows, self.columns)
        return [_clip(float(v)) for v in self.booster.predict(X)]


class MarketOffsetModel:
    """M4: logit(p) = logit(p_market) + b + w.x, L2-shrunk (w,b)->0.

    Fit by L2-penalized logistic regression on the residual scale:
    minimize NLL(sigmoid(offset + b + w.xs)) + 0.5*l2*(||w||^2+b^2).
    Deterministic full-batch gradient descent; zero init => with l2=inf
    predictions equal market exactly (self-check step 24).
    """

    def __init__(self, columns=M2_COLUMNS, l2: float = 1.0, lr: float = 0.5,
                 iters: int = 2000):
        self.columns = tuple(columns)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.iters = int(iters)
        self.scaler: SkScaler | None = None
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, rows: list[dict]):
        X = _rows_to_X(rows, self.columns)
        y = _targets(rows).astype(float)
        off = np.asarray([logit(float(r["p_up_market"])) for r in rows])
        self.scaler = fit_scaler(X)
        Xs = self.scaler.transform(X)
        w = np.zeros(Xs.shape[1])
        b = 0.0
        n = len(y)
        for _ in range(self.iters):
            z = off + b + Xs @ w
            p = 1.0 / (1.0 + np.exp(-z))
            err = (p - y) / n
            gw = Xs.T @ err + self.l2 * w / max(n, 1)
            gb = float(err.sum()) + self.l2 * b / max(n, 1)
            w -= self.lr * gw
            b -= self.lr * gb
        self.w, self.b = w, b
        return self

    def predict_proba_up(self, rows: list[dict]) -> list[float]:
        assert self.scaler is not None and self.w is not None
        X = self.scaler.transform(_rows_to_X(rows, self.columns))
        off = np.asarray([logit(float(r["p_up_market"])) for r in rows])
        z = off + self.b + X @ self.w
        return [_clip(float(1.0 / (1.0 + math.exp(-v)))) for v in z]

    def zeroed(self) -> "MarketOffsetModel":
        m = MarketOffsetModel(self.columns, self.l2, self.lr, 0)
        m.scaler = self.scaler
        m.w = np.zeros(len(self.columns))
        m.b = 0.0
        return m
