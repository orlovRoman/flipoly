"""train.py v1.0.0 — market_relative_v1 models (spec v1.0.4).

Formulation: logit(p_final) = logit(p_market) + delta(x).
  linear:  delta(x) = b + w.x  via Newton/IRLS, L2 (intercept unpenalized)
  lgbm:    delta(x) = sum of trees, trained with init_score = logit(p_market)
Baseline MARKET_ONLY: delta = 0 (p_final == p_market bit-exact).
Deterministic: fixed init, fixed seeds, lgbm deterministic/num_threads=4.
"""
import json

import numpy as np

MODEL_CODE_VERSION = "v1.0.0"
SEED = 20260912
LGBM_THREADS = 4

LINEAR = {"loss": "binary_logloss_with_fixed_market_offset", "fit_intercept": True,
          "penalty": "l2", "lambda_l2": 10.0, "standardize_continuous": True,
          "max_iterations": 2000, "tolerance": 1.0e-8, "seed": SEED}

LGBM_COMMON = {"objective": "binary", "metric": "binary_logloss",
               "boosting_type": "gbdt", "learning_rate": 0.03,
               "n_estimators": 2000, "early_stopping_rounds": 100,
               "max_bin": 255, "deterministic": True, "force_col_wise": True,
               "feature_pre_filter": False, "seed": SEED,
               "feature_fraction_seed": SEED, "bagging_seed": SEED,
               "data_random_seed": SEED, "verbosity": -1,
               "num_threads": LGBM_THREADS}
LGBM_CONFIGS = {
    "A": {"max_depth": 3, "num_leaves": 7, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l1": 0.0, "lambda_l2": 10.0, "min_gain_to_split": 0.01},
    "B": {"max_depth": 4, "num_leaves": 15, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l1": 1.0, "lambda_l2": 10.0, "min_gain_to_split": 0.01},
    "C": {"max_depth": 5, "num_leaves": 15, "min_data_in_leaf": 500,
          "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
          "lambda_l1": 2.0, "lambda_l2": 20.0, "min_gain_to_split": 0.02},
}
ES_FRACTION = 0.2
ES_MIN_ROWS = 200


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def baseline_predict(p_market):
    return np.asarray(p_market, dtype=np.float64).copy()


class Standardizer:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


def irls_offset(Xs, y, offset, lam=10.0, max_iter=2000, tol=1e-8):
    """Newton/IRLS for logit(p) = offset + b + Xs.w, L2 on w (b unpenalized)."""
    n, k = Xs.shape
    y = np.asarray(y, dtype=np.float64)
    beta = np.zeros(k + 1)
    D = np.ones(n)
    for _ in range(max_iter):
        eta = offset + beta[0] + Xs.dot(beta[1:])
        mu = sigmoid(eta)
        w = np.maximum(mu * (1.0 - mu), 1e-12)
        z = eta + (y - mu) / w
        A = np.column_stack([D, Xs])
        Aw = A * w[:, None]
        H = Aw.T.dot(A)
        H[1:, 1:] += lam * np.eye(k)
        g = Aw.T.dot(z - offset)
        g[1:] -= lam * beta[1:]
        step = np.linalg.solve(H, g - H.dot(beta))
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            break
    return beta


def train_linear(Xtr, ytr, ptr, feature_names):
    sc = Standardizer().fit(Xtr)
    Xs = sc.transform(Xtr)
    beta = irls_offset(Xs, ytr, logit(ptr), lam=LINEAR["lambda_l2"],
                       max_iter=LINEAR["max_iterations"], tol=LINEAR["tolerance"])
    return {"kind": "linear", "intercept": float(beta[0]),
            "coef": [float(v) for v in beta[1:]],
            "mean": [float(v) for v in sc.mean_],
            "scale": [float(v) for v in sc.scale_],
            "features": list(feature_names), "params": LINEAR}


def linear_predict(model, X, p_market):
    Xs = (np.asarray(X, dtype=np.float64) - np.array(model["mean"])) / np.array(model["scale"])
    return sigmoid(logit(p_market) + model["intercept"] + Xs.dot(np.array(model["coef"])))


def es_split(n):
    """Early-stopping holdout: most recent max(20%, 200 rows) of time-ordered train."""
    n_es = max(int(n * ES_FRACTION), ES_MIN_ROWS)
    n_es = min(n_es, n - 1)
    return n - n_es


def train_lgbm(Xtr, ytr, ptr, cfg_name, feature_names):
    import lightgbm as lgb
    params = dict(LGBM_COMMON)
    params.update(LGBM_CONFIGS[cfg_name])
    n = len(ytr)
    cut = es_split(n)
    dtr = lgb.Dataset(Xtr[:cut], label=ytr[:cut], init_score=logit(ptr[:cut]),
                      feature_name=list(feature_names))
    des = lgb.Dataset(Xtr[cut:], label=ytr[cut:], init_score=logit(ptr[cut:]),
                      feature_name=list(feature_names), reference=dtr)
    booster = lgb.train(params, dtr, valid_sets=[des])
    return {"kind": "lgbm", "config": cfg_name, "params": params,
            "features": list(feature_names),
            "model_string": booster.model_to_string(),
            "best_iteration": int(booster.best_iteration)}


def lgbm_predict(model, X, p_market):
    import lightgbm as lgb
    booster = lgb.Booster(model_str=model["model_string"])
    raw = booster.predict(np.asarray(X, dtype=np.float64),
                          num_iteration=model["best_iteration"])
    return sigmoid(logit(p_market) + raw), raw


def config_registry():
    return {"linear": LINEAR, "lgbm": {"common": LGBM_COMMON, "configs": LGBM_CONFIGS}}
