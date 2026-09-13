"""T12-T15 self-tests: baseline identity, linear/IRLS, lgbm offset, registry."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import train as T

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MKTREL_ROOT", os.path.dirname(BASE))
BUILD = os.environ.get("MKTREL_BUILD", os.path.join(ROOT, "build"))
DS = os.path.join(BUILD, "dataset.csv")


@pytest.fixture(scope="module")
def sample():
    if not os.path.exists(DS):
        pytest.skip("frozen dataset is not present; run the read-only export/build step")
    df = pd.read_csv(DS, dtype={"market_id": str})
    use = [c for c in df.columns if c in T_sentinel_features()]
    sub = df.iloc[:1500].reset_index(drop=True)
    X = sub[use].fillna(0.0).to_numpy(dtype=np.float64)
    y = sub["label"].to_numpy(dtype=np.float64)
    p = sub["p_market_yes"].to_numpy(dtype=np.float64)
    return X, y, p, use


def T_sentinel_features():
    import build_dataset as B
    return B.FEATURES


def test_baseline_exact(sample):
    _, _, p, _ = sample
    assert np.array_equal(T.baseline_predict(p), p)


def test_registry_counts():
    reg = T.config_registry()
    assert set(reg) == {"linear", "lgbm"}
    assert len(reg["lgbm"]["configs"]) == 3
    assert set(reg["lgbm"]["configs"]) == {"A", "B", "C"}
    assert reg["lgbm"]["configs"]["A"]["max_depth"] == 3
    assert reg["lgbm"]["configs"]["A"]["num_leaves"] == 7
    assert reg["lgbm"]["common"]["learning_rate"] == 0.03
    assert reg["lgbm"]["common"]["n_estimators"] == 2000
    assert reg["linear"]["lambda_l2"] == 10.0


def test_linear_trains_and_reproducible(sample):
    X, y, p, use = sample
    m1 = T.train_linear(X, y, p, use)
    m2 = T.train_linear(X, y, p, use)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert len(m1["coef"]) == X.shape[1]
    assert m1["params"]["lambda_l2"] == 10.0


def test_linear_zero_coef_is_market_only(sample):
    X, y, p, use = sample
    m = T.train_linear(X, y, p, use)
    m["intercept"] = 0.0
    m["coef"] = [0.0] * len(m["coef"])
    assert np.allclose(T.linear_predict(m, X, p), p, atol=1e-12)


def test_linear_sign_and_bounds(sample):
    X, y, p, use = sample
    m = T.train_linear(X, y, p, use)
    pr = T.linear_predict(m, X, p)
    assert bool(((pr > 0) & (pr < 1)).all())
    d = np.array(m["coef"])
    assert bool((np.abs(d) < 1e6).all())


def test_lgbm_trains_offset_and_reproducible(sample):
    X, y, p, use = sample
    m1 = T.train_lgbm(X, y, p, "A", use)
    m2 = T.train_lgbm(X, y, p, "A", use)
    pr1, _ = T.lgbm_predict(m1, X, p)
    pr2, _ = T.lgbm_predict(m2, X, p)
    assert np.array_equal(pr1, pr2)
    assert bool(((pr1 > 0) & (pr1 < 1)).all())
    assert m1["best_iteration"] > 0


def test_lgbm_inference_applies_offset(sample):
    import lightgbm as lgb
    X, y, p, use = sample
    m = T.train_lgbm(X, y, p, "A", use)
    pr, raw = T.lgbm_predict(m, X, p)
    assert np.allclose(pr, T.sigmoid(T.logit(p) + raw), atol=1e-12)


def test_lgbm_serialize_reload_identical(sample):
    X, y, p, use = sample
    m = T.train_lgbm(X, y, p, "B", use)
    blob = json.dumps({"s": m["model_string"], "it": m["best_iteration"]})
    m2 = {"model_string": json.loads(blob)["s"], "best_iteration": json.loads(blob)["it"]}
    pr1, _ = T.lgbm_predict(m, X, p)
    pr2, _ = T.lgbm_predict(m2, X, p)
    assert np.array_equal(pr1, pr2)
