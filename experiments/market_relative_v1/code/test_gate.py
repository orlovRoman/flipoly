"""T18 self-tests: gate order-invariance, bad-model FAIL, spec pin."""
import json

import numpy as np
import pandas as pd
import pytest

import gate as G


def _synth_eval(tmp_path, flip=False):
    rng = np.random.RandomState(11)
    n = 3000
    y = (rng.uniform(0, 1, n) < 0.5).astype(float)
    base = np.clip(0.5 + (y - 0.5) * 0.1 + rng.normal(0, 0.02, n), 0.01, 0.99)
    if flip:
        base = 1 - base
    folds = np.array(["F1", "F2", "F3", "F4", "F5", "F6"] * 500)[:n]
    oof = pd.DataFrame({"market_id": [str(i) for i in range(n)], "fold": folds,
                        "y": y, "baseline": 0.5, "linear": base,
                        "lgbm_A": base, "lgbm_B": base, "lgbm_C": base})
    d = tmp_path / "ev"
    d.mkdir()
    oof.to_csv(d / "oof.csv", index=False)
    per = n // 6
    fm, i = {}, 0
    for f in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        fm[f] = {"baseline": {"logloss": 0.6931},
                 "linear": {"logloss": 0.69 if not flip else 0.72},
                 "lgbm_A": {"logloss": 0.69 if not flip else 0.72},
                 "lgbm_B": {"logloss": 0.69 if not flip else 0.72},
                 "lgbm_C": {"logloss": 0.69 if not flip else 0.72}}
    one = {"n_signals": 1200, "raw_total": 50.0, "canon_total": 20.0,
           "roi": 20.0 / 1200, "maxdd_over_staked": -0.02,
           "ci95": [0.005, 0.03],
           "by_fold": {f: 20.0 / 6 for f in ["F1", "F2", "F3", "F4", "F5", "F6"]},
           "n_by_fold": {f: 200 for f in ["F1", "F2", "F3", "F4", "F5", "F6"]},
           "fold_pos_share": [1 / 6, "F1"],
           "asset_pos_share": [0.3, "BTC"]}
    econ = {m: dict(one, by_fold=dict(one["by_fold"]), n_by_fold=dict(one["n_by_fold"]),
                    fold_pos_share=list(one["fold_pos_share"]),
                    asset_pos_share=list(one["asset_pos_share"]))
            for m in ["linear", "lgbm_A", "lgbm_B", "lgbm_C"]}
    if flip:
        for m in econ:
            econ[m]["canon_total"] = -20.0
            econ[m]["roi"] = -20.0 / 1200
    json.dump({"folds": fm, "econ": econ}, open(d / "metrics.json", "w"))
    return str(d)


def test_cal_order():
    """cal_slope_intercept returns (intercept, slope); perfectly calibrated
    input must give slope~=1, intercept~=0. Guards against it/sl swaps."""
    import train as T
    from metrics import cal_slope_intercept
    rng = np.random.RandomState(5)
    x = rng.normal(0, 1, 4000)
    p = T.sigmoid(2.0 * x)
    y = (rng.uniform(0, 1, 4000) < p).astype(float)
    it, sl = cal_slope_intercept(y, p)
    assert abs(sl - 1.0) < 0.1
    assert abs(it) < 0.1


def test_order_invariance(tmp_path):
    import gate as G2
    d = _synth_eval(tmp_path)
    v1 = G.main(evaldir=d)["verdicts"]
    G.CANDIDATES = ["lgbm_C", "lgbm_B", "lgbm_A", "linear"]
    try:
        v2 = G2.main(evaldir=d)["verdicts"]
    finally:
        G.CANDIDATES = ["linear", "lgbm_A", "lgbm_B", "lgbm_C"]
    assert v1["linear"] == v2["linear"]


def test_bad_model_fails(tmp_path):
    d = _synth_eval(tmp_path, flip=True)
    out = G.main(evaldir=d)
    assert out["verdicts"]["linear"]["verdict"] == "FAIL"
    assert any("logloss" in f or "roi" in f or "canon" in f
               for f in out["verdicts"]["linear"]["failed"])
