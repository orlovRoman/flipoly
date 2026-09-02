from polyflip.ai_lab.executor import StepContext
from polyflip.ai_lab import lgbm_adapters


def _context(**overrides):
    payload = {
        "run_id": 14,
        "step_id": 1,
        "action": "TRAIN_MODEL",
        "config_id": 12,
        "config_hash": "hash",
        "objective": "test resilience",
        "scope": {},
        "input_payload": {},
        "model_family": "LIGHTGBM",
        "feature_set": "C",
        "asset": "XRPUSDT",
        "regime": None,
        "interval": "15m",
        "model_params": {},
        "calibration_params": {},
        "strategy_params": {"strategy_branch": "COMBINED"},
        "backtest_params": {},
    }
    payload.update(overrides)
    return StepContext(**payload)


def test_normalized_config_handles_llm_aliases_and_min_split_gain():
    context = _context(
        model_params={
            "bagging_fraction": 0.8,
            "feature_fraction": 0.7,
            "lambda_l2": 2.0,
            "lambda_l1": 0.1,
            "min_data_in_leaf": 50,
            "bagging_freq": 1,
            "min_split_gain": 0.05,
            "learning_rate": 0.03,
            "num_leaves": 24,
            "max_depth": 5,
        }
    )

    config = lgbm_adapters._normalized_config(context)
    model = config["model"]

    assert model["subsample"] == 0.8
    assert model["colsample_bytree"] == 0.7
    assert model["reg_lambda"] == 2.0
    assert model["reg_alpha"] == 0.1
    assert model["min_child_samples"] == 50
    assert model["subsample_freq"] == 1
    assert model["min_split_gain"] == 0.05
    assert model["learning_rate"] == 0.03
    assert model["num_leaves"] == 24
    assert model["max_depth"] == 5


def test_normalized_config_soft_fails_unknown_params():
    context = _context(
        model_params={
            "unknown_llm_experimental_param": "foo",
            "bagging_fraction": 0.8,
            "learning_rate": 0.03,
        }
    )

    config = lgbm_adapters._normalized_config(context)
    assert "unknown_llm_experimental_param" not in config["model"]
    assert config["model"]["subsample"] == 0.8
    assert config["model"]["learning_rate"] == 0.03
