import pytest

from polyflip.crypto.experiment_configs import (
    experiment_config_hash,
    legacy_threshold_backtest_options,
    normalize_experiment_config,
)


def test_config_is_canonical_and_hash_is_order_independent():
    left = normalize_experiment_config(
        {"feature_set": "b", "model": {"num_leaves": "17", "learning_rate": 0.03}}
    )
    right = normalize_experiment_config(
        {"model": {"learning_rate": 0.03, "num_leaves": 17}, "feature_set": "B"}
    )

    assert left == right
    assert left["feature_set_version"] == "B-direction-sequence-v1"
    assert experiment_config_hash(left) == experiment_config_hash(right)


def test_lgbm_aliases_canonicalization():
    config = normalize_experiment_config(
        {
            "model": {
                "bagging_fraction": 0.75,
                "feature_fraction": 0.65,
                "lambda_l2": 2.5,
                "lambda_l1": 0.5,
                "min_data_in_leaf": 45,
                "bagging_freq": 2,
                "eta": 0.02,
                "num_iterations": 220,
                "max_leaves": 42,
            }
        }
    )
    model = config["model"]
    assert model["subsample"] == 0.75
    assert model["colsample_bytree"] == 0.65
    assert model["reg_lambda"] == 2.5
    assert model["reg_alpha"] == 0.5
    assert model["min_child_samples"] == 45
    assert model["subsample_freq"] == 2
    assert model["learning_rate"] == 0.02
    assert model["n_estimators"] == 220
    assert model["num_leaves"] == 42


def test_min_split_gain_accepted():
    config = normalize_experiment_config(
        {"model": {"min_split_gain": 0.15}}
    )
    assert config["model"]["min_split_gain"] == 0.15


def test_unknown_model_parameters_soft_fail():
    config = normalize_experiment_config(
        {"model": {"unknown_optimizer_custom_param": 123, "learning_rate": 0.04}}
    )
    assert "unknown_optimizer_custom_param" not in config["model"]
    assert config["model"]["learning_rate"] == 0.04


def test_out_of_bounds_model_parameters_clamped():
    config = normalize_experiment_config(
        {"model": {"learning_rate": 5.0, "subsample": 0.01, "max_depth": 100}}
    )
    assert config["model"]["learning_rate"] == 1.0
    assert config["model"]["subsample"] == 0.1
    assert config["model"]["max_depth"] == 32


def test_config_rejects_unsafe_backtest_parameters():
    with pytest.raises(ValueError, match="must be between"):
        normalize_experiment_config({"backtest": {"min_edge": 2}})
    with pytest.raises(ValueError, match="min_price"):
        normalize_experiment_config({"backtest": {"min_price": 0.9, "max_price": 0.2}})


def test_config_rejects_unknown_calibration_method():
    with pytest.raises(ValueError, match="calibration.method"):
        normalize_experiment_config({"calibration": {"method": "MAGIC"}})


def test_weighted_backtest_config_is_canonical_and_threshold_safe():
    config = normalize_experiment_config({
        "backtest": {
            "policy_mode": "weighted_active",
            "market_weight": "0.80",
            "lgbm_weight": "0.20",
            "weighted_fee_rate": "0.07",
            "execution_role": "taker",
        }
    })

    assert config["backtest"]["policy_mode"] == "WEIGHTED_ACTIVE"
    assert config["backtest"]["execution_role"] == "TAKER"
    assert config["backtest"]["market_weight"] == pytest.approx(0.80)
    assert config["backtest"]["weighted_fee_exponent"] == pytest.approx(1.0)
    assert set(legacy_threshold_backtest_options(config["backtest"])) == {
        "min_edge",
        "cost_buffer",
        "fee_rate",
        "min_price",
        "max_price",
        "outsider_max_price",
        "stake_usdc",
        "slippage_pct",
    }
