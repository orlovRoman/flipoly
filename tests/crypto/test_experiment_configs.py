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


def test_config_rejects_unknown_or_unsafe_parameters():
    with pytest.raises(ValueError, match="Unknown experiment parameter"):
        normalize_experiment_config({"model": {"unknown": 1}})
    with pytest.raises(ValueError, match="must be between"):
        normalize_experiment_config({"backtest": {"min_edge": 2}})
    with pytest.raises(ValueError, match="min_price"):
        normalize_experiment_config({"backtest": {"min_price": 0.9, "max_price": 0.2}})


def test_config_rejects_unknown_calibration_method():
    with pytest.raises(ValueError, match="calibration.method"):
        normalize_experiment_config({"calibration": {"method": "MAGIC"}})


def test_config_rejects_boolean_as_number():
    with pytest.raises(ValueError, match="must be a number"):
        normalize_experiment_config({"model": {"n_estimators": True}})


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
