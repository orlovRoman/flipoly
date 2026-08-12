import pytest

from polyflip.crypto.experiment_configs import (
    experiment_config_hash,
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
