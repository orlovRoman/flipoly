import pytest
import numpy as np
from polyflip.settings_registry import registry_defaults
from polyflip.models.trainer import _compute_sample_weights


def test_registry_defaults_contain_lr_train_settings():
    defaults = registry_defaults()
    assert "LR_TRAIN_MAX_TIME_LEFT_MIN" in defaults
    assert float(defaults["LR_TRAIN_MAX_TIME_LEFT_MIN"]) == 15.0
    assert "LR_TRAIN_MIN_TIME_LEFT_MIN" in defaults
    assert float(defaults["LR_TRAIN_MIN_TIME_LEFT_MIN"]) == 0.5
    assert "LR_SAMPLE_WEIGHT_MODE" in defaults
    assert defaults["LR_SAMPLE_WEIGHT_MODE"] == "time_decay"
    assert "LR_SAMPLE_WEIGHT_TAU" in defaults
    assert float(defaults["LR_SAMPLE_WEIGHT_TAU"]) == 5.0


def test_compute_sample_weights_uniform():
    time_left = np.array([0.5, 5.0, 15.0])
    weights = _compute_sample_weights(time_left, mode="uniform")
    assert weights is None


def test_compute_sample_weights_time_decay():
    time_left = np.array([0.5, 5.0, 15.0])
    weights = _compute_sample_weights(time_left, mode="time_decay")
    assert weights is not None
    assert len(weights) == 3
    # Nearer snapshots (smaller time_left) must get higher weight
    assert weights[0] > weights[1] > weights[2]
    # Mean weight normalized to ~1.0
    assert abs(weights.mean() - 1.0) < 1e-4


def test_compute_sample_weights_exp_decay():
    time_left = np.array([0.5, 5.0, 15.0])
    weights = _compute_sample_weights(time_left, mode="exp_decay", tau=5.0)
    assert weights is not None
    assert len(weights) == 3
    assert weights[0] > weights[1] > weights[2]
    assert abs(weights.mean() - 1.0) < 1e-4


def test_compute_sample_weights_unknown_fallback():
    time_left = np.array([1.0, 2.0])
    weights = _compute_sample_weights(time_left, mode="invalid_mode")
    assert weights is None
