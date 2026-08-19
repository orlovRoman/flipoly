"""Behavioral tests for Trainer Quality Gate validation."""
import pickle

import numpy as np
import pytest

from polyflip.crypto.trainer import CRYPTO_FEATURES, _evaluate_quality_gate


class _SmokeModel:
    n_features_in_ = len(CRYPTO_FEATURES)

    def __init__(self, probabilities=(0.4, 0.6)):
        self.probabilities = probabilities

    def predict_proba(self, rows):
        return np.asarray([self.probabilities] * len(rows), dtype=float)


def _evaluate(**overrides):
    values = {
        "model_bytes": pickle.dumps(_SmokeModel()),
        "val_auc": 0.62,
        "baseline_auc": 0.55,
        "ece": 0.08,
        "threshold": 0.55,
        "threshold_down": 0.53,
        "active_accuracy": 0.61,
        "active_version": 4,
    }
    values.update(overrides)
    return _evaluate_quality_gate(**values)


def test_valid_model_passes_quality_gate():
    passed, reasons, threshold_up, threshold_down = _evaluate()

    assert passed is True
    assert reasons == []
    assert threshold_up == pytest.approx(0.55)
    assert threshold_down == pytest.approx(0.53)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"val_auc": 0.50, "baseline_auc": 0.52}, "Negative lift vs baseline"),
        ({"ece": 0.151}, "Excessive ECE calibration error"),
        ({"val_auc": 0.58, "active_accuracy": 0.61}, "Accuracy degraded vs active model v4"),
        ({"val_auc": float("nan")}, "Non-finite quality metrics"),
    ],
)
def test_metric_failures_are_audited(overrides, reason):
    passed, reasons, _, _ = _evaluate(**overrides)

    assert passed is True
    assert any(reason in item for item in reasons)


def test_both_decision_thresholds_are_validated_and_sanitized():
    passed, reasons, threshold_up, threshold_down = _evaluate(
        threshold=-0.2,
        threshold_down=1.2,
    )

    assert passed is False
    assert threshold_up == pytest.approx(0.0)
    assert threshold_down == pytest.approx(1.0)
    assert any("UP threshold" in item for item in reasons)
    assert any("DOWN threshold" in item for item in reasons)


@pytest.mark.parametrize(
    "probabilities",
    [
        (float("nan"), 0.5),
        (-0.1, 1.1),
        (0.2, 0.2),
    ],
)
def test_smoke_test_rejects_invalid_probabilities(probabilities):
    passed, reasons, _, _ = _evaluate(
        model_bytes=pickle.dumps(_SmokeModel(probabilities)),
    )

    assert passed is False
    assert any("invalid predict_proba result" in item for item in reasons)
