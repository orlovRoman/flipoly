"""Probability metrics and B0 calibration (train-only fitting)."""

import math

import pytest

from polyflip.research.oracle_basis.models import (
    apply_b0_calibration,
    brier_score,
    compare_incremental,
    fit_b0_calibration,
    hit_rate,
    log_loss,
)


def test_brier_logloss_hit_known_values():
    probs = [0.9, 0.1]
    labels = [1, 0]
    assert brier_score(probs, labels) == pytest.approx(0.01)
    assert log_loss(probs, labels) == pytest.approx(-math.log(0.9))
    assert hit_rate(probs, labels) == 1.0
    assert hit_rate(probs, labels, threshold=0.95) == 0.5
    with pytest.raises(ValueError):
        brier_score([0.5], [1, 0])
    with pytest.raises(ValueError):
        log_loss([1.5], [1])


def test_b0_calibration_fits_train_only_and_stays_ordered():
    model = fit_b0_calibration([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    calibrated = apply_b0_calibration(model, [0.05, 0.5, 0.95])
    assert all(0.0 <= p <= 1.0 for p in calibrated)
    assert calibrated[0] < calibrated[1] < calibrated[2]
    with pytest.raises(ValueError):
        fit_b0_calibration([0.1, 0.2], [0, 0])


def test_compare_incremental_reports_signed_gain():
    diff = compare_incremental(
        {"brier": 0.20, "log_loss": 0.60}, {"brier": 0.19, "log_loss": 0.61}
    )
    assert diff == {"brier": pytest.approx(-0.01), "log_loss": pytest.approx(0.01)}
