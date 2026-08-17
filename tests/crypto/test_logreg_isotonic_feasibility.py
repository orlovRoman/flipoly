import pytest
from polyflip.crypto.logreg_isotonic_feasibility import (
    assess_fold_metadata,
    fit_per_fold_isotonic,
    MissingFoldMetadata,
    InsufficientCalibrationData,
)


def test_assess_fold_metadata_missing_fold():
    res = assess_fold_metadata(["score", "outcome"])
    assert res["status"] == "UNAVAILABLE_WITHOUT_RETRAIN"


def test_assess_fold_metadata_available():
    frame_columns = ["score", "outcome", "fold_id", "split"]
    metadata = {
        "calibration": True,
        "validation": True,
        "calibration_rows_by_fold": {0: 2, 1: 2},
    }
    res = assess_fold_metadata(frame_columns, metadata=metadata, min_calibration_rows=2)
    assert res["status"] == "AVAILABLE_FOR_RETROSPECTIVE_ISOTONIC"


def test_two_folds_produce_predictions():
    calibration_rows = [
        {"fold_id": 0, "score": 0.1, "outcome": 0},
        {"fold_id": 0, "score": 0.8, "outcome": 1},
        {"fold_id": 1, "score": 0.2, "outcome": 0},
        {"fold_id": 1, "score": 0.9, "outcome": 1},
    ]
    future_rows = [
        {"fold_id": 0, "score": 0.5, "outcome": 0},
        {"fold_id": 1, "score": 0.7, "outcome": 1},
    ]
    preds = fit_per_fold_isotonic(calibration_rows, future_rows)
    assert len(preds) == 2
    for p in preds:
        assert isinstance(p, dict)
        assert set(p) >= {"fold_id", "row_index", "prediction"}
        assert isinstance(p["prediction"], (float, int))
        assert 0.0 <= p["prediction"] <= 1.0


def test_changing_future_outcomes_does_not_change_predictions():
    calibration_rows = [
        {"fold_id": 0, "score": 0.2, "outcome": 0},
        {"fold_id": 0, "score": 0.7, "outcome": 1},
        {"fold_id": 1, "score": 0.3, "outcome": 0},
        {"fold_id": 1, "score": 0.8, "outcome": 1},
    ]
    future_rows_1 = [
        {"fold_id": 0, "score": 0.5, "outcome": 0},
        {"fold_id": 1, "score": 0.6, "outcome": 0},
    ]
    future_rows_2 = [
        {"fold_id": 0, "score": 0.5, "outcome": 1},
        {"fold_id": 1, "score": 0.6, "outcome": 1},
    ]
    preds_1 = fit_per_fold_isotonic(calibration_rows, future_rows_1)
    preds_2 = fit_per_fold_isotonic(calibration_rows, future_rows_2)
    assert preds_1 == preds_2


def test_independent_folds():
    calibration_rows_a = [
        {"fold_id": 0, "score": 0.1, "outcome": 0},
        {"fold_id": 0, "score": 0.9, "outcome": 1},
        {"fold_id": 1, "score": 0.2, "outcome": 0},
        {"fold_id": 1, "score": 0.8, "outcome": 1},
    ]
    calibration_rows_b = [
        {"fold_id": 0, "score": 0.4, "outcome": 1},
        {"fold_id": 0, "score": 0.6, "outcome": 0},
        {"fold_id": 1, "score": 0.2, "outcome": 0},
        {"fold_id": 1, "score": 0.8, "outcome": 1},
    ]
    future_rows = [
        {"fold_id": 1, "score": 0.5, "outcome": 0},
    ]
    preds_a = fit_per_fold_isotonic(calibration_rows_a, future_rows)
    preds_b = fit_per_fold_isotonic(calibration_rows_b, future_rows)
    assert preds_a == preds_b


def test_missing_fold_id_raises_missing_fold_metadata():
    calibration_rows = [
        {"score": 0.1, "outcome": 0},
        {"score": 0.9, "outcome": 1},
    ]
    future_rows = [
        {"fold_id": 0, "score": 0.5, "outcome": 0},
    ]
    with pytest.raises(MissingFoldMetadata):
        fit_per_fold_isotonic(calibration_rows, future_rows)


def test_insufficient_calibration_data_raises():
    calibration_rows = [
        {"fold_id": 0, "score": 0.1, "outcome": 0},
        {"fold_id": 1, "score": 0.2, "outcome": 0},
        {"fold_id": 1, "score": 0.8, "outcome": 1},
    ]
    future_rows = [
        {"fold_id": 0, "score": 0.5, "outcome": 0},
        {"fold_id": 1, "score": 0.5, "outcome": 0},
    ]
    with pytest.raises(InsufficientCalibrationData):
        fit_per_fold_isotonic(calibration_rows, future_rows)

