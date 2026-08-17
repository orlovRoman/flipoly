"""retrospective isotonic calibration is forbidden when persisted OOF artifacts lack fold_id and calibration membership metadata"""

from typing import Any, Dict, Iterable, List, Optional
from sklearn.isotonic import IsotonicRegression


class IsotonicFeasibilityError(ValueError):
    """Base exception for isotonic feasibility errors."""


class MissingFoldMetadata(IsotonicFeasibilityError):
    """Raised when required fold metadata is missing."""


class InsufficientCalibrationData(IsotonicFeasibilityError):
    """Raised when calibration data is insufficient to fit isotonic regression."""


def assess_fold_metadata(
    frame_columns: Iterable[str],
    metadata: Optional[Dict[str, Any]] = None,
    min_calibration_rows: int = 2,
) -> Dict[str, Any]:
    missing = []
    cols = set(frame_columns) if frame_columns is not None else set()

    if "fold_id" not in cols:
        missing.append("fold_id")
    if "split" not in cols and "partition" not in cols:
        missing.append("split_or_partition")

    if metadata is None:
        missing.append("metadata")
    else:
        if "calibration" not in metadata:
            missing.append("calibration")
        if "validation" not in metadata and "OOT" not in metadata:
            missing.append("validation_or_OOT")
        if "calibration_rows_by_fold" not in metadata:
            missing.append("calibration_rows_by_fold")
        else:
            counts = metadata.get("calibration_rows_by_fold", {})
            if not isinstance(counts, dict) or not counts:
                missing.append("calibration_rows_by_fold_empty")
            else:
                for fold, count in counts.items():
                    if count < min_calibration_rows:
                        missing.append(f"calibration_rows_by_fold[{fold}] < {min_calibration_rows}")

    if not missing:
        return {
            "status": "AVAILABLE_FOR_RETROSPECTIVE_ISOTONIC",
            "missing_metadata": [],
        }
    return {
        "status": "UNAVAILABLE_WITHOUT_RETRAIN",
        "missing_metadata": missing,
    }


def fit_per_fold_isotonic(
    calibration_rows: List[Dict[str, Any]],
    future_rows: List[Dict[str, Any]],
    score_key: str = "score",
    outcome_key: str = "outcome",
    fold_key: str = "fold_id",
) -> List[Dict[str, Any]]:
    if not calibration_rows:
        raise InsufficientCalibrationData("Calibration rows cannot be empty.")

    calib_by_fold: Dict[Any, Dict[str, List[float]]] = {}
    for row in calibration_rows:
        if fold_key not in row:
            raise MissingFoldMetadata(f"Missing '{fold_key}' in calibration row.")
        f_id = row[fold_key]
        if score_key not in row or outcome_key not in row:
            raise InsufficientCalibrationData("Missing score or outcome key in calibration row.")
        if f_id not in calib_by_fold:
            calib_by_fold[f_id] = {"scores": [], "outcomes": []}
        calib_by_fold[f_id]["scores"].append(float(row[score_key]))
        calib_by_fold[f_id]["outcomes"].append(float(row[outcome_key]))

    models: Dict[Any, IsotonicRegression] = {}
    for f_id, data in calib_by_fold.items():
        if len(data["scores"]) < 2:
            raise InsufficientCalibrationData(f"Fold '{f_id}' has fewer than 2 calibration rows.")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(data["scores"], data["outcomes"])
        models[f_id] = iso

    predictions: List[Dict[str, Any]] = []
    for idx, row in enumerate(future_rows):
        if fold_key not in row:
            raise MissingFoldMetadata(f"Missing '{fold_key}' in future row.")
        f_id = row[fold_key]
        if f_id not in models:
            raise MissingFoldMetadata(f"No calibration model found for fold '{f_id}'.")
        if score_key not in row:
            raise InsufficientCalibrationData(f"Missing '{score_key}' in future row.")

        score_val = float(row[score_key])
        pred = float(models[f_id].predict([score_val])[0])
        predictions.append({
            "fold_id": f_id,
            "row_index": idx,
            "prediction": pred,
        })

    return predictions
