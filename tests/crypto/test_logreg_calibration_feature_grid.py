from collections import Counter
import csv
import json
from pathlib import Path
import pytest


REPORT_JSON_PATH = Path("reports/logreg_calibration_feature_grid_20260817.json")
REPORT_CSV_PATH = Path("reports/logreg_calibration_feature_grid_20260817.csv")

EXPECTED_PLATT_BASE_IDS = [
    821, 824, 827, 830, 833, 836, 839, 842, 845, 848,
    851, 854, 857, 860, 863, 866, 869, 872, 875, 878,
]


def _resolve_report_path(rel_path: Path) -> Path:
    if rel_path.exists():
        return rel_path
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / rel_path
    if candidate.exists():
        return candidate
    return rel_path


def _get_model_id(row: dict) -> int:
    val = row.get("model_registry_id", row.get("model_id"))
    return int(val)


def _get_calibration(row: dict) -> str:
    return str(row.get("calibration", row.get("calibration_method", ""))).strip().upper()


def _get_feature_set(row: dict) -> str:
    return str(row.get("feature_set", row.get("feature_variant", row.get("features", "")))).strip().upper()


def _get_status(row: dict) -> str:
    return str(row.get("status", "")).strip().upper()


def _get_coverage(row: dict) -> float:
    return float(row.get("coverage", 0.0))


@pytest.fixture(scope="module")
def json_rows() -> list[dict]:
    path = _resolve_report_path(REPORT_JSON_PATH)
    assert path.exists(), f"JSON report file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("rows", data.get("grid", []))
    raise AssertionError(f"Unexpected JSON structure in {path}")


@pytest.fixture(scope="module")
def csv_data() -> dict:
    path = _resolve_report_path(REPORT_CSV_PATH)
    assert path.exists(), f"CSV report file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return {"fieldnames": fieldnames, "rows": rows}


def test_json_and_csv_row_counts_and_csv_columns(json_rows: list[dict], csv_data: dict):
    assert len(json_rows) == 720, f"Expected 720 JSON rows, got {len(json_rows)}"
    assert len(csv_data["rows"]) == 720, f"Expected 720 CSV rows, got {len(csv_data['rows'])}"

    fieldnames = set(csv_data["fieldnames"])
    required_columns = {"model_registry_id", "status"}
    assert required_columns.issubset(fieldnames), f"CSV missing required columns: {required_columns - fieldnames}"


def test_model_registry_id_distribution(json_rows: list[dict]):
    model_ids = [_get_model_id(r) for r in json_rows]
    counts = Counter(model_ids)
    expected_ids = set(range(820, 880))

    assert set(counts.keys()) == expected_ids, "model_registry_id does not cover exactly 820..879"
    for m_id in expected_ids:
        assert counts[m_id] == 12, f"model_registry_id {m_id} has {counts[m_id]} rows, expected 12"


def test_status_counts(json_rows: list[dict]):
    status_counts = Counter(_get_status(r) for r in json_rows)
    assert status_counts["REPLAY_READY"] == 80, f"Expected 80 REPLAY_READY, got {status_counts['REPLAY_READY']}"
    assert status_counts["RETRAIN_REQUIRED"] == 640, f"Expected 640 RETRAIN_REQUIRED, got {status_counts['RETRAIN_REQUIRED']}"


def test_base_raw_rows_replay_ready(json_rows: list[dict]):
    base_raw_rows = [
        r for r in json_rows
        if _get_feature_set(r) == "BASE" and _get_calibration(r) == "RAW"
    ]
    assert len(base_raw_rows) == 60, f"Expected 60 BASE/RAW rows, got {len(base_raw_rows)}"
    for r in base_raw_rows:
        assert _get_status(r) == "REPLAY_READY", f"BASE/RAW row for model {_get_model_id(r)} is not REPLAY_READY"


def test_platt_base_replay_ready_ids(json_rows: list[dict]):
    platt_base_ready_ids = sorted([
        _get_model_id(r)
        for r in json_rows
        if _get_feature_set(r) == "BASE"
        and _get_calibration(r) == "PLATT"
        and _get_status(r) == "REPLAY_READY"
    ])
    assert platt_base_ready_ids == EXPECTED_PLATT_BASE_IDS, (
        f"PLATT/BASE replay-ready IDs mismatch: {platt_base_ready_ids} != {EXPECTED_PLATT_BASE_IDS}"
    )


def test_no_isotonic_and_no_non_base_is_replay_ready(json_rows: list[dict]):
    isotonic_ready = [
        r for r in json_rows
        if _get_calibration(r) == "ISOTONIC" and _get_status(r) == "REPLAY_READY"
    ]
    assert len(isotonic_ready) == 0, f"Found {len(isotonic_ready)} ISOTONIC rows with REPLAY_READY status"

    non_base_ready = [
        r for r in json_rows
        if _get_feature_set(r) != "BASE" and _get_status(r) == "REPLAY_READY"
    ]
    assert len(non_base_ready) == 0, f"Found {len(non_base_ready)} non-BASE rows with REPLAY_READY status"


def test_sequence_rows_retrain_required(json_rows: list[dict]):
    sequence_rows = [r for r in json_rows if _get_feature_set(r) == "SEQUENCE"]
    assert len(sequence_rows) > 0, "No SEQUENCE rows found in dataset"
    for r in sequence_rows:
        assert _get_status(r) == "RETRAIN_REQUIRED", (
            f"SEQUENCE row for model {_get_model_id(r)} (coverage={_get_coverage(r)}) is not RETRAIN_REQUIRED"
        )
