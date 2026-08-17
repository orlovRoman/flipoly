from __future__ import annotations

import json
from pathlib import Path

from polyflip.scripts.audit_logreg_oof_quality import build_quality_report


def _record(*, close_status: str = "RECOVERED", market_id: bool = True) -> dict:
    return {
        "model_registry_id": 820,
        "model_version": 48,
        "oof_artifact_id": 80,
        "artifact_schema_version": 2,
        "stored_row_count": 2,
        "decoded_row_count": 2,
        "quote_row_count": 1,
        "artifact_sha256": "a" * 64,
        "frame_columns": ["market_id", "recorded_at", "final_outcome"],
        "quote_columns": ["market_id", "mid_price"],
        "field_presence": {
            "market_id": market_id,
            "raw_score": True,
            "calibrated_score": True,
            "final_outcome": True,
            "quote_snapshot": True,
            "recorded_at": True,
        },
        "missing_fields": [],
        "close_time_fields_present_in_artifact": [],
        "close_time_recovery": {
            "status": close_status,
            "join_key": "market_id",
            "missing_count": 0 if close_status == "RECOVERED" else 1,
            "ambiguous_count": 0,
        },
    }


def test_quality_report_classifies_valid_and_invalid_records():
    report = build_quality_report(
        {
            "generated_at": "now",
            "evaluation_commit": "test",
            "metrics_schema_version": "canonical_pnl_v1",
            "records": [_record(), _record(close_status="INCOMPLETE")],
        }
    )

    assert report["counts"]["valid_for_replay"] == 1
    assert report["counts"]["invalid_oot_artifact"] == 1
    assert report["records"][0]["artifact_status"] == "VALID_FOR_REPLAY"
    assert report["records"][1]["invalid_reason"] == "MISSING_CLOSE_TIME"


def test_persisted_quality_report_has_explicit_status_for_all_candidates():
    path = Path(__file__).resolve().parents[2] / "reports/logreg_oof_quality_20260817.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["candidate_count_observed"] == 60
    assert len(report["records"]) == 60
    assert all(record["artifact_status"] in {"VALID_FOR_REPLAY", "INVALID_OOT_ARTIFACT"} for record in report["records"])
