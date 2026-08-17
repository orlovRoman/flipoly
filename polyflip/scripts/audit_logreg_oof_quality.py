"""Build a detailed replay-quality report from the read-only OOF audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


INVALID_REASONS = {
    "MISSING_MARKET_ID",
    "MISSING_CLOSE_TIME",
    "AMBIGUOUS_CLOSE_TIME",
    "MISSING_SCORE",
    "MISSING_OUTCOME",
    "CHECKSUM_MISMATCH",
    "SCHEMA_MISMATCH",
    "AMBIGUOUS_JOIN",
}


def _checks(record: dict[str, Any]) -> dict[str, bool]:
    fields = record.get("field_presence", {})
    recovery = record.get("close_time_recovery", {})
    return {
        "schema_valid": (
            record.get("artifact_schema_version") == 2
            and record.get("stored_row_count") == record.get("decoded_row_count")
        ),
        "checksum_present": len(str(record.get("artifact_sha256", ""))) == 64,
        "model_registry_match": (
            record.get("model_registry_id") is not None
            and record.get("oof_artifact_id") is not None
        ),
        "market_id": bool(fields.get("market_id")),
        "raw_score": bool(fields.get("raw_score")),
        "calibrated_score": bool(fields.get("calibrated_score")),
        "final_outcome": bool(fields.get("final_outcome")),
        "quote_snapshot": bool(fields.get("quote_snapshot")),
        "recorded_at": bool(fields.get("recorded_at")),
        "close_time": recovery.get("status") == "RECOVERED",
        "no_missing_close_time": recovery.get("missing_count", 0) == 0,
        "no_ambiguous_close_time": recovery.get("ambiguous_count", 0) == 0,
        "no_ambiguous_join": recovery.get("ambiguous_count", 0) == 0,
    }


def _invalid_reason(checks: dict[str, bool]) -> str | None:
    if not checks["schema_valid"]:
        return "SCHEMA_MISMATCH"
    if not checks["checksum_present"]:
        return "CHECKSUM_MISMATCH"
    if not checks["model_registry_match"]:
        return "AMBIGUOUS_JOIN"
    if not checks["market_id"]:
        return "MISSING_MARKET_ID"
    if not checks["raw_score"] or not checks["calibrated_score"]:
        return "MISSING_SCORE"
    if not checks["final_outcome"]:
        return "MISSING_OUTCOME"
    if not checks["close_time"] or not checks["no_missing_close_time"]:
        return "MISSING_CLOSE_TIME"
    if not checks["no_ambiguous_close_time"]:
        return "AMBIGUOUS_CLOSE_TIME"
    if not checks["no_ambiguous_join"]:
        return "AMBIGUOUS_JOIN"
    if not checks["quote_snapshot"] or not checks["recorded_at"]:
        return "MISSING_OUTCOME"
    return None


def build_quality_report(audit: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for original in audit["records"]:
        checks = _checks(original)
        reason = _invalid_reason(checks)
        status = "VALID_FOR_REPLAY" if reason is None else "INVALID_OOT_ARTIFACT"
        records.append({
            "model_registry_id": original.get("model_registry_id"),
            "model_version": original.get("model_version"),
            "oof_artifact_id": original.get("oof_artifact_id"),
            "artifact_schema_version": original.get("artifact_schema_version"),
            "stored_row_count": original.get("stored_row_count"),
            "decoded_row_count": original.get("decoded_row_count"),
            "quote_row_count": original.get("quote_row_count"),
            "artifact_sha256": original.get("artifact_sha256"),
            "frame_columns": original.get("frame_columns", []),
            "quote_columns": original.get("quote_columns", []),
            "close_time_fields_present_in_artifact": original.get(
                "close_time_fields_present_in_artifact", []
            ),
            "close_time_recovery": original.get("close_time_recovery", {}),
            "checks": checks,
            "missing_fields": original.get("missing_fields", []),
            "artifact_status": status,
            "invalid_reason": reason,
            "replay_allowed": status == "VALID_FOR_REPLAY",
            "retrain_required": status != "VALID_FOR_REPLAY",
        })
    counts = {
        "total": len(records),
        "valid_for_replay": sum(r["artifact_status"] == "VALID_FOR_REPLAY" for r in records),
        "invalid_oot_artifact": sum(r["artifact_status"] == "INVALID_OOT_ARTIFACT" for r in records),
        "invalid_reasons": {
            reason: sum(r["invalid_reason"] == reason for r in records)
            for reason in sorted(INVALID_REASONS)
            if any(r["invalid_reason"] == reason for r in records)
        },
        "replay_allowed": sum(r["replay_allowed"] for r in records),
        "retrain_required": sum(r["retrain_required"] for r in records),
    }
    return {
        "report_version": "logreg_oof_quality_v1",
        "generated_at": audit.get("generated_at"),
        "candidate_id_range": [820, 879],
        "candidate_count_expected": 60,
        "candidate_count_observed": len(records),
        "source_audit": "reports/logreg_oof_artifact_audit_20260817.json",
        "evaluation_commit": audit.get("evaluation_commit"),
        "metrics_schema_version": audit.get("metrics_schema_version"),
        "read_only": True,
        "counts": counts,
        "records": records,
    }


def write_quality_report(audit_path: Path, out_dir: Path) -> None:
    json_path = out_dir / "logreg_oof_quality_20260817.json"
    csv_path = out_dir / "logreg_oof_quality_20260817.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("quality report already exists")
    report = build_quality_report(json.loads(audit_path.read_text(encoding="utf-8")))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    fields = [
        "model_registry_id", "model_version", "oof_artifact_id", "artifact_status",
        "invalid_reason", "artifact_schema_version", "stored_row_count",
        "decoded_row_count", "quote_row_count", "artifact_sha256", "replay_allowed",
        "retrain_required", "close_time_status", "close_time_join_key",
        "close_time_missing_count", "close_time_ambiguous_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in report["records"]:
            recovery = record["close_time_recovery"]
            writer.writerow({
                "model_registry_id": record["model_registry_id"],
                "model_version": record["model_version"],
                "oof_artifact_id": record["oof_artifact_id"],
                "artifact_status": record["artifact_status"],
                "invalid_reason": record["invalid_reason"],
                "artifact_schema_version": record["artifact_schema_version"],
                "stored_row_count": record["stored_row_count"],
                "decoded_row_count": record["decoded_row_count"],
                "quote_row_count": record["quote_row_count"],
                "artifact_sha256": record["artifact_sha256"],
                "replay_allowed": record["replay_allowed"],
                "retrain_required": record["retrain_required"],
                "close_time_status": recovery.get("status"),
                "close_time_join_key": recovery.get("join_key"),
                "close_time_missing_count": recovery.get("missing_count"),
                "close_time_ambiguous_count": recovery.get("ambiguous_count"),
            })


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    write_quality_report(
        root / "reports/logreg_oof_artifact_audit_20260817.json",
        root / "reports",
    )
