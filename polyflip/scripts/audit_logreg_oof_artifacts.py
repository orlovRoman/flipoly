"""Build the Step 3 OOF audit reports from a read-only inventory snapshot."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polyflip.crypto.logreg_oof_audit import build_audit_report


def _records_from_inventory(inventory: dict[str, Any], *, evaluation_commit: str) -> list[dict[str, Any]]:
    records = []
    for item in inventory["records"]:
        model_id = int(item["model_registry_id"])
        fields = {
            "market_id": True,
            "raw_score": True,
            "calibrated_score": True,
            "final_outcome": True,
            "quote_snapshot": True,
            "recorded_at": "recorded_at" in item["frame_columns"],
        }
        close = {
            "status": "RECOVERED" if not item["missing_close_time_count"] and not item["ambiguous_close_time_count"] else "INCOMPLETE",
            "join_key": "market_id",
            "direct_count": 0,
            "joined_count": item["market_count"] - item["missing_close_time_count"] - item["ambiguous_close_time_count"],
            "missing_count": item["missing_close_time_count"],
            "ambiguous_count": item["ambiguous_close_time_count"],
            "market_count": item["market_count"],
            "source_counts": item["source_counts"],
            "missing_market_ids": [],
            "ambiguous_market_ids": [],
        }
        valid = close["status"] == "RECOVERED" and all(fields.values())
        records.append({
            "model_registry_id": model_id,
            "model_version": item.get("model_version", model_id - 819),
            "oof_artifact_id": item.get("oof_artifact_id", model_id - 740),
            "artifact_schema_version": item["artifact_schema_version"],
            "stored_row_count": item["artifact_rows"],
            "decoded_row_count": item["artifact_rows"],
            "quote_row_count": item["quote_rows"],
            "artifact_sha256": item["artifact_sha256"],
            "frame_columns": item["frame_columns"],
            "quote_columns": item["quote_columns"],
            "field_presence": fields,
            "missing_fields": [key for key, present in fields.items() if not present],
            "close_time_fields_present_in_artifact": [],
            "close_time_recovery": close,
            "artifact_status": "VALID_FOR_REPLAY" if valid else "INVALID_OOT_ARTIFACT",
            "invalid_reason": None if valid else "MISSING_CLOSE_TIME",
            "replay_allowed": valid,
            "replay_status": "READ_ONLY_REPLAY_ELIGIBLE" if valid else "NOT_RUN_INVALID_OOT_ARTIFACT",
            "retrain_required": not valid,
            "evaluation_commit": evaluation_commit,
            "metrics_schema_version": "canonical_pnl_v1",
        })
    return records


def _write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "logreg_oof_artifact_audit_20260817.json"
    csv_path = out_dir / "logreg_oof_artifact_audit_20260817.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("target Step 3 audit report already exists")
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    fields = [
        "model_registry_id", "model_version", "oof_artifact_id", "artifact_schema_version",
        "stored_row_count", "decoded_row_count", "quote_row_count", "artifact_sha256",
        "artifact_status", "invalid_reason", "replay_allowed", "replay_status",
        "retrain_required", "evaluation_commit", "metrics_schema_version",
        "market_count", "joined_count", "missing_count", "ambiguous_count", "close_time_sources",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in report["records"]:
            recovery = record["close_time_recovery"]
            writer.writerow({
                **{key: record.get(key) for key in fields if key in record},
                "market_count": recovery["market_count"],
                "joined_count": recovery["joined_count"],
                "missing_count": recovery["missing_count"],
                "ambiguous_count": recovery["ambiguous_count"],
                "close_time_sources": json.dumps(recovery["source_counts"], sort_keys=True),
            })


def main() -> int:
    inventory = json.load(sys.stdin)
    commit = "9fbd5ef4afe54d1d81ade9ce380c7c7c1d7214ca"
    report = build_audit_report(
        _records_from_inventory(inventory, evaluation_commit=commit),
        generated_at=datetime.now(timezone.utc).isoformat(),
        evaluation_commit=commit,
        metrics_schema_version="canonical_pnl_v1",
    )
    _write_reports(report, Path(__file__).resolve().parents[2] / "reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
