"""Persist a read-only replay payload produced from the remote DB snapshot."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPORT_JSON = "logreg_candidate_replay_20260817_v2.json"
REPORT_CSV = "logreg_candidate_replay_20260817_v2.csv"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in payload["records"]:
        for calibration, calibration_payload in record["evaluations"].items():
            for window, branches in calibration_payload["windows"].items():
                for branch, metrics in branches.items():
                    rows.append({
                        "model_registry_id": record["model_registry_id"],
                        "model_version": record["model_version"],
                        "oof_artifact_id": record["oof_artifact_id"],
                        "artifact_status": record["artifact_status"],
                        "evaluation_commit": record["evaluation_commit"],
                        "metrics_schema_version": record["metrics_schema_version"],
                        "evaluation_protocol_version": record["evaluation_protocol_version"],
                        "calibration": calibration,
                        "window": window,
                        "strategy_branch": branch,
                        **metrics,
                    })
    return rows


def write_reports(
    payload: dict[str, Any],
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    if payload.get("candidate_count_observed") != 60:
        raise ValueError("replay payload must contain exactly 60 candidates")
    if any(record.get("artifact_status") != "VALID_FOR_REPLAY" for record in payload["records"]):
        raise ValueError("invalid OOT artifacts must not be replayed")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPORT_JSON
    csv_path = out_dir / REPORT_CSV
    if not overwrite and (json_path.exists() or csv_path.exists()):
        raise FileExistsError("target replay report already exists")
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    rows = _rows(payload)
    fieldnames = [
        "model_registry_id", "model_version", "oof_artifact_id", "artifact_status",
        "evaluation_commit", "metrics_schema_version", "evaluation_protocol_version",
        "calibration", "window", "strategy_branch", "coverage_pct", "n_trades",
        "win_rate", "net_profit", "roi_pct", "max_drawdown_usdc", "brier", "ece", "log_loss",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def main() -> int:
    payload = json.load(sys.stdin)
    write_reports(payload, Path(__file__).resolve().parents[2] / "reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
