"""Read-only classification helpers for saved LogReg OOF artifacts."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

import pandas as pd

CLOSE_TIME_PRIORITY = (
    "market_close_at",
    "resolved_at",
    "end_time_est",
    "market_start",
)
AUDIT_REQUIRED_FIELDS = (
    "market_id",
    "raw_score",
    "calibrated_score",
    "final_outcome",
    "quote_snapshot",
    "recorded_at",
)


def _has_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, pd.Series):
        return bool(value.notna().any())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _market_ids(frame: pd.DataFrame) -> list[str]:
    if "market_id" not in frame.columns:
        return []
    return frame["market_id"].dropna().astype(str).unique().tolist()


def _frame_source(frame: pd.DataFrame, market_id: str) -> str | None:
    if "market_id" not in frame.columns:
        return None
    rows = frame[frame["market_id"].astype(str) == market_id]
    for column in CLOSE_TIME_PRIORITY:
        if column not in rows.columns:
            continue
        if pd.to_datetime(rows[column], utc=True, errors="coerce").notna().any():
            return column
    return None


def recover_close_time_sources(
    frame: pd.DataFrame,
    live_markets: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Recover canonical close-time provenance without mutating either input."""
    ids = _market_ids(frame)
    recovered: dict[str, str] = {}
    joined: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    direct_count = 0

    for market_id in ids:
        source = _frame_source(frame, market_id)
        if source:
            recovered[market_id] = source
            direct_count += 1
            continue
        candidates = list(live_markets.get(market_id, ()))
        if len(candidates) != 1:
            (ambiguous if len(candidates) > 1 else missing).append(market_id)
            joined[market_id] = {"key": "market_id", "live_market_rows": len(candidates)}
            continue
        row = candidates[0]
        source = next(
            (
                logical
                for column, logical in (
                    ("end_date", "market_close_at"),
                    ("resolved_at", "resolved_at"),
                    ("end_time_est", "end_time_est"),
                )
                if row.get(column) is not None
            ),
            None,
        )
        joined[market_id] = {
            "key": "market_id",
            "live_market_rows": 1,
            "source": source,
        }
        if source:
            recovered[market_id] = source
        else:
            missing.append(market_id)

    source_counts = dict(Counter(recovered.values()))
    return {
        "status": "RECOVERED" if len(recovered) == len(ids) else "INCOMPLETE",
        "join_key": "market_id",
        "direct_count": direct_count,
        "joined_count": len(recovered) - direct_count,
        "missing_count": len(missing),
        "ambiguous_count": len(ambiguous),
        "market_count": len(ids),
        "source_counts": source_counts,
        "missing_market_ids": missing[:50],
        "ambiguous_market_ids": ambiguous[:50],
        "join_samples": dict(list(joined.items())[:10]),
    }


def classify_oof_artifact(
    *,
    model_registry_id: int,
    model_version: int | None,
    oof_artifact_id: int | None,
    artifact_blob: bytes,
    schema_version: int | None,
    row_count: int | None,
    frame: pd.DataFrame,
    quotes: pd.DataFrame,
    raw_scores: Iterable[Any] | None,
    calibrated_scores: Iterable[Any] | None,
    live_markets: Mapping[str, Iterable[Mapping[str, Any]]],
    evaluation_commit: str,
    metrics_schema_version: str,
    replay_status: str = "READ_ONLY_REPLAY_ELIGIBLE",
) -> dict[str, Any]:
    """Return a serializable audit record; this function never writes to a DB."""
    raw = list(raw_scores) if raw_scores is not None else []
    calibrated = list(calibrated_scores) if calibrated_scores is not None else []
    ids = _market_ids(frame)
    fields = {
        "market_id": bool(ids),
        "raw_score": len(raw) == len(frame) and bool(raw),
        "calibrated_score": len(calibrated) == len(frame) and bool(calibrated),
        "final_outcome": (
            "final_outcome" in frame.columns and _has_values(frame["final_outcome"])
        )
        or ("final_outcome" in quotes.columns and _has_values(quotes["final_outcome"])),
        "quote_snapshot": bool(len(quotes)) and any(
            column in quotes.columns
            for column in ("mid_price", "best_bid", "best_ask", "yes_price", "no_price")
        ),
        "recorded_at": "recorded_at" in frame.columns and _has_values(frame["recorded_at"]),
    }
    close_recovery = recover_close_time_sources(frame, live_markets)
    missing_fields = [name for name, present in fields.items() if not present]
    close_ok = close_recovery["status"] == "RECOVERED"
    valid = not missing_fields and bool(ids) and close_ok
    invalid_reason = None
    if not valid:
        invalid_reason = (
            "MISSING_CLOSE_TIME"
            if not close_ok
            else "MISSING_REQUIRED_FIELD"
        )
    return {
        "model_registry_id": model_registry_id,
        "model_version": model_version,
        "oof_artifact_id": oof_artifact_id,
        "artifact_schema_version": schema_version,
        "stored_row_count": row_count,
        "decoded_row_count": int(len(frame)),
        "quote_row_count": int(len(quotes)),
        "artifact_sha256": sha256(artifact_blob).hexdigest(),
        "frame_columns": list(frame.columns),
        "quote_columns": list(quotes.columns),
        "field_presence": fields,
        "missing_fields": missing_fields,
        "close_time_fields_present_in_artifact": [
            column for column in CLOSE_TIME_PRIORITY if column in frame.columns
        ],
        "close_time_recovery": close_recovery,
        "artifact_status": "VALID_FOR_REPLAY" if valid else "INVALID_OOT_ARTIFACT",
        "invalid_reason": invalid_reason,
        "replay_allowed": valid,
        "replay_status": replay_status if valid else "NOT_RUN_INVALID_OOT_ARTIFACT",
        "retrain_required": not valid,
        "evaluation_commit": evaluation_commit,
        "metrics_schema_version": metrics_schema_version,
    }


def build_audit_report(
    records: list[dict[str, Any]],
    *,
    generated_at: str,
    evaluation_commit: str,
    metrics_schema_version: str,
) -> dict[str, Any]:
    valid = sum(record["artifact_status"] == "VALID_FOR_REPLAY" for record in records)
    invalid = len(records) - valid
    recovery = [record["close_time_recovery"] for record in records]
    return {
        "report_version": "logreg_oof_artifact_audit_v1",
        "generated_at": generated_at,
        "candidate_id_range": [820, 879],
        "candidate_count_expected": 60,
        "candidate_count_observed": len(records),
        "evaluation_commit": evaluation_commit,
        "metrics_schema_version": metrics_schema_version,
        "read_only": True,
        "counts": {
            "total": len(records),
            "valid_for_replay": valid,
            "invalid_oot_artifact": invalid,
            "close_time_recovered_markets": sum(x["market_count"] - x["missing_count"] - x["ambiguous_count"] for x in recovery),
            "close_time_missing_markets": sum(x["missing_count"] for x in recovery),
            "close_time_ambiguous_markets": sum(x["ambiguous_count"] for x in recovery),
            "retrain_required": sum(record["retrain_required"] for record in records),
            "replay_allowed": sum(record["replay_allowed"] for record in records),
        },
        "records": records,
    }
