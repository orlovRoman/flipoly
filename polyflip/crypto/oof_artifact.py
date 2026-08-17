"""Safe serialization helpers for reproducible LightGBM OOF backtests."""
from __future__ import annotations

import gzip
import io
import json
from typing import Any

import numpy as np
import pandas as pd

# Version 2 removes pickle from the artifact boundary. Existing v1 blobs must
# be retrained rather than deserialized through an executable format.
OOF_ARTIFACT_SCHEMA_VERSION = 2
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_ROWS = 500_000
_DATETIME_COLUMNS = {
    "market_start", "recorded_at", "market_close_at", "resolved_at", "end_time_est",
}


def _frame_columns(frame: pd.DataFrame) -> list[str]:
    preferred = (
        "market_id", "asset", "market_start", "market_close_at", "resolved_at",
        "end_time_est", "recorded_at", "time_left_min", "vol_regime", "mid_price",
        "spread", "best_bid", "best_ask", "yes_price", "no_price", "target",
        "final_outcome",
    )
    return [column for column in preferred if column in frame.columns]


def _json_frame(frame: pd.DataFrame, *, label: str) -> dict[str, Any]:
    if len(frame) > _MAX_ROWS:
        raise ValueError(f"{label} exceeds the row limit")
    frame = frame.reset_index(drop=True)
    try:
        encoded = frame.to_json(
            orient="split", date_format="iso", double_precision=15,
            force_ascii=True,
        )
        value = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} cannot be encoded as JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("columns"), list) or not isinstance(value.get("data"), list):
        raise ValueError(f"{label} JSON shape is invalid")
    return value


def _decode_frame(value: Any, *, label: str) -> pd.DataFrame:
    if not isinstance(value, dict):
        raise ValueError(f"{label} payload is invalid")
    columns = value.get("columns")
    data = value.get("data")
    if (
        not isinstance(columns, list)
        or len(columns) > 64
        or any(not isinstance(column, str) for column in columns)
        or len(set(columns)) != len(columns)
        or not isinstance(data, list)
        or len(data) > _MAX_ROWS
    ):
        raise ValueError(f"{label} schema is invalid")
    width = len(columns)
    if any(not isinstance(row, list) or len(row) != width for row in data):
        raise ValueError(f"{label} rows are invalid")
    frame = pd.DataFrame(data, columns=columns)
    for column in _DATETIME_COLUMNS.intersection(frame.columns):
        original = frame[column]
        parsed = pd.to_datetime(original, utc=True, errors="coerce")
        if parsed.isna().any() and original.notna().any():
            raise ValueError(f"{label} contains invalid {column}")
        frame[column] = parsed
    return frame


def _decompress(blob: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as stream:
            data = stream.read(_MAX_DECOMPRESSED_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise ValueError(f"Invalid OOF artifact compression: {exc}") from exc
    if len(data) > _MAX_DECOMPRESSED_BYTES:
        raise ValueError("OOF artifact exceeds the decompressed size limit")
    return data


def serialize_oof_artifact(
    frame: pd.DataFrame,
    oof_scores: Any,
    quotes: pd.DataFrame | None,
    *,
    feature_set: str,
    feature_schema_hash: str | None = None,
    raw_scores: Any | None = None,
) -> bytes:
    """Serialize aligned OOF rows and quotes without executable payloads."""
    if "market_id" not in frame.columns or "target" not in frame.columns:
        raise ValueError("OOF artifact requires market_id and target columns")
    scores = np.asarray(oof_scores, dtype=np.float64)
    if len(scores) != len(frame):
        raise ValueError("OOF scores must align with artifact frame")
    if np.isinf(scores).any() or ((scores[~np.isnan(scores)] < 0) | (scores[~np.isnan(scores)] > 1)).any():
        raise ValueError("OOF scores must be in [0, 1] or NaN")
    rows = frame[_frame_columns(frame)].reset_index(drop=True).copy()
    quote_rows = quotes.copy() if quotes is not None else pd.DataFrame()
    if not quote_rows.empty and "market_id" in quote_rows.columns:
        quote_rows = quote_rows.drop_duplicates("market_id", keep="first").reset_index(drop=True)
    raw = scores if raw_scores is None else np.asarray(raw_scores, dtype=np.float64)
    if len(raw) != len(frame):
        raise ValueError("raw OOF scores must align with artifact frame")
    if np.isinf(raw).any() or ((raw[~np.isnan(raw)] < 0) | (raw[~np.isnan(raw)] > 1)).any():
        raise ValueError("raw OOF scores must be in [0, 1] or NaN")
    payload = {
        "schema_version": OOF_ARTIFACT_SCHEMA_VERSION,
        "feature_set": str(feature_set),
        "feature_schema_hash": feature_schema_hash,
        "frame": _json_frame(rows, label="frame"),
        "quotes": _json_frame(quote_rows, label="quotes"),
        "oof_scores": [None if np.isnan(value) else float(value) for value in scores],
        "raw_oof_scores": [None if np.isnan(value) else float(value) for value in raw],
    }
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return gzip.compress(raw, compresslevel=6)


def deserialize_oof_artifact(blob: bytes) -> dict[str, Any]:
    """Load and validate a JSON OOF artifact; never execute object hooks."""
    if not blob or len(blob) > _MAX_ARTIFACT_BYTES:
        raise ValueError("OOF artifact is empty or exceeds the size limit")
    try:
        payload = json.loads(_decompress(blob).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid OOF artifact JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != OOF_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported OOF artifact schema")
    frame = _decode_frame(payload.get("frame"), label="frame")
    quotes = _decode_frame(payload.get("quotes"), label="quotes")
    raw_scores = payload.get("oof_scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != len(frame):
        raise ValueError("OOF artifact rows and scores are misaligned")
    scores: list[float] = []
    for value in raw_scores:
        if value is None:
            scores.append(np.nan)
            continue
        if isinstance(value, bool):
            raise ValueError("OOF score is invalid")
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("OOF score is invalid") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("OOF score is outside [0, 1]")
        scores.append(score)
    raw_payload = payload.get("raw_oof_scores", raw_scores)
    if not isinstance(raw_payload, list) or len(raw_payload) != len(frame):
        raise ValueError("OOF artifact rows and raw scores are misaligned")
    raw_values: list[float] = []
    for value in raw_payload:
        if value is None:
            raw_values.append(np.nan)
            continue
        if isinstance(value, bool):
            raise ValueError("raw OOF score is invalid")
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("raw OOF score is invalid") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("raw OOF score is outside [0, 1]")
        raw_values.append(score)
    return {
        "schema_version": OOF_ARTIFACT_SCHEMA_VERSION,
        "feature_set": payload.get("feature_set"),
        "feature_schema_hash": payload.get("feature_schema_hash"),
        "frame": frame,
        "quotes": quotes,
        "oof_scores": np.asarray(scores, dtype=np.float64),
        "raw_oof_scores": np.asarray(raw_values, dtype=np.float64),
    }
