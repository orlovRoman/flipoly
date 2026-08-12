"""Serialization helpers for reproducible LightGBM OOF backtests."""
from __future__ import annotations

import gzip
import pickle
from typing import Any

import numpy as np
import pandas as pd

OOF_ARTIFACT_SCHEMA_VERSION = 1
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


def _frame_columns(frame: pd.DataFrame) -> list[str]:
    preferred = (
        "market_id", "asset", "market_start", "recorded_at",
        "time_left_min", "vol_regime", "target", "final_outcome",
    )
    return [column for column in preferred if column in frame.columns]


def serialize_oof_artifact(
    frame: pd.DataFrame,
    oof_scores: Any,
    quotes: pd.DataFrame | None,
    *,
    feature_set: str,
    feature_schema_hash: str | None = None,
) -> bytes:
    """Serialize aligned OOF rows and quotes without storing model features."""
    if "market_id" not in frame.columns or "target" not in frame.columns:
        raise ValueError("OOF artifact requires market_id and target columns")
    scores = np.asarray(oof_scores, dtype=np.float64)
    if len(scores) != len(frame):
        raise ValueError("OOF scores must align with artifact frame")
    rows = frame[_frame_columns(frame)].reset_index(drop=True).copy()
    quote_rows = quotes.copy() if quotes is not None else pd.DataFrame()
    if not quote_rows.empty and "market_id" in quote_rows.columns:
        quote_rows = quote_rows.drop_duplicates("market_id", keep="first").reset_index(drop=True)
    payload = {
        "schema_version": OOF_ARTIFACT_SCHEMA_VERSION,
        "feature_set": feature_set,
        "feature_schema_hash": feature_schema_hash,
        "frame": rows,
        "quotes": quote_rows,
        "oof_scores": scores,
    }
    return gzip.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6)


def deserialize_oof_artifact(blob: bytes) -> dict[str, Any]:
    """Load and validate an OOF artifact before it reaches the backtester."""
    if not blob or len(blob) > _MAX_ARTIFACT_BYTES:
        raise ValueError("OOF artifact is empty or exceeds the size limit")
    try:
        payload = pickle.loads(gzip.decompress(blob))
    except Exception as exc:  # pragma: no cover - exact pickle errors vary
        raise ValueError(f"Invalid OOF artifact: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != OOF_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported OOF artifact schema")
    frame = payload.get("frame")
    scores = np.asarray(payload.get("oof_scores"), dtype=np.float64)
    if not isinstance(frame, pd.DataFrame) or "market_id" not in frame or "target" not in frame:
        raise ValueError("OOF artifact frame is invalid")
    if len(frame) != len(scores):
        raise ValueError("OOF artifact rows and scores are misaligned")
    quotes = payload.get("quotes")
    if quotes is None:
        quotes = pd.DataFrame()
    if not isinstance(quotes, pd.DataFrame):
        raise ValueError("OOF artifact quotes are invalid")
    payload["oof_scores"] = scores
    payload["quotes"] = quotes
    return payload
