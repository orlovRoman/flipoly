"""Shared immutable artifact and evaluation-contract helpers for AI Lab."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any


class TrainingRows(list):
    """Rows loaded from one successful TRAIN result plus its artifact."""

    artifact: Any | None = None


def mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def parse_datetime(value: Any) -> datetime | None:
    """Parse current and legacy window values into comparable UTC datetimes."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical_window_value(value: Any) -> str | None:
    parsed = parse_datetime(value)
    if parsed is not None:
        return parsed.isoformat()
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pair(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("window_start"))
        end = value.get("end", value.get("window_end"))
        if start is None:
            start = value.get("oot_window_start", value.get("train_window_start"))
        if end is None:
            end = value.get("oot_window_end", value.get("train_window_end"))
        return (start, end) if start is not None or end is not None else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return (value[0], value[1]) if len(value) >= 2 else None
    return None


def resolve_window(sources: Sequence[Any], prefix: str) -> tuple[Any, Any] | None:
    """Resolve an explicit or legacy ``train``/``oot`` window."""
    names = (prefix, f"{prefix}_window", f"{prefix}ing_window")
    for source in sources:
        payload = mapping(source)
        if not payload:
            continue
        start = payload.get(f"{prefix}_window_start")
        end = payload.get(f"{prefix}_window_end")
        if start is None:
            start = payload.get(f"{prefix}ing_window_start")
        if end is None:
            end = payload.get(f"{prefix}ing_window_end")
        if start is not None or end is not None:
            return start, end
        for name in names:
            pair = _pair(payload.get(name))
            if pair is not None:
                return pair
    return None


def resolve_windows(sources: Sequence[Any], prefix: str) -> list[tuple[Any, Any]]:
    """Resolve a list of explicit windows from current or legacy payloads."""
    names = (f"{prefix}_windows", "oot_windows" if prefix == "oot" else "train_windows")
    for source in sources:
        payload = mapping(source)
        for name in names:
            values = payload.get(name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            windows = [pair for item in values if (pair := _pair(item)) is not None]
            if windows:
                return windows
    return []


def window_dict(window: tuple[Any, Any] | None) -> dict[str, str | None]:
    if window is None:
        return {"start": None, "end": None}
    return {
        "start": canonical_window_value(window[0]),
        "end": canonical_window_value(window[1]),
    }


def datetime_window(window: tuple[Any, Any] | None) -> tuple[datetime, datetime] | None:
    if window is None:
        return None
    start = parse_datetime(window[0])
    end = parse_datetime(window[1])
    return (start, end) if start is not None and end is not None else None


def dataset_fingerprint(rows: Sequence[Any]) -> str | None:
    values = sorted(
        str(getattr(row, "dataset_fingerprint"))
        for row in rows
        if getattr(row, "dataset_fingerprint", None)
    )
    if not values:
        return None
    if len(set(values)) == 1:
        return values[0]
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:32]


def bundle_bytes(
    rows: Sequence[Any],
    *,
    run_id: int,
    config_id: int,
    step_id: int,
    artifact_kind: str,
    target_semantics: str,
) -> bytes:
    """Serialize the exact registry bundle stored in ``AIModelArtifact``."""
    models = []
    for row in sorted(rows, key=lambda item: int(item.id)):
        raw = bytes(row.model_blob or b"")
        if not raw:
            raise ValueError(f"ModelRegistry {row.id} has no model_blob")
        models.append(
            {
                "id": int(row.id),
                "asset": str(row.asset),
                "version": int(row.version),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": base64.b64encode(raw).decode("ascii"),
            }
        )
    document = {
        "schema_version": 1,
        "artifact_kind": artifact_kind,
        "provenance": {
            "run_id": int(run_id),
            "config_id": int(config_id),
            "step_id": int(step_id),
        },
        "target_semantics": target_semantics,
        "models": models,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def artifact_metadata(
    *,
    context: Any,
    rows: Sequence[Any],
    artifact_kind: str,
    feature_pipeline_version: str,
    target_semantics: str,
    feature_semantics: Mapping[str, Any],
    train_window: tuple[Any, Any] | None,
    oot_window: tuple[Any, Any] | None,
    strategy_branch: str,
) -> dict[str, Any]:
    row_fingerprints = {
        str(row.asset): getattr(row, "dataset_fingerprint", None)
        for row in rows
    }
    return {
        "artifact_kind": artifact_kind,
        "artifact_id": None,
        "provenance": {
            "run_id": int(context.run_id),
            "config_id": int(context.config_id),
            "step_id": int(context.step_id),
            "model_registry_ids": [int(row.id) for row in rows],
        },
        "run_id": int(context.run_id),
        "config_id": int(context.config_id),
        "step_id": int(context.step_id),
        "model_registry_ids": [int(row.id) for row in rows],
        "dataset_fingerprint": dataset_fingerprint(rows),
        "dataset_fingerprints": row_fingerprints,
        "train_window": window_dict(train_window),
        "oot_window": window_dict(oot_window),
        "strategy_branch": strategy_branch,
        "feature_semantics": dict(feature_semantics),
        "target_semantics": target_semantics,
        "prediction_semantics": target_semantics,
        "feature_pipeline_version": feature_pipeline_version,
        "loadability": {"status": "VALID", "exact_bundle_bytes": True},
        "loadability_status": "VALID",
    }


def artifact_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
