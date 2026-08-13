"""Canonical, hashable manifests for reproducible AI Lab work."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Mapping


class ManifestError(ValueError):
    """Raised when a manifest is not deterministic or misses required fields."""


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_normalize(item) for item in sorted(value, key=repr)]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ManifestError("naive datetimes are not allowed in manifests")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError("non-finite floats are not allowed in manifests")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ManifestError(f"unsupported manifest value: {type(value).__name__}")


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return deterministic JSON suitable for hashing and audit storage."""
    normalized = _normalize(payload)
    if not isinstance(normalized, dict):
        raise ManifestError("manifest payload must be a mapping")
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_manifest_hash(payload: Mapping[str, Any]) -> str:
    """Compute the content hash of a canonical manifest payload."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _build_manifest(
    *,
    kind: str,
    payload: Mapping[str, Any],
    required: tuple[str, ...],
) -> dict[str, Any]:
    missing = [key for key in required if key not in payload or payload[key] in (None, "")]
    if missing:
        raise ManifestError(f"{kind} manifest is missing required fields: {missing}")
    result = dict(payload)
    result["manifest_kind"] = kind
    result.setdefault("schema_version", "1")
    result["manifest_hash"] = compute_manifest_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    return _normalize(result)


def build_experiment_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build an immutable manifest for training/backtest reproducibility."""
    return _build_manifest(
        kind="experiment",
        payload=payload,
        required=(
            "code_sha",
            "dataset_fingerprint",
            "feature_pipeline_version",
            "train_window",
            "oot_window",
            "seed",
            "model_params",
            "strategy_params",
            "backtest_params",
        ),
    )


def build_deployment_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build an immutable model/strategy bundle for activation or rollback."""
    return _build_manifest(
        kind="deployment",
        payload=payload,
        required=("models", "strategy", "risk_policy", "execution_policy"),
    )
