"""Versioned LightGBM experiment configuration contracts.

The dashboard and future optimizers use this module as the single boundary for
validating experiment parameters.  RuntimeSettings remains the compatibility
fallback, while a saved experiment is immutable and reproducible.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from polyflip.crypto.feature_sets import get_feature_set


MODEL_DEFAULTS: dict[str, int | float] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

CALIBRATION_DEFAULTS: dict[str, Any] = {"method": "AUTO"}

BACKTEST_DEFAULTS: dict[str, int | float] = {
    "min_edge": 0.04,
    "cost_buffer": 0.02,
    "fee_rate": 0.002,
    "min_price": 0.05,
    "max_price": 0.95,
    "outsider_max_price": 0.45,
    "stake_usdc": 1.0,
    "slippage_pct": 0.0,
}

_MODEL_BOUNDS: dict[str, tuple[float, float, type]] = {
    "n_estimators": (10, 5000, int),
    "learning_rate": (0.0001, 1.0, float),
    "num_leaves": (2, 512, int),
    "max_depth": (-1, 32, int),
    "min_child_samples": (1, 10000, int),
    "subsample": (0.1, 1.0, float),
    "colsample_bytree": (0.1, 1.0, float),
    "reg_alpha": (0.0, 1000.0, float),
    "reg_lambda": (0.0, 1000.0, float),
}

_BACKTEST_BOUNDS: dict[str, tuple[float, float]] = {
    "min_edge": (-1.0, 1.0),
    "cost_buffer": (0.0, 1.0),
    "fee_rate": (0.0, 1.0),
    "min_price": (0.001, 0.999),
    "max_price": (0.001, 0.999),
    "outsider_max_price": (0.001, 0.999),
    "stake_usdc": (0.000001, 1_000_000.0),
    "slippage_pct": (0.0, 0.999),
}


def _coerce_value(name: str, value: Any, *, integer: bool) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    try:
        coerced = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    return coerced


def _validate_group(
    values: Mapping[str, Any] | None,
    defaults: Mapping[str, int | float],
    bounds: Mapping[str, tuple[float, float] | tuple[float, float, type]],
) -> dict[str, int | float]:
    result = deepcopy(dict(defaults))
    for name, value in (values or {}).items():
        if name not in bounds:
            raise ValueError(f"Unknown experiment parameter: {name}")
        bound = bounds[name]
        lower, upper = bound[0], bound[1]
        integer = len(bound) == 3 and bound[2] is int
        coerced = _coerce_value(name, value, integer=integer)
        if not lower <= float(coerced) <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        result[name] = coerced
    return result


def normalize_experiment_config(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and canonicalize one config before it is persisted."""
    data = payload or {}
    feature_set = str(data.get("feature_set", "A")).strip().upper()
    feature_spec = get_feature_set(feature_set)
    calibration = deepcopy(CALIBRATION_DEFAULTS)
    calibration.update(data.get("calibration", {}) or {})
    method = str(calibration.get("method", "AUTO")).strip().upper()
    if method not in {"AUTO", "NONE", "TEMPERATURE", "PLATT", "ISOTONIC"}:
        raise ValueError("calibration.method must be AUTO, NONE, TEMPERATURE, PLATT or ISOTONIC")
    calibration["method"] = method
    backtest = _validate_group(data.get("backtest"), BACKTEST_DEFAULTS, _BACKTEST_BOUNDS)
    if backtest["min_price"] > backtest["max_price"]:
        raise ValueError("backtest.min_price must not exceed max_price")
    model = _validate_group(data.get("model"), MODEL_DEFAULTS, _MODEL_BOUNDS)
    return {
        "feature_set": feature_spec.key,
        "feature_set_version": feature_spec.version,
        "model": model,
        "calibration": calibration,
        "backtest": backtest,
    }


def experiment_config_hash(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
