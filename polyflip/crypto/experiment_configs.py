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
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

CALIBRATION_DEFAULTS: dict[str, Any] = {"method": "AUTO"}

THRESHOLD_DEFAULTS: dict[str, float] = {"target_coverage": 0.40}

BACKTEST_DEFAULTS: dict[str, Any] = {
    "min_edge": 0.04,
    "cost_buffer": 0.02,
    "fee_rate": 0.002,
    "min_price": 0.05,
    "max_price": 0.95,
    "outsider_max_price": 0.45,
    "stake_usdc": 1.0,
    "slippage_pct": 0.0,
    # ``LEGACY`` preserves the historical branch backtest.  ``WEIGHTED``
    # replays the same cost-aware scorer used by the runtime policy.
    "policy_mode": "LEGACY",
    "market_weight": 0.90,
    "logreg_weight": 0.05,
    "lgbm_weight": 0.05,
    "mrf_beta": 0.0,
    "weighted_fee_rate": 0.07,
    "weighted_fee_exponent": 1.0,
    "weighted_slippage_rate": 0.005,
    "execution_role": "TAKER",
}

# ``optimize_joint_thresholds`` still audits the directional LightGBM
# threshold itself. Weighted policy parameters are valid for the economic
# replay, but are not threshold-optimizer arguments. Keep this allow-list in
# one place so saved weighted configs do not break the legacy threshold audit.
LEGACY_THRESHOLD_BACKTEST_OPTION_KEYS = frozenset({
    "min_edge",
    "cost_buffer",
    "fee_rate",
    "min_price",
    "max_price",
    "outsider_max_price",
    "stake_usdc",
    "slippage_pct",
})


def legacy_threshold_backtest_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only options accepted by the directional threshold optimizer."""
    return {
        key: value
        for key, value in (options or {}).items()
        if key in LEGACY_THRESHOLD_BACKTEST_OPTION_KEYS
    }

_MODEL_BOUNDS: dict[str, tuple[float, float, type]] = {
    "n_estimators": (10, 5000, int),
    "learning_rate": (0.0001, 1.0, float),
    "num_leaves": (2, 512, int),
    "max_depth": (-1, 32, int),
    "min_child_samples": (1, 10000, int),
    "subsample": (0.1, 1.0, float),
    "subsample_freq": (0, 100, int),
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
    "market_weight": (0.0, 1.0),
    "logreg_weight": (0.0, 1.0),
    "lgbm_weight": (0.0, 1.0),
    "mrf_beta": (-2.0, 2.0),
    "weighted_fee_rate": (0.0, 1.0),
    "weighted_fee_exponent": (0.0, 16.0),
    "weighted_slippage_rate": (0.0, 0.999),
}

_BACKTEST_POLICY_MODES = {"LEGACY", "WEIGHTED", "WEIGHTED_SHADOW", "WEIGHTED_ACTIVE"}
_BACKTEST_EXECUTION_ROLES = {"MAKER", "TAKER"}

_THRESHOLD_BOUNDS: dict[str, tuple[float, float]] = {
    "target_coverage": (0.05, 0.95),
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
    if method not in {"AUTO", "NONE", "PLATT", "ISOTONIC"}:
        raise ValueError("calibration.method must be AUTO, NONE, PLATT or ISOTONIC")
    calibration["method"] = method
    raw_backtest = dict(data.get("backtest", {}) or {})
    policy_mode = str(raw_backtest.pop("policy_mode", BACKTEST_DEFAULTS["policy_mode"])).strip().upper()
    if policy_mode not in _BACKTEST_POLICY_MODES:
        raise ValueError(
            "backtest.policy_mode must be LEGACY, WEIGHTED, WEIGHTED_SHADOW or WEIGHTED_ACTIVE"
        )
    execution_role = str(raw_backtest.pop("execution_role", BACKTEST_DEFAULTS["execution_role"])).strip().upper()
    if execution_role not in _BACKTEST_EXECUTION_ROLES:
        raise ValueError("backtest.execution_role must be MAKER or TAKER")
    backtest = _validate_group(raw_backtest, BACKTEST_DEFAULTS, _BACKTEST_BOUNDS)
    backtest["policy_mode"] = policy_mode
    backtest["execution_role"] = execution_role
    if backtest["min_price"] > backtest["max_price"]:
        raise ValueError("backtest.min_price must not exceed max_price")
    model = _validate_group(data.get("model"), MODEL_DEFAULTS, _MODEL_BOUNDS)
    thresholds = _validate_group(data.get("thresholds"), THRESHOLD_DEFAULTS, _THRESHOLD_BOUNDS)
    return {
        "feature_set": feature_spec.key,
        "feature_set_version": feature_spec.version,
        "model": model,
        "calibration": calibration,
        "thresholds": thresholds,
        "backtest": backtest,
    }


def experiment_config_hash(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
