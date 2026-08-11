"""Leakage-safe feature importance diagnostics for LightGBM experiments.

The audit is intentionally descriptive: it records how often a feature is
used across temporal folds, but it never removes a feature or blocks model
activation. This keeps A/B/C experiments comparable while making unstable or
consistently unused inputs visible before the next training run.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import logging
from typing import Any

import numpy as np


FEATURE_AUDIT_VERSION = "feature-audit-v1"
FEATURE_AUDIT_STABILITY_MIN = 0.60

logger = logging.getLogger(__name__)


def _coerce_importance(values: Any, feature_names: Sequence[str]) -> np.ndarray:
    """Return a finite, non-negative vector aligned to feature_names."""
    array = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    result = np.zeros(len(feature_names), dtype=float)
    if len(array) != len(feature_names):
        logger.warning(
            "feature_importance_length_mismatch expected=%d got=%d",
            len(feature_names),
            len(array),
        )
    count = min(len(result), len(array))
    if count:
        result[:count] = array[:count]
    result[~np.isfinite(result)] = 0.0
    return np.maximum(result, 0.0)


def model_gain_importance(model: Any, feature_names: Sequence[str]) -> np.ndarray:
    """Return finite gain importances, with a split-count fallback."""
    values: Any = None
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            values = booster.feature_importance(importance_type="gain")
            booster_names = tuple(booster.feature_name())
            if len(booster_names) == len(values):
                by_name = dict(zip(booster_names, values))
                feature_name_set = set(feature_names)
                missing = [name for name in feature_names if name not in by_name]
                extra = [name for name in booster_names if name not in feature_name_set]
                if missing or extra:
                    logger.warning(
                        "feature_importance_name_mismatch missing=%s extra=%s",
                        missing,
                        extra,
                    )
                return _coerce_importance(
                    [by_name.get(name, 0.0) for name in feature_names],
                    feature_names,
                )
        except Exception:
            values = None
    if values is None:
        values = getattr(model, "feature_importances_", None)
    return _coerce_importance(values, feature_names)


def summarize_fold_importance(
    fold_importances: Iterable[Sequence[float]],
    feature_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Summarize gain share, fold coverage, and rank for every feature."""
    names = tuple(feature_names)
    rows = [_coerce_importance(values, names) for values in fold_importances]
    if not rows:
        return {
            name: {"mean_gain_share": 0.0, "fold_presence": 0.0, "mean_rank": 0.0}
            for name in names
        }

    matrix = np.vstack(rows)
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    presence = (matrix > 0).mean(axis=0)

    ranks = np.zeros_like(matrix)
    for row_idx, row in enumerate(shares):
        order = np.argsort(-row, kind="mergesort")
        ranks[row_idx, order] = np.arange(1, len(names) + 1, dtype=float)
        ranks[row_idx, row <= 0] = 0.0

    audit: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(names):
        active_ranks = ranks[:, idx][ranks[:, idx] > 0]
        audit[name] = {
            "mean_gain_share": round(float(shares[:, idx].mean()), 6),
            "fold_presence": round(float(presence[idx]), 6),
            "mean_rank": round(float(active_ranks.mean()), 3) if len(active_ranks) else 0.0,
        }
    return audit


def feature_audit_summary(
    audit: dict[str, dict[str, float | int]],
    *,
    top_n: int = 5,
) -> dict[str, object]:
    """Build compact dashboard metadata without making a quality decision."""
    ordered = sorted(
        audit.items(),
        key=lambda item: (
            float(item[1].get("mean_gain_share", 0.0)),
            float(item[1].get("fold_presence", 0.0)),
        ),
        reverse=True,
    )
    stable = [
        name for name, values in ordered
        if float(values.get("fold_presence", 0.0)) >= FEATURE_AUDIT_STABILITY_MIN
        and float(values.get("mean_gain_share", 0.0)) > 0.0
    ]
    zero_gain = [
        name for name, values in audit.items()
        if float(values.get("mean_gain_share", 0.0)) == 0.0
    ]
    return {
        "version": FEATURE_AUDIT_VERSION,
        "stability_min": FEATURE_AUDIT_STABILITY_MIN,
        "stable_features": stable[:top_n],
        "top_features": [name for name, _ in ordered[:top_n]],
        "zero_gain_features": sorted(zero_gain),
        "feature_count": len(audit),
    }
