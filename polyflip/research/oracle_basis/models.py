"""Probability metrics and baseline calibration for the oracle-basis study."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _probability_vector(values: Iterable[float], name: str) -> list[float]:
    probs = [float(v) for v in values]
    if not probs:
        raise ValueError(f"{name} is empty")
    for p in probs:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return probs


def _label_vector(values: Iterable[int], name: str) -> list[int]:
    labels = [int(v) for v in values]
    if not labels:
        raise ValueError(f"{name} is empty")
    for y in labels:
        if y not in (0, 1):
            raise ValueError(f"{name} must contain binary outcomes")
    return labels


def brier_score(p_up: Iterable[float], outcome_up: Iterable[int]) -> float:
    """Mean squared error of p_up against the actual binary outcome."""
    probs = _probability_vector(p_up, "p_up")
    labels = _label_vector(outcome_up, "outcome_up")
    if len(probs) != len(labels):
        raise ValueError("p_up and outcome_up must have equal length")
    return float(sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs))


def log_loss(
    p_up: Iterable[float], outcome_up: Iterable[int], eps: float = 1e-6
) -> float:
    """Natural-log loss against the actual outcome with fixed clipping."""
    if not math.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    probs = _probability_vector(p_up, "p_up")
    labels = _label_vector(outcome_up, "outcome_up")
    if len(probs) != len(labels):
        raise ValueError("p_up and outcome_up must have equal length")
    total = 0.0
    for p, y in zip(probs, labels):
        q = min(1.0 - eps, max(eps, p))
        total += y * math.log(q) + (1 - y) * math.log(1.0 - q)
    return float(-total / len(probs))


def hit_rate(
    p_up: Iterable[float], outcome_up: Iterable[int], threshold: float = 0.5
) -> float:
    """Directional measurement only; not a passing criterion by itself."""
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be in (0, 1)")
    probs = _probability_vector(p_up, "p_up")
    labels = _label_vector(outcome_up, "outcome_up")
    if len(probs) != len(labels):
        raise ValueError("p_up and outcome_up must have equal length")
    hits = sum(1 for p, y in zip(probs, labels) if (p >= threshold) == bool(y))
    return hits / len(probs)


def fit_b0_calibration(mid_train: Iterable[float], outcome_train: Iterable[int]):
    """Calibrate B0 mid-based probabilities on train only (Platt scaling)."""
    from sklearn.linear_model import LogisticRegression

    mids = _probability_vector(mid_train, "mid_train")
    labels = _label_vector(outcome_train, "outcome_train")
    if len(mids) != len(labels):
        raise ValueError("mid_train and outcome_train must have equal length")
    if len(set(labels)) != 2:
        raise ValueError("calibration needs both outcome classes on train")
    model = LogisticRegression(solver="lbfgs")
    model.fit(
        np.asarray(mids, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int)
    )
    return model


def apply_b0_calibration(model, mid_eval: Iterable[float]) -> list[float]:
    """Apply a train-fitted B0 calibration to evaluation mids."""
    mids = _probability_vector(mid_eval, "mid_eval")
    probs = model.predict_proba(np.asarray(mids, dtype=float).reshape(-1, 1))[:, 1]
    return [float(p) for p in probs]


def compare_incremental(
    base: dict[str, float], upgraded: dict[str, float]
) -> dict[str, float]:
    """Difference of upgraded minus base metrics on identical eligible rows."""
    keys = set(base) | set(upgraded)
    result = {}
    for key in sorted(keys):
        if key not in base or key not in upgraded:
            raise ValueError(f"metric {key!r} must exist for both models")
        result[key] = float(upgraded[key]) - float(base[key])
    return result


def summarize_probabilities(p_up: Sequence[float]) -> dict[str, float]:
    probs = _probability_vector(p_up, "p_up")
    ordered = sorted(probs)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {
        "n": float(n),
        "min": ordered[0],
        "median": float(median),
        "max": ordered[-1],
    }
