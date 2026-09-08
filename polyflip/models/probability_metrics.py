"""
polyflip/models/probability_metrics.py

Single canonical implementation of probability calibration and loss metrics:
- Brier Score
- Log Loss
- Expected Calibration Error (ECE) with 0.05 binning (20 bins)
- Reliability Diagrams / Tables

Eliminates ad-hoc `ece or 0.5` fallbacks; returns None/uncertainty for small N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, Any


def brier_score(
    y_true: Sequence[float | int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Mean squared probability error."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must have identical length")
    if len(y_t) == 0:
        return 0.0
    weights = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
    return float(np.average((y_t - y_p) ** 2, weights=weights))


def log_loss_score(
    y_true: Sequence[float | int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    eps: float = 1e-15,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Cross-entropy / log loss with probability clipping."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must have identical length")
    if len(y_t) == 0:
        return 0.0
    weights = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
    loss = -(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
    return float(np.average(loss, weights=weights))


def expected_calibration_error(
    y_true: Sequence[float | int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 20,
    min_samples: int = 10,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """
    Expected Calibration Error with uniform bins of width 1.0 / n_bins (default 0.05).
    Returns (ece, diagnostics). If samples < min_samples, returns (None, diagnostics)
    without fabricating arbitrary ECE = 0.5.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    n = len(y_t)
    if n != len(y_p):
        raise ValueError("y_true and y_prob must have identical length")

    weights = (
        np.asarray(sample_weight, dtype=float)
        if sample_weight is not None
        else np.ones(n, dtype=float)
    )

    if n < min_samples:
        return None, {
            "sample_count": n,
            "status": "TOO_FEW_SAMPLES",
            "min_required": min_samples,
            "n_bins": n_bins,
        }

    # Bins 0..n_bins-1, where bin width is 1.0 / n_bins (0.05 for n_bins=20)
    bin_ids = np.minimum((np.clip(y_p, 0.0, 1.0) * n_bins).astype(int), n_bins - 1)
    total_weight = float(weights.sum())

    ece = 0.0
    bin_records = []
    for b in range(n_bins):
        mask = (bin_ids == b)
        count = int(mask.sum())
        bin_lower = b / float(n_bins)
        bin_upper = (b + 1) / float(n_bins)
        if count > 0:
            w_sub = weights[mask]
            w_sum = float(w_sub.sum())
            p_mean = float(np.average(y_p[mask], weights=w_sub))
            obs_mean = float(np.average(y_t[mask], weights=w_sub))
            weight_fraction = w_sum / total_weight if total_weight > 0 else count / n
            cal_err = abs(obs_mean - p_mean)
            ece += weight_fraction * cal_err
            bin_records.append({
                "bin": b,
                "lower": round(bin_lower, 4),
                "upper": round(bin_upper, 4),
                "count": count,
                "predicted_mean": round(p_mean, 6),
                "observed_mean": round(obs_mean, 6),
                "error": round(cal_err, 6),
            })
        else:
            bin_records.append({
                "bin": b,
                "lower": round(bin_lower, 4),
                "upper": round(bin_upper, 4),
                "count": 0,
                "predicted_mean": None,
                "observed_mean": None,
                "error": None,
            })

    return round(float(ece), 6), {
        "sample_count": n,
        "status": "VALID",
        "n_bins": n_bins,
        "bins": bin_records,
    }


def reliability_table(
    y_true: Sequence[float | int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 20,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build detailed reliability table for dashboard and reports."""
    _, diag = expected_calibration_error(
        y_true, y_prob, n_bins=n_bins, min_samples=1, sample_weight=sample_weight
    )
    return diag.get("bins", [])
