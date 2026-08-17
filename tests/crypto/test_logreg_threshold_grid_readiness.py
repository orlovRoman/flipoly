"""Deterministic pytest test suite for logistic regression threshold grid readiness.

Validates threshold pair generation, score classification, and threshold grid
readiness report invariants.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from polyflip.crypto.threshold_optimizer import (
    TARGET_COVERAGES,
    ThresholdPair,
    classify_scores,
    threshold_grid,
)


def _get_report_path() -> Path:
    """Resolve the path to the threshold grid readiness report."""
    candidates = [
        Path("reports/logreg_threshold_grid_20260817.json"),
        Path(__file__).resolve().parents[2] / "reports" / "logreg_threshold_grid_20260817.json",
        Path(__file__).resolve().parents[3] / "reports" / "logreg_threshold_grid_20260817.json",
        Path(__file__).resolve().parents[4] / "reports" / "logreg_threshold_grid_20260817.json",
        Path("C:/Users/orlov/.gemini/antigravity/scratch/flipoly/reports/logreg_threshold_grid_20260817.json"),
        Path("C:/Users/orlov/.gemini/antigravity-cli/scratch/flipoly/reports/logreg_threshold_grid_20260817.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_report_rows() -> List[Dict[str, Any]]:
    """Load and return rows from the logreg threshold grid report."""
    report_path = _get_report_path()
    assert report_path.exists(), f"Report file not found: {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("rows", data.get("grid", []))
    raise AssertionError(f"Unexpected JSON structure in {report_path}: {type(data)}")


def test_threshold_pair_ordering_and_target_coverages() -> None:
    """1. Assert every generated pair has lower_threshold < upper_threshold for each target coverage 0.20, 0.40, 0.60, 0.80."""
    expected_coverages = {0.20, 0.40, 0.60, 0.80}
    actual_coverages = {round(float(c), 2) for c in TARGET_COVERAGES}
    assert actual_coverages == expected_coverages, (
        f"TARGET_COVERAGES {TARGET_COVERAGES} does not match expected {expected_coverages}"
    )

    scores = np.linspace(0.01, 0.99, 1000)
    pairs: List[ThresholdPair] = [
        pair
        for target in sorted(expected_coverages)
        for pair in threshold_grid(scores, target_coverage=target)
    ]

    assert len(pairs) >= len(expected_coverages)
    for pair in pairs:
        assert pair.lower < pair.upper, (
            f"Expected lower < upper, got {pair.lower} >= {pair.upper}"
        )
        assert math.isfinite(pair.lower)
        assert math.isfinite(pair.upper)


def test_threshold_grid_signal_coverage_on_fixed_scores() -> None:
    """2. Assert threshold_grid on a fixed sorted score array produces finite bounded pairs and a direction-neutral signal coverage within tolerance."""
    scores = np.linspace(0.0, 1.0, 1001)

    for target_cov in [0.20, 0.40, 0.60, 0.80]:
        pair = threshold_grid(scores, target_coverage=target_cov)[0]
        assert math.isfinite(pair.lower)
        assert math.isfinite(pair.upper)
        assert 0.0 <= pair.lower <= 1.0
        assert 0.0 <= pair.upper <= 1.0
        assert pair.lower < pair.upper

        signals = classify_scores(scores, lower=pair.lower, upper=pair.upper)

        signals_arr = np.asarray(signals)
        non_neutral_mask = np.isin(signals_arr, ["UP", "DOWN"])
        signal_coverage = float(np.mean(non_neutral_mask))

        # Quantile discretization tolerance
        assert abs(signal_coverage - target_cov) <= 0.05, (
            f"Coverage {signal_coverage} for target {target_cov} exceeded tolerance 0.05"
        )


def test_logreg_threshold_grid_report_structure() -> None:
    """3. Load reports/logreg_threshold_grid_20260817.json and assert:
    - 240 rows
    - target coverage set {0.2, 0.4, 0.6, 0.8}
    - 20 distinct asset/phase slices
    - all status UNAVAILABLE_WITHOUT_OOF_SCORE_ARRAYS
    - all future_data_used false
    - all threshold/sample/trade/actual coverage fields null
    """
    rows = _load_report_rows()
    assert len(rows) == 240, f"Expected 240 rows in report, got {len(rows)}"

    target_coverages = {r["target_coverage"] for r in rows if "target_coverage" in r}
    assert target_coverages == {0.2, 0.4, 0.6, 0.8} or target_coverages == {0.20, 0.40, 0.60, 0.80}

    slices = {(r["asset"], r["phase"]) for r in rows if "asset" in r and "phase" in r}
    assert len(slices) == 20, f"Expected 20 distinct asset/phase slices, got {len(slices)}"

    null_field_keys = [
        "lower_threshold",
        "upper_threshold",
        "up_lower_threshold",
        "up_upper_threshold",
        "down_lower_threshold",
        "down_upper_threshold",
        "actual_coverage",
        "sample_count",
        "sample_size",
        "trade_count",
        "n_samples",
        "n_trades",
        "coverage",
    ]

    for idx, row in enumerate(rows):
        assert row.get("status") == "UNAVAILABLE_WITHOUT_OOF_SCORE_ARRAYS", (
            f"Row {idx} status {row.get('status')} != UNAVAILABLE_WITHOUT_OOF_SCORE_ARRAYS"
        )
        assert row.get("future_data_used") is False, (
            f"Row {idx} future_data_used is {row.get('future_data_used')}, expected False"
        )

        for key in null_field_keys:
            if key in row:
                assert row[key] is None, f"Row {idx} expected field '{key}' to be null, got {row[key]}"


def test_report_no_mixed_asset_phase_aggregation() -> None:
    """4. Assert the report has no mixed asset/phase aggregation: every row has non-empty asset and phase and each candidate has four coverage rows per slice."""
    rows = _load_report_rows()

    candidate_slice_coverages = defaultdict(list)

    for idx, row in enumerate(rows):
        asset = row.get("asset")
        phase = row.get("phase")
        assert asset is not None and str(asset).strip() != "", f"Row {idx} has empty or missing asset"
        assert phase is not None and str(phase).strip() != "", f"Row {idx} has empty or missing phase"

        candidate = (
            row.get("model_registry_id")
            or row.get("candidate")
            or row.get("candidate_id")
            or row.get("model")
            or row.get("model_name")
            or "default"
        )
        key = (str(asset).strip(), str(phase).strip(), str(candidate).strip())
        target_cov = row.get("target_coverage")
        assert target_cov is not None, f"Row {idx} missing target_coverage"
        candidate_slice_coverages[key].append(round(float(target_cov), 2))

    for key, coverages in candidate_slice_coverages.items():
        assert len(coverages) == 4, (
            f"Slice/candidate {key} expected 4 coverage rows, got {len(coverages)}"
        )
        assert set(coverages) == {0.20, 0.40, 0.60, 0.80}, (
            f"Slice/candidate {key} coverage set mismatch: {set(coverages)}"
        )
