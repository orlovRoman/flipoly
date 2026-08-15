import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.threshold_optimizer import (
    TARGET_COVERAGES,
    classify_scores,
    evaluate_thresholded_polymarket,
    optimize_joint_thresholds,
    raw_opinion_for_score,
)


def _fixture():
    raw = np.asarray([0.05, 0.10, 0.20, 0.35, 0.45, 0.50, 0.55, 0.65, 0.80, 0.90, 0.95, 0.98])
    frame = pd.DataFrame({
        "market_id": [f"m{i}" for i in range(len(raw))],
        "asset": ["BTCUSDT"] * len(raw),
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "final_outcome": ["NO", "NO", "NO", "NO", "NO", "YES", "YES", "YES", "YES", "YES", "YES", "YES"],
        "recorded_at": pd.date_range("2026-01-01", periods=len(raw), freq="15min", tz="UTC"),
        "time_left_min": [10.0] * len(raw),
    })
    quotes = pd.DataFrame({
        "market_id": frame["market_id"],
        "best_ask": [0.20] * len(raw),
        "best_bid": [0.15] * len(raw),
    })
    return frame, raw, quotes


def test_classification_has_one_non_overlapping_dead_zone():
    assert classify_scores([0.1, 0.5, 0.9], lower=0.3, upper=0.7).tolist() == ["DOWN", "NONE", "UP"]
    with pytest.raises(ValueError, match="lower < upper"):
        classify_scores([0.1], lower=0.7, upper=0.3)
    assert raw_opinion_for_score(0.5) == "UP"
    assert raw_opinion_for_score(0.49) == "DOWN"


def test_threshold_audit_reports_direction_role_and_three_oot_windows():
    frame, raw, quotes = _fixture()
    result = evaluate_thresholded_polymarket(
        frame, raw, raw, quotes,
        lower=0.1, upper=0.8, strategy_branch="COMBINED",
        min_edge=0.0, cost_buffer=0.0,
    )
    assert result["lower_threshold"] < result["upper_threshold"]
    assert result["coverage_pct"] == pytest.approx(50.0)
    assert result["none_pct"] == pytest.approx(50.0)
    assert result["n_trades"] == 6
    assert {item["dimension"] for item in result["slices"]} == {"DIRECTION", "ROLE"}
    assert len(result["oot_windows"]) == 3


def test_joint_optimizer_sweeps_requested_coverages_and_never_overlaps():
    frame, raw, quotes = _fixture()
    result = optimize_joint_thresholds(
        frame, raw, raw, quotes,
        target_coverages=TARGET_COVERAGES,
        selected_target_coverage=0.4,
        min_edge=0.0, cost_buffer=0.0,
    )
    assert {row["target_coverage_pct"] for row in result["sweep"]} == {20.0, 40.0, 60.0, 80.0}
    assert result["selected_lower_threshold"] < result["selected_upper_threshold"]
    assert result["selected"]["coverage_pct"] >= 0.0
