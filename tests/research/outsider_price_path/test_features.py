"""
tests/research/outsider_price_path/test_features.py

Comprehensive unit tests for causal trajectory feature extraction (Stage 3):
- Monotonic fall
- Fall and rebound
- Trough before peak (anti-leakage guarantee)
- No movement / flat series
- Duplicate snapshots (continuous time invariance)
- Strict causality (future snapshots do not leak)
"""
import math
from datetime import datetime, timezone, timedelta
import pytest

from polyflip.research.outsider_price_path.features import (
    Observation,
    TrajectoryFeatures,
    compute_trajectory_features,
)


def _dt(sec: int) -> datetime:
    return datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def test_monotonic_fall():
    """Test monotonic price decrease with no subsequent rebound."""
    decision_at = _dt(600)
    # Price drops from 0.80 down to 0.20 steadily
    snaps = [
        Observation(_dt(0), 0.80),
        Observation(_dt(60), 0.70),
        Observation(_dt(120), 0.60),
        Observation(_dt(180), 0.50),
        Observation(_dt(240), 0.40),
        Observation(_dt(300), 0.30),
        Observation(_dt(360), 0.20),
        Observation(_dt(420), 0.20),
        Observation(_dt(480), 0.20),
        Observation(_dt(540), 0.20),
        Observation(_dt(590), 0.20),
    ]
    feat = compute_trajectory_features(snaps, decision_at)
    assert feat is not None
    assert feat.peak_mid == 0.80
    assert feat.subsequent_trough_mid == 0.20
    assert feat.current_mid == 0.20
    assert abs(feat.drawdown - 0.60) < 1e-6
    assert abs(feat.rebound - 0.00) < 1e-6
    assert feat.recovery_fraction == 0.0
    assert feat.is_rebound is False
    assert feat.rebound_category == "NO_REBOUND"
    assert feat.time_above_50_sec > 0.0


def test_fall_and_rebound():
    """Test price drop followed by genuine rebound."""
    decision_at = _dt(600)
    # Drops from 0.75 to 0.15, then rebounds to 0.30
    snaps = [
        Observation(_dt(0), 0.75),
        Observation(_dt(60), 0.65),
        Observation(_dt(120), 0.50),
        Observation(_dt(180), 0.30),
        Observation(_dt(240), 0.15),  # Trough
        Observation(_dt(300), 0.20),
        Observation(_dt(360), 0.25),
        Observation(_dt(420), 0.28),
        Observation(_dt(480), 0.30),
        Observation(_dt(540), 0.30),
        Observation(_dt(595), 0.30),
    ]
    feat = compute_trajectory_features(snaps, decision_at)
    assert feat is not None
    assert feat.peak_mid == 0.75
    assert feat.subsequent_trough_mid == 0.15
    assert feat.current_mid == 0.30
    assert abs(feat.drawdown - 0.60) < 1e-6
    assert abs(feat.rebound - 0.15) < 1e-6
    assert abs(feat.recovery_fraction - (0.15 / 0.60)) < 1e-6
    # drawdown >= 0.05 (0.60) and rebound >= 0.02 (0.15) -> REBOUND
    assert feat.is_rebound is True
    assert feat.rebound_category == "REBOUND"


def test_trough_before_peak_not_used():
    """
    CRITICAL CHECK: Trough BEFORE peak must NEVER be used as subsequent trough.
    e.g. 0.10 at start, then rises to 0.90, then drops to 0.70.
    Subsequent trough must be 0.70, NOT 0.10!
    """
    decision_at = _dt(600)
    snaps = [
        Observation(_dt(0), 0.10),    # Low price before peak
        Observation(_dt(60), 0.20),
        Observation(_dt(120), 0.50),
        Observation(_dt(180), 0.70),
        Observation(_dt(240), 0.90),  # Peak
        Observation(_dt(300), 0.85),
        Observation(_dt(360), 0.80),
        Observation(_dt(420), 0.75),
        Observation(_dt(480), 0.70),  # Subsequent trough
        Observation(_dt(540), 0.75),  # Slight rebound to 0.75
        Observation(_dt(595), 0.75),
    ]
    feat = compute_trajectory_features(snaps, decision_at)
    assert feat is not None
    assert feat.min_mid == 0.10  # Overall min
    assert feat.peak_mid == 0.90  # Peak
    assert feat.subsequent_trough_mid == 0.70  # Subsequent trough is 0.70, NEVER 0.10!
    assert abs(feat.drawdown - 0.20) < 1e-6
    assert abs(feat.rebound - 0.05) < 1e-6


def test_multiple_peaks_trough_not_leaked():
    """
    CRITICAL CHECK: When peak price is reached multiple times, a trough between
    the peaks must NEVER be used as subsequent trough for the latest drop.
    e.g. 0.80 at t=0, drops to 0.10 at t=60, rises back to 0.80 at t=180, then drops to 0.20 at t=300..595.
    The subsequent trough after the 0.80 peak must be 0.20, NOT 0.10!
    Rebound must be 0.00, NOT 0.10, and must NOT be classified as REBOUND.
    """
    decision_at = _dt(600)
    snaps = [
        Observation(_dt(0), 0.80),    # First peak
        Observation(_dt(60), 0.10),   # Trough between peaks
        Observation(_dt(120), 0.50),
        Observation(_dt(180), 0.80),  # Second peak
        Observation(_dt(240), 0.60),
        Observation(_dt(300), 0.40),
        Observation(_dt(360), 0.20),  # Subsequent trough after latest peak
        Observation(_dt(420), 0.20),
        Observation(_dt(480), 0.20),
        Observation(_dt(540), 0.20),
        Observation(_dt(595), 0.20),  # Current price is 0.20 (no rebound after 2nd peak)
    ]
    feat = compute_trajectory_features(snaps, decision_at)
    assert feat is not None
    assert feat.peak_mid == 0.80
    assert feat.subsequent_trough_mid == 0.20  # Must be 0.20, NEVER the 0.10 between peaks!
    assert feat.current_mid == 0.20
    assert abs(feat.drawdown - 0.60) < 1e-6
    assert abs(feat.rebound - 0.00) < 1e-6
    assert feat.recovery_fraction == 0.0
    assert feat.is_rebound is False
    assert feat.rebound_category == "NO_REBOUND"


def test_flat_series_no_movement():
    """Test series with constant price."""
    decision_at = _dt(600)
    snaps = [Observation(_dt(i * 50), 0.25) for i in range(12)]
    feat = compute_trajectory_features(snaps, decision_at)
    assert feat is not None
    assert feat.drawdown == 0.0
    assert feat.rebound == 0.0
    assert feat.recovery_fraction is None  # Undefined when no drawdown
    assert feat.is_rebound is False
    assert feat.rebound_category == "NO_REBOUND"
    assert feat.range_position == 0.5


def test_snapshot_duplication_time_invariance():
    """
    Item 14 check: Inserting duplicate snapshots at identical timestamps
    must NOT distort time_above_50_sec or span_seconds.
    """
    decision_at = _dt(600)
    snaps_clean = [
        Observation(_dt(0), 0.80),
        Observation(_dt(100), 0.80),
        Observation(_dt(200), 0.40),
        Observation(_dt(300), 0.30),
        Observation(_dt(400), 0.30),
        Observation(_dt(500), 0.20),
        Observation(_dt(590), 0.20),
    ]
    feat_clean = compute_trajectory_features(snaps_clean, decision_at)

    # Add duplicate observations
    snaps_dup = list(snaps_clean)
    snaps_dup.append(Observation(_dt(100), 0.80))
    snaps_dup.append(Observation(_dt(100), 0.80))
    snaps_dup.append(Observation(_dt(300), 0.30))
    feat_dup = compute_trajectory_features(snaps_dup, decision_at)

    assert feat_clean is not None and feat_dup is not None
    assert abs(feat_clean.time_above_50_sec - feat_dup.time_above_50_sec) < 1e-4
    assert abs(feat_clean.span_seconds - feat_dup.span_seconds) < 1e-4
    assert abs(feat_clean.drawdown - feat_dup.drawdown) < 1e-4


def test_strict_causality_future_invariance():
    """
    Self-check: observations AFTER decision_at must be completely ignored.
    """
    decision_at = _dt(500)
    snaps_pre = [Observation(_dt(i * 45), 0.30 - i * 0.01) for i in range(11)]
    feat_pre = compute_trajectory_features(snaps_pre, decision_at)

    # Add future snapshots with crazy prices
    snaps_with_future = list(snaps_pre)
    snaps_with_future.append(Observation(_dt(550), 0.99))
    snaps_with_future.append(Observation(_dt(590), 0.01))
    feat_post = compute_trajectory_features(snaps_with_future, decision_at)

    assert feat_pre is not None and feat_post is not None
    assert feat_pre.to_dict() == feat_post.to_dict()
