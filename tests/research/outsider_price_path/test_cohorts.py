"""
tests/research/outsider_price_path/test_cohorts.py

Unit tests for history quality audit and cohort assignment (Stage 3):
- FORMER_FAVORITE vs OBSERVED_ALWAYS_OUTSIDER
- Quality filter exclusions (too few obs, late start, excessive gap, stale quote)
- INSUFFICIENT_HISTORY isolation
- 2x2 matrix cell assignment
- Single-touch favorite sensitivity
"""
from datetime import datetime, timezone, timedelta
import pytest

from polyflip.research.outsider_price_path.features import (
    Observation,
    compute_trajectory_features,
)
from polyflip.research.outsider_price_path.cohorts import (
    HistoryQualityConfig,
    QualityAuditResult,
    CohortAssignment,
    audit_history_quality,
    assign_cohort,
)


def _dt(sec: int) -> datetime:
    return datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def test_former_favorite_assignment():
    """Test valid history with max_mid > 0.50 gets FORMER_FAVORITE."""
    m_start = _dt(0)
    decision_at = _dt(600)
    snaps = [
        Observation(_dt(30), 0.65),  # Peak > 0.50
        Observation(_dt(90), 0.60),
        Observation(_dt(150), 0.55),
        Observation(_dt(210), 0.45),
        Observation(_dt(270), 0.40),
        Observation(_dt(330), 0.35),
        Observation(_dt(390), 0.30),
        Observation(_dt(450), 0.25),
        Observation(_dt(510), 0.20),
        Observation(_dt(570), 0.25),  # Rebound
        Observation(_dt(595), 0.25),
    ]
    feat = compute_trajectory_features(snaps, decision_at)
    qual = audit_history_quality(feat, m_start, snaps[0].timestamp)
    assert qual.is_valid is True
    cohort = assign_cohort(feat, qual)
    assert cohort.primary_cohort == "FORMER_FAVORITE"
    assert cohort.quality_status == "VALID"
    assert cohort.matrix_2x2_cell == "FORMER_FAVORITE__REBOUND"


def test_observed_always_outsider_assignment():
    """Test valid history with max_mid <= 0.50 gets OBSERVED_ALWAYS_OUTSIDER."""
    m_start = _dt(0)
    decision_at = _dt(600)
    snaps = [Observation(_dt(30 + i * 50), 0.30) for i in range(11)]
    snaps.append(Observation(_dt(595), 0.30))
    feat = compute_trajectory_features(snaps, decision_at)
    qual = audit_history_quality(feat, m_start, snaps[0].timestamp)
    assert qual.is_valid is True
    cohort = assign_cohort(feat, qual)
    assert cohort.primary_cohort == "OBSERVED_ALWAYS_OUTSIDER"
    assert cohort.matrix_2x2_cell == "OBSERVED_ALWAYS_OUTSIDER__NO_REBOUND"


def test_quality_too_few_observations():
    """Test history with < 10 observations is excluded as INSUFFICIENT_HISTORY."""
    m_start = _dt(0)
    decision_at = _dt(600)
    snaps = [Observation(_dt(30 + i * 100), 0.30) for i in range(6)]  # only 6 obs
    feat = compute_trajectory_features(snaps, decision_at)
    qual = audit_history_quality(feat, m_start, snaps[0].timestamp)
    assert qual.is_valid is False
    assert any("TOO_FEW_OBSERVATIONS" in r for r in qual.failure_reasons)
    cohort = assign_cohort(feat, qual)
    assert cohort.primary_cohort == "INSUFFICIENT_HISTORY"
    assert cohort.matrix_2x2_cell is None


def test_quality_late_coverage_start():
    """Test first observation > 60s after market start is excluded."""
    m_start = _dt(0)
    decision_at = _dt(600)
    # First snapshot arrives at t=120s (> 60s)
    snaps = [Observation(_dt(120 + i * 40), 0.30) for i in range(12)]
    feat = compute_trajectory_features(snaps, decision_at)
    qual = audit_history_quality(feat, m_start, snaps[0].timestamp)
    assert qual.is_valid is False
    assert any("LATE_COVERAGE_START" in r for r in qual.failure_reasons)


def test_quality_excessive_gap():
    """Test gap > 90s between snapshots is excluded."""
    m_start = _dt(0)
    decision_at = _dt(600)
    snaps = [
        Observation(_dt(20), 0.30),
        Observation(_dt(60), 0.30),
        Observation(_dt(100), 0.30),
        Observation(_dt(250), 0.30),  # Gap of 150s (> 90s)
        Observation(_dt(290), 0.30),
        Observation(_dt(330), 0.30),
        Observation(_dt(370), 0.30),
        Observation(_dt(410), 0.30),
        Observation(_dt(450), 0.30),
        Observation(_dt(500), 0.30),
        Observation(_dt(590), 0.30),
    ]
    feat = compute_trajectory_features(snaps, decision_at)
    qual = audit_history_quality(feat, m_start, snaps[0].timestamp)
    assert qual.is_valid is False
    assert any("EXCESSIVE_GAP" in r for r in qual.failure_reasons)


def test_quality_stale_last_observation():
    """Test last observation > 15s before decision is excluded."""
    m_start = _dt(0)
    decision_at = _dt(600)
    # Last obs at 570s (30s before decision, exceeds 15s max age)
    snaps = [Observation(_dt(20 + i * 55), 0.30) for i in range(11)]
    feat = compute_trajectory_features(snaps, decision_at)
    qual = audit_history_quality(feat, m_start, snaps[0].timestamp)
    assert qual.is_valid is False
    assert any("STALE_LAST_OBSERVATION" in r for r in qual.failure_reasons)


def test_single_touch_favorite_flag():
    """Test single touch above 0.50 sets single_touch_favorite=True."""
    m_start = _dt(0)
    decision_at = _dt(600)
    # Exactly one observation at 0.52, others at 0.35
    snaps = [Observation(_dt(20 + i * 55), 0.35) for i in range(11)]
    snaps[2] = Observation(_dt(130), 0.52)
    snaps.append(Observation(_dt(595), 0.35))
    feat = compute_trajectory_features(snaps, decision_at)
    qual = audit_history_quality(feat, m_start, snaps[0].timestamp)
    cohort = assign_cohort(feat, qual)
    assert cohort.primary_cohort == "FORMER_FAVORITE"
    assert cohort.single_touch_favorite is True
