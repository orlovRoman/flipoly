"""
polyflip/research/outsider_price_path/cohorts.py

Cohort definitions, history quality audits, and 2x2 trajectory matrix (Stage 3).
Mutually exclusive cohorts:
- FORMER_FAVORITE: observed > 0.50 in pre-entry window with valid history.
- OBSERVED_ALWAYS_OUTSIDER: never observed > 0.50 in pre-entry window with valid history.
- INSUFFICIENT_HISTORY: fails fixed data quality requirements.

Orthogonal 2x2 matrix:
(FORMER_FAVORITE vs OBSERVED_ALWAYS_OUTSIDER) x (REBOUND vs NO_REBOUND).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Any

from polyflip.research.outsider_price_path.features import TrajectoryFeatures


@dataclass(frozen=True)
class HistoryQualityConfig:
    """Pre-registered data quality thresholds."""
    min_observations: int = 10
    max_start_delay_sec: float = 60.0
    max_gap_sec: float = 90.0
    max_last_obs_age_sec: float = 15.0


@dataclass(frozen=True)
class QualityAuditResult:
    is_valid: bool
    status: str  # "VALID" or "INSUFFICIENT_HISTORY"
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "failure_reasons": list(self.failure_reasons),
        }


def audit_history_quality(
    features: Optional[TrajectoryFeatures],
    market_start_at: Optional[datetime],
    first_obs_time: Optional[datetime],
    config: Optional[HistoryQualityConfig] = None,
) -> QualityAuditResult:
    """
    Evaluates pre-decision history against pre-registered quality criteria (Item 11).
    Does NOT depend on PnL or trade profitability.
    """
    cfg = config or HistoryQualityConfig()
    reasons: list[str] = []

    if features is None:
        return QualityAuditResult(
            is_valid=False,
            status="INSUFFICIENT_HISTORY",
            failure_reasons=("NO_OBSERVATIONS",),
        )

    # 1. Observation count
    if features.n_observations < cfg.min_observations:
        reasons.append(f"TOO_FEW_OBSERVATIONS: {features.n_observations} < {cfg.min_observations}")

    # 2. Coverage start delay relative to market_start_at
    if market_start_at is not None and first_obs_time is not None:
        m_start = market_start_at
        f_time = first_obs_time
        if m_start.tzinfo is None and f_time.tzinfo is not None:
            m_start = m_start.replace(tzinfo=f_time.tzinfo)
        elif m_start.tzinfo is not None and f_time.tzinfo is None:
            f_time = f_time.replace(tzinfo=m_start.tzinfo)

        start_delay = (f_time - m_start).total_seconds()
        if start_delay > cfg.max_start_delay_sec:
            reasons.append(f"LATE_COVERAGE_START: {round(start_delay, 1)}s > {cfg.max_start_delay_sec}s")

    # 3. Maximum gap between observations
    if features.max_gap_seconds > cfg.max_gap_sec:
        reasons.append(f"EXCESSIVE_GAP: {round(features.max_gap_seconds, 1)}s > {cfg.max_gap_sec}s")

    # 4. Age of most recent observation at decision moment
    if features.last_obs_age_seconds > cfg.max_last_obs_age_sec:
        reasons.append(f"STALE_LAST_OBSERVATION: {round(features.last_obs_age_seconds, 1)}s > {cfg.max_last_obs_age_sec}s")

    is_valid = len(reasons) == 0
    return QualityAuditResult(
        is_valid=is_valid,
        status="VALID" if is_valid else "INSUFFICIENT_HISTORY",
        failure_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class CohortAssignment:
    """Mutually exclusive primary cohort and orthogonal rebound cell."""
    primary_cohort: str  # "FORMER_FAVORITE", "OBSERVED_ALWAYS_OUTSIDER", or "INSUFFICIENT_HISTORY"
    quality_status: str  # "VALID" or "INSUFFICIENT_HISTORY"
    quality_reasons: tuple[str, ...]
    rebound_category: str  # "REBOUND", "NO_REBOUND", or "UNCERTAIN"
    matrix_2x2_cell: Optional[str]  # e.g. "FORMER_FAVORITE__REBOUND" or None if insufficient history
    single_touch_favorite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_cohort": self.primary_cohort,
            "quality_status": self.quality_status,
            "quality_reasons": list(self.quality_reasons),
            "rebound_category": self.rebound_category,
            "matrix_2x2_cell": self.matrix_2x2_cell,
            "single_touch_favorite": self.single_touch_favorite,
        }


def assign_cohort(
    features: Optional[TrajectoryFeatures],
    quality: QualityAuditResult,
) -> CohortAssignment:
    """
    Assigns mutually exclusive primary cohort and orthogonal 2x2 matrix cell (Items 15 & 16).
    """
    if not quality.is_valid or features is None:
        return CohortAssignment(
            primary_cohort="INSUFFICIENT_HISTORY",
            quality_status="INSUFFICIENT_HISTORY",
            quality_reasons=quality.failure_reasons,
            rebound_category="UNCERTAIN",
            matrix_2x2_cell=None,
            single_touch_favorite=False,
        )

    # Primary cohort: FORMER_FAVORITE vs OBSERVED_ALWAYS_OUTSIDER
    if features.max_mid > 0.50:
        primary_cohort = "FORMER_FAVORITE"
        single_touch = features.single_touch_above_50
    else:
        primary_cohort = "OBSERVED_ALWAYS_OUTSIDER"
        single_touch = False

    rebound_cat = features.rebound_category
    matrix_cell = f"{primary_cohort}__{rebound_cat}"

    return CohortAssignment(
        primary_cohort=primary_cohort,
        quality_status="VALID",
        quality_reasons=(),
        rebound_category=rebound_cat,
        matrix_2x2_cell=matrix_cell,
        single_touch_favorite=single_touch,
    )
