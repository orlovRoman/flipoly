"""
polyflip/research/outsider_price_path/dataset.py

Protocol specification, ledger building, and dataset generation (Stages 1, 2, 3, 4).
Adheres strictly to research constraints:
- Immutable protocol with deterministic SHA-256 hash (Item 3).
- Strict separation between historical exploratory period and holdout (Item 4).
- Source mapping for markets, token IDs, quotes, timestamps, outcomes, and fees (Item 5).
- Independent market registry inclusion regardless of funnel (Item 7).
- Outcome verification from actual contract resolution, not Binance proxy (Item 8).
- Causal decision snapshot at T-5 with max 15s delay (Item 9).
- Real quote outsider determination without substituting 1 - YES (Item 10).
- Single unified opportunity ledger (Item 18).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, Mapping, Any, Optional
import pandas as pd

from polyflip.research.outsider_price_path.features import (
    Observation,
    TrajectoryFeatures,
    compute_trajectory_features,
)
from polyflip.research.outsider_price_path.cohorts import (
    HistoryQualityConfig,
    QualityAuditResult,
    CohortAssignment,
    audit_history_quality,
    assign_cohort,
)
from polyflip.research.outsider_price_path.evaluation import OpportunityEconomics
from polyflip.research.regime_features import classify_local_regime


ASK_BINS = [
    (0.01, 0.05, "[0.01,0.05)"),
    (0.05, 0.10, "[0.05,0.10)"),
    (0.10, 0.15, "[0.10,0.15)"),
    (0.15, 0.20, "[0.15,0.20)"),
    (0.20, 0.25, "[0.20,0.25)"),
    (0.25, 0.30, "[0.25,0.30)"),
    (0.30, 0.35, "[0.30,0.35)"),
    (0.35, 0.40, "[0.35,0.40]"),
]


def assign_ask_bin(ask: Optional[float]) -> str:
    """Classifies ask into standardized 0.05 bins."""
    if ask is None or not math.isfinite(ask):
        return "MISSING"
    if ask < 0.01:
        return "BELOW_0.01"
    if ask > 0.40:
        return "ABOVE_0.40"
    for lo, hi, label in ASK_BINS:
        if lo <= ask < hi or (label == "[0.35,0.40]" and ask == hi):
            return label
    return "UNKNOWN"


@dataclass(frozen=True)
class ResearchProtocol:
    """Immutable protocol specification for outsider price path study."""
    protocol_id: str = "OUTSIDER_PRICE_PATH_V1"
    version: int = 1
    assets: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP", "DOGE")
    contract_horizon_min: int = 15
    decision_target_min: float = 5.0  # T-5
    max_decision_delay_sec: float = 15.0
    ask_min: float = 0.01
    ask_max: float = 0.40
    budget_usdc: float = 1.00
    baseline_taker_fee_rate: float = 0.002  # 0.2%
    scenario_taker_fee_rate: float = 0.001  # 0.1%
    parity_tolerance: float = 1e-4

    # History quality criteria
    min_observations: int = 10
    max_start_delay_sec: float = 60.0
    max_gap_sec: float = 90.0
    max_last_obs_age_sec: float = 15.0

    # Orthogonal rebound criteria
    rebound_drawdown_thresh: float = 0.05
    rebound_rebound_thresh: float = 0.02

    # Period split (Item 4)
    exploratory_start_utc: str = "2026-06-25T00:00:00+00:00"
    exploratory_end_utc: str = "2026-08-31T23:59:59+00:00"
    holdout_start_utc: str = "2026-09-01T00:00:00+00:00"
    holdout_end_utc: str = "2026-09-09T23:59:59+00:00"

    def canonical_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return dict(sorted(data.items()))

    @property
    def protocol_hash(self) -> str:
        serialized = json.dumps(self.canonical_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


DEFAULT_PROTOCOL = ResearchProtocol()


@dataclass
class MarketMetadata:
    market_id: str
    asset: str
    expiry: datetime
    start_time: datetime
    final_outcome: str  # "YES", "NO", "INVALID", "PENDING"
    in_funnel: bool = False
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None


@dataclass
class OpportunityRecord:
    """One single row in the unified opportunity ledger (Item 18)."""
    opportunity_id: str
    market_id: str
    asset: str
    side: str  # "UP"
    calendar_date: str
    calendar_week: str
    decision_at: str
    delay_sec: float
    ask: float
    bid: Optional[float]
    mid: float
    spread: Optional[float]
    ask_bin: str
    final_outcome: str
    target: int  # 1 win, 0 loss
    in_funnel: bool

    # Quality & Cohorts
    quality_status: str
    quality_reasons: str
    primary_cohort: str
    single_touch_favorite: bool
    rebound_category: str
    matrix_2x2_cell: Optional[str]

    # CT classifier
    ct_state: str

    # Trajectory features
    initial_mid: Optional[float]
    peak_mid: Optional[float]
    subsequent_trough_mid: Optional[float]
    drawdown: Optional[float]
    rebound: Optional[float]
    recovery_fraction: Optional[float]
    drop_from_peak: Optional[float]
    drop_duration_sec: Optional[float]
    range_position: Optional[float]
    time_above_50_sec: Optional[float]
    fraction_time_above_50: Optional[float]
    n_observations: int

    # Economics
    budget_usdc: float
    shares: float
    gross_pnl: float
    fee_02pct: float
    net_pnl_02pct: float
    fee_01pct: float
    net_pnl_01pct: float

    # Selection status
    selection_status: str  # "OK", "PRICE_FILTER", "DELAY_TOO_LARGE", "PARITY", "MISSING_QUOTE", "UNRESOLVED"
    skip_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def process_single_market(
    meta: MarketMetadata,
    raw_snapshots: Sequence[dict[str, Any]],
    protocol: ResearchProtocol = DEFAULT_PROTOCOL,
) -> OpportunityRecord:
    """
    Evaluates one single market causally:
    1. Finds T-5 decision snapshot strictly after boundary with delay <= 15s.
    2. Identifies outsider side by observed mid price.
    3. Checks ask price in [0.01, 0.40].
    4. Gathers pre-decision history and computes features.
    5. Audits quality and assigns cohort + 2x2 cell.
    6. Runs CT classifier.
    7. Computes line-item economics.
    """
    decision_target = meta.expiry - timedelta(minutes=protocol.decision_target_min)
    if decision_target.tzinfo is None:
        decision_target = decision_target.replace(tzinfo=timezone.utc)

    # Convert snapshot timestamps and sort
    parsed_snaps: list[dict[str, Any]] = []
    for s in raw_snapshots:
        try:
            rec = datetime.fromisoformat(str(s["recorded_at"]))
            if rec.tzinfo is None:
                rec = rec.replace(tzinfo=timezone.utc)
            s_copy = dict(s)
            s_copy["_rec"] = rec
            parsed_snaps.append(s_copy)
        except Exception:
            continue

    parsed_snaps.sort(key=lambda x: x["_rec"])

    # Base opportunity metadata
    opp_id = f"{meta.market_id}_{decision_target.isoformat()}"
    cal_date = decision_target.date().isoformat()
    cal_week = f"{decision_target.year}-W{decision_target.isocalendar()[1]:02d}"

    def _make_skip_record(status: str, reason: str, ask_val: float = 0.0, mid_val: float = 0.5) -> OpportunityRecord:
        return OpportunityRecord(
            opportunity_id=opp_id,
            market_id=meta.market_id,
            asset=meta.asset,
            side="UP",
            calendar_date=cal_date,
            calendar_week=cal_week,
            decision_at=decision_target.isoformat(),
            delay_sec=0.0,
            ask=round(ask_val, 4),
            bid=None,
            mid=round(mid_val, 4),
            spread=None,
            ask_bin=assign_ask_bin(ask_val if ask_val > 0 else None),
            final_outcome=meta.final_outcome,
            target=1 if meta.final_outcome == "YES" else 0,
            in_funnel=meta.in_funnel,
            quality_status="INSUFFICIENT_HISTORY",
            quality_reasons=reason,
            primary_cohort="INSUFFICIENT_HISTORY",
            single_touch_favorite=False,
            rebound_category="UNCERTAIN",
            matrix_2x2_cell=None,
            ct_state="INSUFFICIENT_HISTORY",
            initial_mid=None,
            peak_mid=None,
            subsequent_trough_mid=None,
            drawdown=None,
            rebound=None,
            recovery_fraction=None,
            drop_from_peak=None,
            drop_duration_sec=None,
            range_position=None,
            time_above_50_sec=None,
            fraction_time_above_50=None,
            n_observations=0,
            budget_usdc=protocol.budget_usdc,
            shares=0.0,
            gross_pnl=0.0,
            fee_02pct=0.0,
            net_pnl_02pct=0.0,
            fee_01pct=0.0,
            net_pnl_01pct=0.0,
            selection_status=status,
            skip_reason=reason,
        )

    # Validate market resolution
    if meta.final_outcome not in ("YES", "NO"):
        return _make_skip_record("UNRESOLVED", f"OUTCOME_{meta.final_outcome}")

    # Find earliest observation at or after decision_target (Item 9)
    after_boundary = [s for s in parsed_snaps if s["_rec"] >= decision_target]
    if not after_boundary:
        return _make_skip_record("NO_SNAPSHOT", "NO_OBSERVATION_AFTER_BOUNDARY")

    decision_snap = after_boundary[0]
    decision_at = decision_snap["_rec"]
    delay_sec = (decision_at - decision_target).total_seconds()

    if delay_sec > protocol.max_decision_delay_sec:
        return _make_skip_record("DELAY_TOO_LARGE", f"DELAY_{round(delay_sec, 1)}S_EXCEEDS_{protocol.max_decision_delay_sec}S")

    # Check quotes at decision
    raw_ask = decision_snap.get("best_ask")
    raw_bid = decision_snap.get("best_bid")
    raw_mid = decision_snap.get("mid_price")
    raw_spread = decision_snap.get("spread")

    try:
        ask = float(raw_ask) if raw_ask not in (None, "") else None
        bid = float(raw_bid) if raw_bid not in (None, "") else None
        mid = float(raw_mid) if raw_mid not in (None, "") else None
        spread = float(raw_spread) if raw_spread not in (None, "") else None
    except (ValueError, TypeError):
        return _make_skip_record("MISSING_QUOTE", "NON_FINITE_QUOTE_AT_DECISION")

    if mid is None or ask is None:
        return _make_skip_record("MISSING_QUOTE", "NULL_MID_OR_ASK_AT_DECISION")

    # Symmetric outsider parity check (Item 10)
    diff_from_parity = mid - 0.50
    if abs(diff_from_parity) <= protocol.parity_tolerance:
        return _make_skip_record("PARITY", f"PARITY_DIFF_{round(diff_from_parity, 5)}")

    if mid > 0.50:
        # Chosen outsider would be DOWN (NO). As per Item 10, missing NO is NOT replaced by 1 - YES.
        return _make_skip_record("NOT_YES_OUTSIDER", "OUTSIDER_IS_DOWN_NO_QUOTE_OBSERVED", ask_val=ask, mid_val=mid)

    # Chosen outsider is YES (UP)
    side = "UP"
    target = 1 if meta.final_outcome == "YES" else 0

    # Price range filter [0.01, 0.40]
    if not (protocol.ask_min <= ask <= protocol.ask_max):
        return _make_skip_record("PRICE_FILTER", f"ASK_{round(ask, 4)}_OUTSIDE_[{protocol.ask_min},{protocol.ask_max}]", ask_val=ask, mid_val=mid)

    # Causal pre-decision history strictly: market_start_at <= t <= decision_at
    pre_snaps: list[Observation] = []
    for s in parsed_snaps:
        t = s["_rec"]
        if meta.start_time <= t <= decision_at:
            m_val = s.get("mid_price")
            if m_val not in (None, ""):
                try:
                    f_mid = float(m_val)
                    if math.isfinite(f_mid):
                        b_val = float(s["best_bid"]) if s.get("best_bid") not in (None, "") else None
                        a_val = float(s["best_ask"]) if s.get("best_ask") not in (None, "") else None
                        sp_val = float(s["spread"]) if s.get("spread") not in (None, "") else None
                        pre_snaps.append(Observation(
                            timestamp=t,
                            mid_price=f_mid,
                            best_bid=b_val,
                            best_ask=a_val,
                            spread=sp_val,
                        ))
                except (ValueError, TypeError):
                    continue

    # Trajectory features (Stage 3)
    features = compute_trajectory_features(
        observations=pre_snaps,
        decision_at=decision_at,
        rebound_drawdown_thresh=protocol.rebound_drawdown_thresh,
        rebound_rebound_thresh=protocol.rebound_rebound_thresh,
    )

    # History quality check (Item 11)
    first_obs_time = pre_snaps[0].timestamp if pre_snaps else None
    q_cfg = HistoryQualityConfig(
        min_observations=protocol.min_observations,
        max_start_delay_sec=protocol.max_start_delay_sec,
        max_gap_sec=protocol.max_gap_sec,
        max_last_obs_age_sec=protocol.max_last_obs_age_sec,
    )
    quality = audit_history_quality(
        features=features,
        market_start_at=meta.start_time,
        first_obs_time=first_obs_time,
        config=q_cfg,
    )

    # Cohort assignment (Item 15 & 16)
    cohort = assign_cohort(features, quality)

    # CT calculation (Item 17)
    ct_state = "INSUFFICIENT_HISTORY"
    if features and features.n_observations >= 3:
        try:
            mid_series = [o.mid_price for o in pre_snaps]
            regime = classify_local_regime(mid_series, min_observations=3)
            ct_state = str(regime.get("state", "UNCERTAIN"))
        except Exception:
            ct_state = "CALCULATION_ERROR"

    # Line-item economics (Item 19)
    econ = OpportunityEconomics.compute(ask=ask, target=target, budget=protocol.budget_usdc)

    return OpportunityRecord(
        opportunity_id=opp_id,
        market_id=meta.market_id,
        asset=meta.asset,
        side=side,
        calendar_date=cal_date,
        calendar_week=cal_week,
        decision_at=decision_at.isoformat(),
        delay_sec=round(delay_sec, 2),
        ask=round(ask, 4),
        bid=round(bid, 4) if bid is not None else None,
        mid=round(mid, 4),
        spread=round(spread, 4) if spread is not None else None,
        ask_bin=assign_ask_bin(ask),
        final_outcome=meta.final_outcome,
        target=target,
        in_funnel=meta.in_funnel,
        quality_status=cohort.quality_status,
        quality_reasons="; ".join(cohort.quality_reasons),
        primary_cohort=cohort.primary_cohort,
        single_touch_favorite=cohort.single_touch_favorite,
        rebound_category=cohort.rebound_category,
        matrix_2x2_cell=cohort.matrix_2x2_cell,
        ct_state=ct_state,
        initial_mid=features.initial_mid if features else None,
        peak_mid=features.peak_mid if features else None,
        subsequent_trough_mid=features.subsequent_trough_mid if features else None,
        drawdown=features.drawdown if features else None,
        rebound=features.rebound if features else None,
        recovery_fraction=features.recovery_fraction if features else None,
        drop_from_peak=features.drop_from_peak if features else None,
        drop_duration_sec=features.drop_duration_sec if features else None,
        range_position=features.range_position if features else None,
        time_above_50_sec=features.time_above_50_sec if features else None,
        fraction_time_above_50=features.fraction_time_above_50 if features else None,
        n_observations=features.n_observations if features else 0,
        budget_usdc=econ.budget_usdc,
        shares=econ.shares,
        gross_pnl=econ.gross_pnl,
        fee_02pct=econ.fee_02pct,
        net_pnl_02pct=econ.net_pnl_02pct,
        fee_01pct=econ.fee_01pct,
        net_pnl_01pct=econ.net_pnl_01pct,
        selection_status="OK",
        skip_reason="",
    )
