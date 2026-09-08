"""
tests/models/test_outsider_dataset.py

Unit tests for Item 2.4: unified decision-level dataset builder for outsider models.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from polyflip.models.outsider_dataset import (
    build_outsider_decision_rows,
    prepare_outsider_dataset_cohorts,
)


@pytest.fixture
def sample_market_snapshots() -> pd.DataFrame:
    """Generate multi-market snapshots with various time_left_min."""
    base_t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    # 6 markets
    for m_idx in range(6):
        m_id = f"market_{m_idx:02d}"
        # final outcome alternates
        outcome = "YES" if m_idx % 2 == 0 else "NO"
        # base mid price
        base_mid = 0.35 if m_idx % 2 == 0 else 0.65

        # Snapshots at minutes 1 to 14 (time_left 14 down to 1)
        for minute in range(1, 15):
            tl = 15.0 - minute
            rec_t = base_t + timedelta(hours=m_idx, minutes=minute)
            rows.append({
                "market_id": m_id,
                "recorded_at": rec_t,
                "mid_price": base_mid + 0.01 * (minute % 3),
                "poly_up_best_ask": base_mid + 0.02,
                "poly_down_best_ask": (1.0 - base_mid) + 0.02,
                "spread": 0.02,
                "time_left_min": tl,
                "final_outcome": outcome,
                "underlying_price": 50000.0 + 50.0 * minute,
                "strike_value": 50000.0,
                "underlying_lag_30s": 50000.0 + 50.0 * minute - 10.0,
                "underlying_lag_120s": 50000.0 + 50.0 * minute - 25.0,
                "sigma_1m": 0.0015,
            })

    return pd.DataFrame(rows)


def test_unique_composite_key(sample_market_snapshots: pd.DataFrame):
    """Item 2.4: (market_id, decision_at, candidate_side) is strictly unique."""
    df_dec = build_outsider_decision_rows(sample_market_snapshots)
    assert not df_dec.empty

    key_tuples = list(zip(df_dec["market_id"], df_dec["decision_at"], df_dec["candidate_side"]))
    assert len(key_tuples) == len(set(key_tuples))


def test_fixed_decision_points_extracted(sample_market_snapshots: pd.DataFrame):
    """Item 2.4: Rows correspond to fixed decision points T-10, T-5, T-2."""
    df_dec = build_outsider_decision_rows(sample_market_snapshots)
    points_found = set(df_dec["target_time_point"].unique())
    assert points_found == {10.0, 5.0, 2.0}


def test_market_level_fold_grouping_prevents_leakage(sample_market_snapshots: pd.DataFrame):
    """Item 2.4: All decision points of the same market_id must be in the exact same fold."""
    cohorts = prepare_outsider_dataset_cohorts(sample_market_snapshots, n_splits=3)
    df_b = cohorts.df_full_b
    assert not df_b.empty

    for m_id, grp in df_b.groupby("market_id"):
        # Exactly one fold per market
        assert grp["fold"].nunique() == 1, f"Market {m_id} split across multiple folds!"


def test_fair_cohort_alignment(sample_market_snapshots: pd.DataFrame):
    """Item 2.4: df_full_b contains complete B features; df_broad_a covers all A."""
    cohorts = prepare_outsider_dataset_cohorts(sample_market_snapshots)
    assert len(cohorts.df_full_b) <= len(cohorts.df_broad_a)
    assert cohorts.df_full_b["has_z_ref"].all()
    assert cohorts.df_full_b["has_ret_30s_ref"].all()
    assert cohorts.df_full_b["has_ret_120s_ref"].all()


def test_mid_05_exclusion():
    """Item 2.4: mid_price == 0.5 must be strictly excluded."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([{
        "market_id": "m_neutral",
        "recorded_at": t0,
        "mid_price": 0.50,
        "spread": 0.02,
        "time_left_min": 10.0,
        "final_outcome": "YES",
    }])
    df_dec = build_outsider_decision_rows(df)
    assert df_dec.empty
