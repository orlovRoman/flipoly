"""Comprehensive test suite for LogReg Close-Time OOT windows and invariants."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.logreg_polymarket_backtest import (
    CLOSE_TIME_PRIORITY_COLUMNS,
    compute_logreg_polymarket_backtest,
    resolve_market_close_time,
    resolve_market_close_time_series,
    split_chronological_oot_windows,
)
from polyflip.crypto.oof_artifact import (
    deserialize_oof_artifact,
    serialize_oof_artifact,
)
from polyflip.crypto.polymarket_backtest import METRICS_SCHEMA_VERSION
from polyflip.models.temporal_validation import grouped_walk_forward_folds
from polyflip.scripts.audit_logreg_models import audit_single_model
from polyflip.scripts.retrain_logreg_candidates import train_and_eval_candidate


def test_close_time_fallback_chain_priority():
    """Verify exact fallback chain: market_close_at -> resolved_at -> end_time_est -> market_start."""
    t_close = "2026-08-01T12:00:00Z"
    t_resolved = "2026-08-01T12:05:00Z"
    t_end_est = "2026-08-01T12:15:00Z"
    t_start = "2026-08-01T11:45:00Z"

    # 1. All 4 present -> market_close_at wins
    d1 = {
        "market_close_at": t_close,
        "resolved_at": t_resolved,
        "end_time_est": t_end_est,
        "market_start": t_start,
    }
    assert resolve_market_close_time(d1) == pd.Timestamp(t_close)

    # 2. market_close_at is None -> resolved_at wins
    d2 = {
        "market_close_at": None,
        "resolved_at": t_resolved,
        "end_time_est": t_end_est,
        "market_start": t_start,
    }
    assert resolve_market_close_time(d2) == pd.Timestamp(t_resolved)

    # 3. market_close_at & resolved_at are None -> end_time_est wins
    d3 = {
        "market_close_at": None,
        "resolved_at": None,
        "end_time_est": t_end_est,
        "market_start": t_start,
    }
    assert resolve_market_close_time(d3) == pd.Timestamp(t_end_est)

    # 4. Only market_start present -> market_start wins
    d4 = {
        "market_close_at": None,
        "resolved_at": None,
        "end_time_est": None,
        "market_start": t_start,
    }
    assert resolve_market_close_time(d4) == pd.Timestamp(t_start)

    # 5. All None / missing -> None
    d5 = {
        "market_close_at": None,
        "resolved_at": None,
        "end_time_est": None,
        "market_start": None,
    }
    assert resolve_market_close_time(d5) is None


def test_close_time_fallback_chain_series():
    """Verify vectorized resolve_market_close_time_series matches row-level priority."""
    df = pd.DataFrame([
        {
            "market_id": "m1",
            "market_close_at": "2026-08-01T10:00:00Z",
            "resolved_at": "2026-08-01T10:05:00Z",
            "end_time_est": "2026-08-01T10:15:00Z",
            "market_start": "2026-08-01T09:45:00Z",
        },
        {
            "market_id": "m2",
            "market_close_at": None,
            "resolved_at": "2026-08-01T11:05:00Z",
            "end_time_est": "2026-08-01T11:15:00Z",
            "market_start": "2026-08-01T10:45:00Z",
        },
        {
            "market_id": "m3",
            "market_close_at": None,
            "resolved_at": None,
            "end_time_est": "2026-08-01T12:15:00Z",
            "market_start": "2026-08-01T11:45:00Z",
        },
        {
            "market_id": "m4",
            "market_close_at": None,
            "resolved_at": None,
            "end_time_est": None,
            "market_start": "2026-08-01T12:45:00Z",
        },
        {
            "market_id": "m5",
            "market_close_at": None,
            "resolved_at": None,
            "end_time_est": None,
            "market_start": None,
        },
    ])
    series = resolve_market_close_time_series(df)
    assert series[0] == pd.Timestamp("2026-08-01T10:00:00Z")
    assert series[1] == pd.Timestamp("2026-08-01T11:05:00Z")
    assert series[2] == pd.Timestamp("2026-08-01T12:15:00Z")
    assert series[3] == pd.Timestamp("2026-08-01T12:45:00Z")
    assert pd.isna(series[4])


def test_no_snapshot_time_fallback_when_close_time_fields_missing():
    """Regression test: presence of recorded_at without any of the 4 close-time fields NEVER creates OOT windows."""
    frame = pd.DataFrame([
        {
            "market_id": "m1",
            "recorded_at": "2026-08-01T00:00:00Z",
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
            "market_close_at": None,
            "resolved_at": None,
            "end_time_est": None,
            "market_start": None,
        },
        {
            "market_id": "m2",
            "recorded_at": "2026-08-01T01:00:00Z",
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
            "market_close_at": None,
            "resolved_at": None,
            "end_time_est": None,
            "market_start": None,
        },
        {
            "market_id": "m3",
            "recorded_at": "2026-08-01T02:00:00Z",
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
            "market_close_at": None,
            "resolved_at": None,
            "end_time_est": None,
            "market_start": None,
        },
    ])
    quotes = frame.copy()
    oof_scores = np.array([0.90, 0.90, 0.90])

    windows = split_chronological_oot_windows(
        frame,
        oof_scores,
        quotes,
        strategy_branch="COMBINED",
    )

    # Must be completely EMPTY: recorded_at is NOT a fallback for close_time
    assert windows["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert windows["median_pnl"] is None
    assert windows["non_negative_windows_count"] == 0
    for w_name in ("T1", "T2", "T3"):
        w = windows[w_name]
        assert w["status"] == "EMPTY"
        assert w["start_close_time"] is None
        assert w["end_close_time"] is None
        assert w["unique_market_count"] == 0
        assert w["snapshot_count"] == 0
        assert w["trade_count"] == 0
        assert w["net_profit"] is None


def test_train_and_eval_candidate_blocks_when_close_time_missing():
    """Regression test: train_and_eval_candidate returns None when close_time fields are missing (no snapshot fallback)."""
    rows = []
    base_time = pd.Timestamp("2026-08-01T00:00:00Z")
    for m_idx in range(10):
        m_id = f"market_{m_idx:02d}"
        for s_idx in range(5):
            rows.append({
                "market_id": m_id,
                "asset": "BTC",
                "recorded_at": base_time + timedelta(hours=m_idx, minutes=s_idx),
                "time_left_min": 10.0 - s_idx,
                "mid_price": 0.40 if m_idx % 2 == 0 else 0.60,
                "spread": 0.02,
                "best_bid": 0.39,
                "best_ask": 0.41,
                "volume_5min": 100.0,
                "price_velocity": 0.1,
                "hour_of_day": 12,
                "target": m_idx % 2,
                "final_outcome": "YES" if m_idx % 2 == 1 else "NO",
                "yes_price": 0.40,
                "no_price": 0.60,
                # Explicitly missing all 4 close-time fields
                "market_close_at": None,
                "resolved_at": None,
                "end_time_est": None,
                "market_start": None,
            })
    df_missing_close_time = pd.DataFrame(rows)

    # Candidate evaluation must be blocked (return None), never fall back to recorded_at
    result = train_and_eval_candidate(
        df_missing_close_time,
        variant="BASE",
        C=1.0,
        class_weight=None,
        sample_weight_mode="uniform",
        sample_weight_tau=0.0,
        calibration_method="RAW",
    )
    assert result is None


def test_missing_close_time_markets_are_excluded_from_oot_partitions():
    """Verify that markets lacking close-time fields are excluded while valid markets are partitioned."""
    valid_close_1 = pd.Timestamp("2026-08-01T10:00:00Z")
    valid_close_2 = pd.Timestamp("2026-08-01T11:00:00Z")
    valid_close_3 = pd.Timestamp("2026-08-01T12:00:00Z")

    frame = pd.DataFrame([
        # Valid market 1
        {
            "market_id": "valid_1",
            "recorded_at": "2026-08-01T00:00:00Z",
            "market_close_at": valid_close_1,
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        },
        # Invalid market without close time
        {
            "market_id": "invalid_market",
            "recorded_at": "2026-08-01T00:30:00Z",
            "market_close_at": None,
            "resolved_at": None,
            "end_time_est": None,
            "market_start": None,
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        },
        # Valid market 2
        {
            "market_id": "valid_2",
            "recorded_at": "2026-08-01T01:00:00Z",
            "resolved_at": valid_close_2,
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        },
        # Valid market 3
        {
            "market_id": "valid_3",
            "recorded_at": "2026-08-01T02:00:00Z",
            "end_time_est": valid_close_3,
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        },
    ])
    quotes = frame.copy()
    oof_scores = np.array([0.90, 0.90, 0.90, 0.90])

    windows = split_chronological_oot_windows(
        frame,
        oof_scores,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.01,
        cost_buffer=0.0,
    )

    # 3 valid markets partitioned into T1, T2, T3 (1 each)
    assert windows["T1"]["unique_market_count"] == 1
    assert windows["T1"]["start_close_time"] == valid_close_1.isoformat()
    assert windows["T2"]["unique_market_count"] == 1
    assert windows["T2"]["start_close_time"] == valid_close_2.isoformat()
    assert windows["T3"]["unique_market_count"] == 1
    assert windows["T3"]["start_close_time"] == valid_close_3.isoformat()


def test_oot_windows_no_market_id_overlap():
    """Verify that T1, T2, T3 partitions have zero market_id overlap and group all snapshots."""
    rows = []
    base_time = pd.Timestamp("2026-08-01T00:00:00Z")
    for m_idx in range(9):
        m_id = f"market_{m_idx:02d}"
        m_close = base_time + timedelta(hours=m_idx)
        for s_idx in range(3):
            s_rec = m_close - timedelta(minutes=15 - s_idx * 5)
            rows.append({
                "market_id": m_id,
                "recorded_at": s_rec,
                "market_close_at": m_close,
                "resolved_at": m_close + timedelta(minutes=2),
                "end_time_est": m_close,
                "market_start": m_close - timedelta(minutes=15),
                "mid_price": 0.30,
                "best_bid": 0.29,
                "best_ask": 0.31,
                "spread": 0.02,
                "final_outcome": "YES",
                "target": 1,
            })
    frame = pd.DataFrame(rows)
    quotes = frame.drop_duplicates("market_id")[["market_id", "recorded_at", "mid_price", "best_bid", "best_ask", "final_outcome"]].copy()
    oof_scores = np.full(len(frame), 0.90)

    windows = split_chronological_oot_windows(
        frame,
        oof_scores,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.01,
        cost_buffer=0.0,
    )

    assert windows["metrics_schema_version"] == METRICS_SCHEMA_VERSION

    t1 = windows["T1"]
    t2 = windows["T2"]
    t3 = windows["T3"]

    assert t1["unique_market_count"] == 3
    assert t2["unique_market_count"] == 3
    assert t3["unique_market_count"] == 3

    assert t1["snapshot_count"] == 9
    assert t2["snapshot_count"] == 9
    assert t3["snapshot_count"] == 9

    assert t1["trade_count"] == 3
    assert t2["trade_count"] == 3
    assert t3["trade_count"] == 3

    for label, w in (("T1", t1), ("T2", t2), ("T3", t3)):
        assert w["start_close_time"] is not None
        assert w["end_close_time"] is not None
        assert isinstance(w["unique_market_count"], int)
        assert isinstance(w["snapshot_count"], int)
        assert isinstance(w["trade_count"], int)
        assert isinstance(w["coverage"], float)
        assert isinstance(w["net_profit"], float)
        assert isinstance(w["roi_pct"], float)
        assert isinstance(w["max_drawdown"], float)


def test_oot_markets_ordered_strictly_by_close_time():
    """Verify that markets are partitioned sequentially by market close time."""
    times = [
        pd.Timestamp("2026-08-01T10:00:00Z"),
        pd.Timestamp("2026-08-01T11:00:00Z"),
        pd.Timestamp("2026-08-01T12:00:00Z"),
        pd.Timestamp("2026-08-01T13:00:00Z"),
        pd.Timestamp("2026-08-01T14:00:00Z"),
        pd.Timestamp("2026-08-01T15:00:00Z"),
    ]
    rows = []
    for i, t in enumerate(times):
        rows.append({
            "market_id": f"m{i}",
            "recorded_at": t - timedelta(minutes=5),
            "market_close_at": t,
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        })
    np.random.seed(42)
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    quotes = frame[["market_id", "recorded_at", "mid_price", "best_bid", "best_ask", "final_outcome"]].copy()
    oof_scores = np.full(len(frame), 0.90)

    windows = split_chronological_oot_windows(
        frame,
        oof_scores,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.01,
        cost_buffer=0.0,
    )

    t1 = windows["T1"]
    t2 = windows["T2"]
    t3 = windows["T3"]

    assert pd.to_datetime(t1["start_close_time"]) == times[0]
    assert pd.to_datetime(t1["end_close_time"]) == times[1]

    assert pd.to_datetime(t2["start_close_time"]) == times[2]
    assert pd.to_datetime(t2["end_close_time"]) == times[3]

    assert pd.to_datetime(t3["start_close_time"]) == times[4]
    assert pd.to_datetime(t3["end_close_time"]) == times[5]

    assert pd.to_datetime(t1["end_close_time"]) <= pd.to_datetime(t2["start_close_time"])
    assert pd.to_datetime(t2["end_close_time"]) <= pd.to_datetime(t3["start_close_time"])


def test_oot_window_date_determined_by_close_time_not_snapshot_time():
    """Verify that window assignment is governed strictly by market close time, NOT snapshot time."""
    # Counter-example construction:
    # Market A: early snapshot at 06:00, but late close time at 18:00
    # Market B: late snapshot at 09:00, but early close time at 10:00
    # Market C: mid snapshot at 08:00, mid close time at 14:00
    frame = pd.DataFrame([
        {
            "market_id": "market_A",  # Close 18:00 -> T3
            "recorded_at": "2026-08-01T06:00:00Z",
            "market_close_at": "2026-08-01T18:00:00Z",
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        },
        {
            "market_id": "market_B",  # Close 10:00 -> T1
            "recorded_at": "2026-08-01T09:00:00Z",
            "market_close_at": "2026-08-01T10:00:00Z",
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        },
        {
            "market_id": "market_C",  # Close 14:00 -> T2
            "recorded_at": "2026-08-01T08:00:00Z",
            "market_close_at": "2026-08-01T14:00:00Z",
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        },
    ])
    quotes = frame.copy()
    oof_scores = np.full(len(frame), 0.90)

    windows = split_chronological_oot_windows(
        frame,
        oof_scores,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.01,
        cost_buffer=0.0,
    )

    # T1 must be market_B (close 10:00)
    assert windows["T1"]["start_close_time"] == "2026-08-01T10:00:00+00:00"
    assert windows["T1"]["end_close_time"] == "2026-08-01T10:00:00+00:00"
    assert windows["T1"]["unique_market_count"] == 1

    # T2 must be market_C (close 14:00)
    assert windows["T2"]["start_close_time"] == "2026-08-01T14:00:00+00:00"
    assert windows["T2"]["end_close_time"] == "2026-08-01T14:00:00+00:00"
    assert windows["T2"]["unique_market_count"] == 1

    # T3 must be market_A (close 18:00)
    assert windows["T3"]["start_close_time"] == "2026-08-01T18:00:00+00:00"
    assert windows["T3"]["end_close_time"] == "2026-08-01T18:00:00+00:00"
    assert windows["T3"]["unique_market_count"] == 1


def test_train_temporal_validation_uses_only_past_markets():
    """Verify that walk-forward validation folds group whole markets and forbid future leaks."""
    base_time = pd.Timestamp("2026-08-01T00:00:00Z")
    rows = []
    for m_idx in range(12):
        m_id = f"m_{m_idx:02d}"
        m_close = base_time + timedelta(hours=m_idx)
        for s_idx in range(2):
            rows.append({
                "market_id": m_id,
                "close_time": m_close,
                "recorded_at": m_close - timedelta(minutes=10 - s_idx * 5),
                "target": m_idx % 2,
            })
    df = pd.DataFrame(rows)

    folds = grouped_walk_forward_folds(
        groups=df["market_id"],
        timestamps=df["close_time"],
        n_splits=3,
    )
    assert len(folds) >= 2

    for fold in folds:
        train_groups = set(fold.train_groups)
        val_groups = set(fold.validation_groups)
        assert len(train_groups & val_groups) == 0

        train_ct = df[df["market_id"].isin(train_groups)]["close_time"]
        val_ct = df[df["market_id"].isin(val_groups)]["close_time"]
        assert train_ct.max() <= val_ct.min()


def test_oof_artifact_serialization_preserves_close_time_and_source_columns():
    """Verify that OOF artifact schema preserves close_time, market_close_at, resolved_at, end_time_est, market_start."""
    frame = pd.DataFrame({
        "market_id": ["m1", "m2"],
        "market_start": pd.to_datetime(["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z"], utc=True),
        "end_time_est": pd.to_datetime(["2026-08-01T00:15:00Z", "2026-08-01T01:15:00Z"], utc=True),
        "resolved_at": pd.to_datetime(["2026-08-01T00:16:00Z", "2026-08-01T01:16:00Z"], utc=True),
        "market_close_at": pd.to_datetime(["2026-08-01T00:15:00Z", "2026-08-01T01:15:00Z"], utc=True),
        "close_time": pd.to_datetime(["2026-08-01T00:15:00Z", "2026-08-01T01:15:00Z"], utc=True),
        "recorded_at": pd.to_datetime(["2026-08-01T00:05:00Z", "2026-08-01T01:05:00Z"], utc=True),
        "time_left_min": [10.0, 10.0],
        "mid_price": [0.45, 0.55],
        "target": [0, 1],
        "final_outcome": ["NO", "YES"],
    })
    quotes = pd.DataFrame({
        "market_id": ["m1", "m2"],
        "best_bid": [0.44, 0.54],
        "best_ask": [0.46, 0.56],
    })

    blob = serialize_oof_artifact(
        frame,
        np.array([0.2, 0.8]),
        quotes,
        feature_set="BASE",
    )
    payload = deserialize_oof_artifact(blob)
    d_frame = payload["frame"]

    for col in ("market_start", "end_time_est", "resolved_at", "market_close_at", "close_time"):
        assert col in d_frame.columns
        assert pd.api.types.is_datetime64_any_dtype(d_frame[col])
        assert d_frame[col].iloc[0] == frame[col].iloc[0]
        assert d_frame[col].iloc[1] == frame[col].iloc[1]


def test_audit_single_model_runs_with_canonical_adapter():
    """Regression test: audit_single_model evaluates branches and OOT windows without AttributeError on n_markets."""
    model = MagicMock()
    model.id = 101
    model.asset = "BTC"
    model.version = 5
    model.is_active = True
    model.decision_threshold = 0.55
    model.decision_threshold_down = 0.45
    model.features = "mid_price,spread"
    model.training_params = {}
    model.trained_at = None

    times = [
        pd.Timestamp("2026-08-01T10:00:00Z"),
        pd.Timestamp("2026-08-01T11:00:00Z"),
        pd.Timestamp("2026-08-01T12:00:00Z"),
    ]
    frame = pd.DataFrame([
        {
            "market_id": f"m{i}",
            "recorded_at": t - timedelta(minutes=5),
            "market_close_at": t,
            "mid_price": 0.30,
            "best_bid": 0.29,
            "best_ask": 0.31,
            "spread": 0.02,
            "final_outcome": "YES",
            "target": 1,
        }
        for i, t in enumerate(times)
    ])
    quotes = frame.copy()
    oof_scores = np.array([0.90, 0.90, 0.90])

    payload = {
        "frame": frame,
        "quotes": quotes,
        "oof_scores": oof_scores,
        "raw_oof_scores": oof_scores,
    }

    res = audit_single_model(model, payload)
    assert res["model_id"] == 101
    assert "combined_branch" in res
    assert isinstance(res["combined_branch"]["n_markets"], int)
    assert isinstance(res["combined_branch"]["net_profit"], float)
    assert "oot_windows" in res
    assert "T1" in res["oot_windows"]
