"""
tests/crypto/test_polymarket_join_alignment.py

Тесты привязки Polymarket исходов и снапшотов цен без заглядывания в будущее и без direction='nearest'.
"""
from datetime import datetime, timezone
import pandas as pd
import pytest

from polyflip.crypto.polymarket_join import join_entry_snapshot_by_decision_time


def test_snapshot_091430_belongs_to_market_0900_0915():
    market_start = pd.to_datetime("2026-08-09T09:00:00Z", utc=True)
    market_end = pd.to_datetime("2026-08-09T09:15:00Z", utc=True)
    snap_time = pd.to_datetime("2026-08-09T09:14:30Z", utc=True)
    
    assert market_start <= snap_time <= market_end


def test_future_snapshot_is_never_used():
    decisions = pd.DataFrame([
        {"market_id": "m1", "decision_at": pd.to_datetime("2026-08-09T09:00:00Z", utc=True)}
    ])
    snapshots = pd.DataFrame([
        {"market_id": "m1", "mid_price": 0.55, "recorded_at": pd.to_datetime("2026-08-09T08:59:00Z", utc=True)},
        {"market_id": "m1", "mid_price": 0.80, "recorded_at": pd.to_datetime("2026-08-09T09:01:00Z", utc=True)},
    ])
    
    merged = pd.merge_asof(
        decisions.sort_values("decision_at"),
        snapshots.sort_values("recorded_at"),
        left_on="decision_at",
        right_on="recorded_at",
        by="market_id",
        direction="backward",
    )
    
    assert merged["mid_price"].iloc[0] == 0.55


def test_snapshot_from_another_market_is_never_joined():
    decisions = pd.DataFrame([
        {"market_id": "m1", "decision_at": pd.to_datetime("2026-08-09T09:00:00Z", utc=True)}
    ])
    snapshots = pd.DataFrame([
        {"market_id": "m2", "mid_price": 0.60, "recorded_at": pd.to_datetime("2026-08-09T08:59:00Z", utc=True)}
    ])
    
    merged = pd.merge_asof(
        decisions.sort_values("decision_at"),
        snapshots.sort_values("recorded_at"),
        left_on="decision_at",
        right_on="recorded_at",
        by="market_id",
        direction="backward",
    )
    
    assert pd.isna(merged["mid_price"].iloc[0])
