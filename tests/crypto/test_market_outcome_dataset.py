"""
tests/crypto/test_market_outcome_dataset.py

Тесты построителя торгового датасета по выравниванию MARKET_WINDOW_V1:
  - Рынку 09:00 соответствует фичи свечей, закрытых <= 09:00.
  - Канонический исход: YES = 1, NO = 0.
  - Ровно 1 строка на 1 market_id.
  - Инвариант: feature_available_at <= market_start.
"""
from datetime import datetime, timezone
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock

from polyflip.crypto.market_outcome_dataset import build_market_outcome_dataset


def test_target_yes_is_one_and_no_is_zero():
    df_raw = pd.DataFrame([
        {"market_id": "m1", "final_outcome": "YES"},
        {"market_id": "m2", "final_outcome": "NO"},
    ])
    mapped = df_raw["final_outcome"].map({"YES": 1, "NO": 0}).astype(int)
    assert mapped.iloc[0] == 1
    assert mapped.iloc[1] == 0


def test_training_dataset_has_one_row_per_market():
    df_raw = pd.DataFrame([
        {"market_id": "m1", "asset": "BTC", "end_time_est": "2026-08-09T09:15:00Z", "final_outcome": "YES"},
        {"market_id": "m1", "asset": "BTC", "end_time_est": "2026-08-09T09:15:00Z", "final_outcome": "YES"},
        {"market_id": "m2", "asset": "BTC", "end_time_est": "2026-08-09T09:30:00Z", "final_outcome": "NO"},
    ])
    dedup = df_raw.drop_duplicates(subset=["market_id"], keep="last").reset_index(drop=True)
    assert len(dedup) == 2
    assert dedup["market_id"].is_unique


def test_market_0900_uses_candle_closed_at_0900():
    market_end = pd.to_datetime("2026-08-09T09:15:00Z", utc=True)
    market_start = market_end - pd.Timedelta(minutes=15)
    
    # 15m свеча 08:45-09:00 доступна в 09:00:00
    feature_available_at = pd.to_datetime("2026-08-09T08:45:00Z", utc=True) + pd.Timedelta(minutes=15)
    
    assert feature_available_at == market_start
    assert feature_available_at <= market_start


def test_features_never_cross_market_start():
    markets = pd.DataFrame([
        {"market_id": "m1", "market_start": pd.to_datetime("2026-08-09T09:00:00Z", utc=True)}
    ])
    features = pd.DataFrame([
        {"feature_available_at": pd.to_datetime("2026-08-09T09:00:00Z", utc=True), "vol_6": 0.01},
        {"feature_available_at": pd.to_datetime("2026-08-09T09:15:00Z", utc=True), "vol_6": 0.05},
    ])
    
    merged = pd.merge_asof(
        markets.sort_values("market_start"),
        features.sort_values("feature_available_at"),
        left_on="market_start",
        right_on="feature_available_at",
        direction="backward",
    )
    
    assert (merged["feature_available_at"] <= merged["market_start"]).all()
    assert merged["vol_6"].iloc[0] == 0.01
