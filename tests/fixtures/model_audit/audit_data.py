"""
Reproducible test fixtures for model audit (Step 1.1).
Contains compact, anonymized synthetic datasets with known ground truths:
- Candles with non-constant volatility (demonstrating ddof=0 vs ddof=1 discrepancy)
- Past market snapshots + independent future market / future snapshots (prefix invariance)
- Opposing side contracts (YES/NO target alignment)
- Decision snapshots before and after decision_at
- BTC 800 points integration fixture generator
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


def generate_variable_volatility_candles(n: int = 120) -> pd.DataFrame:
    """Generate candles where volatility changes dramatically, highlighting ddof effects."""
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # First 60 candles low vol, next 60 high vol
    np.random.seed(42)
    closes = [50000.0]
    for i in range(1, n):
        vol = 0.0005 if i < 60 else 0.008
        change = np.random.normal(0, vol)
        closes.append(closes[-1] * (1.0 + change))

    closes = np.array(closes)
    highs = closes * (1.0 + np.abs(np.random.normal(0, 0.001, n)))
    lows = closes * (1.0 - np.abs(np.random.normal(0, 0.001, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.uniform(10.0, 100.0, n)
    tbvs = volumes * np.random.uniform(0.3, 0.7, n)

    return pd.DataFrame({
        "open_time": [base_time + timedelta(minutes=15 * i) for i in range(n)],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "taker_buy_volume": tbvs,
    })


def generate_market_history_with_future() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - past_df: 5 snapshots for market-1 up to T
    - combined_df: past_df + 3 future snapshots for market-1 (T+1 to T+3) + 5 snapshots for market-2
    """
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    past_rows = []
    for i in range(5):
        past_rows.append({
            "market_id": "market-1",
            "recorded_at": base_time + timedelta(minutes=i),
            "mid_price": 0.40 + 0.02 * i,
            "spread": 0.02,
            "volume_5min": 100.0 + 10.0 * i,
            "price_velocity": 0.005,
            "time_left_min": 15.0 - i,
            "market_duration_min": 15.0,
        })
    past_df = pd.DataFrame(past_rows)

    future_rows = []
    for i in range(5, 8):
        future_rows.append({
            "market_id": "market-1",
            "recorded_at": base_time + timedelta(minutes=i),
            "mid_price": 0.60 + 0.05 * (i - 5),
            "spread": 0.04,
            "volume_5min": 500.0,
            "price_velocity": 0.05,
            "time_left_min": 15.0 - i,
            "market_duration_min": 15.0,
        })
    # Independent market-2
    m2_rows = []
    for i in range(5):
        m2_rows.append({
            "market_id": "market-2",
            "recorded_at": base_time + timedelta(minutes=i),
            "mid_price": 0.80 - 0.01 * i,
            "spread": 0.01,
            "volume_5min": 50.0,
            "price_velocity": -0.01,
            "time_left_min": 20.0 - i,
            "market_duration_min": 20.0,
        })

    combined_df = pd.concat([past_df, pd.DataFrame(future_rows), pd.DataFrame(m2_rows)], ignore_index=True)
    return past_df, combined_df


def generate_post_decision_snapshots() -> pd.DataFrame:
    """Snapshots spanning before, at, and after a decision point."""
    decision_at = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
    rows = []
    for offset_min in range(-5, 6):
        rows.append({
            "market_id": "market-decision-test",
            "recorded_at": decision_at + timedelta(minutes=offset_min),
            "mid_price": 0.45 + 0.01 * offset_min,
            "spread": 0.02,
            "volume_5min": 100.0,
            "price_velocity": 0.01,
            "time_left_min": 10.0 - offset_min,
            "market_duration_min": 15.0,
        })
    return pd.DataFrame(rows)


def generate_btc_800_integration_points() -> pd.DataFrame:
    """
    Integration dataset of 800 synthetic BTC points modeling realistic
    crypto volatility, used for checking routing and inference stability.
    """
    n = 800
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    np.random.seed(123)
    closes = [45000.0]
    for _ in range(1, n + 120):
        ret = np.random.normal(0.0001, 0.004)
        closes.append(closes[-1] * (1.0 + ret))
    closes = np.array(closes)

    highs = closes * (1.0 + np.abs(np.random.normal(0, 0.002, len(closes))))
    lows = closes * (1.0 - np.abs(np.random.normal(0, 0.002, len(closes))))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.uniform(50.0, 500.0, len(closes))
    tbvs = volumes * np.random.uniform(0.4, 0.6, len(closes))

    df = pd.DataFrame({
        "open_time": [base_time + timedelta(minutes=15 * i) for i in range(len(closes))],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "taker_buy_volume": tbvs,
    })
    return df
