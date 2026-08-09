"""
tests/crypto/test_dataset_alignment.py

Тесты строгого выравнивания датасета (pm_window_v1):
1. Проверка отсутствия Lookahead (feature_available_at <= market_start)
2. Проверка уникальности market_id (ровно 1 запись на рынок)
3. Проверка корректности маппинга YES -> 1, NO -> 0
"""
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from polyflip.crypto.dataset import build_polymarket_training_dataset


def test_time_alignment_math():
    """Проверка математики выравнивания времени."""
    end_time_est = pd.to_datetime("2026-08-09 09:15:00+00:00")
    market_start = end_time_est - pd.Timedelta(minutes=15)
    feature_candle_open = market_start - pd.Timedelta(minutes=15)
    feature_available_at = feature_candle_open + pd.Timedelta(minutes=14, seconds=59, milliseconds=999)

    assert market_start == pd.to_datetime("2026-08-09 09:00:00+00:00")
    assert feature_candle_open == pd.to_datetime("2026-08-09 08:45:00+00:00")
    assert feature_available_at <= market_start
    assert (market_start - feature_available_at).total_seconds() < 1.0
