import pandas as pd
from polyflip.models.trainer import add_derived_features

def test_price_distance_from_max_respects_time_order():
    """expanding max должен учитывать только прошлые снапшоты (по времени)."""
    df = pd.DataFrame({
        "market_id":    ["m1", "m1", "m1"],
        "recorded_at":  pd.to_datetime(["2026-07-01 10:00", "2026-07-01 10:15", "2026-07-01 10:30"]),
        "mid_price":    [0.4, 0.8, 0.6],  # max исторически = 0.4 в t=0, 0.8 в t=1
        "spread":       [0.01, 0.01, 0.01],
        "time_left_min": [60, 45, 30],
    })
    result = add_derived_features(df.copy())
    # t=0: expanding_max(t=0) = 0.4, distance = 0.0
    assert abs(result.loc[0, "price_distance_from_max"] - 0.0) < 1e-6
    # t=1: expanding_max(t=0..1) = 0.8, distance = 0.0 (сам максимум)
    assert abs(result.loc[1, "price_distance_from_max"] - 0.0) < 1e-6
    # t=2: expanding_max(t=0..2) = 0.8, distance = 0.8 - 0.6 = 0.2
    assert abs(result.loc[2, "price_distance_from_max"] - 0.2) < 1e-6

def test_price_distance_from_max_no_future_leakage():
    """Строка с низкой ценой в начале НЕ должна знать о будущем высоком max."""
    df = pd.DataFrame({
        "market_id":   ["m1", "m1"],
        "recorded_at": pd.to_datetime(["2026-07-01 10:00", "2026-07-01 10:15"]),
        "mid_price":   [0.3, 0.9],
        "spread":      [0.01, 0.01],
        "time_left_min": [60, 30],
    })
    result = add_derived_features(df.copy())
    # t=0: future max = 0.9, но expanding max должен знать только 0.3
    assert result.loc[0, "price_distance_from_max"] == 0.0  # expanding max = 0.3, price = 0.3
