import pandas as pd
from polyflip.models.trainer import add_derived_features

def test_interaction_features_no_nan():
    df = pd.DataFrame({
        "mid_price":      [0.50, 0.65, 0.80, 0.30],
        "spread":         [0.02, 0.03, 0.01, 0.04],
        "time_left_min":  [1.0,  5.0,  12.0, 0.5],
        "volume_5min":    [100, 200, 50, 300],
        "price_velocity": [0.01, -0.02, 0.0, 0.05],
        "hour_of_day":    [10, 14, 20, 8],
        "market_id":      ["m1", "m1", "m2", "m2"],
    })
    result = add_derived_features(df)

    for feat in ["is_final_phase", "high_price_final", "velocity_x_phase", "dev_sq_x_phase"]:
        assert feat in result.columns, f"Missing feature: {feat}"
        assert result[feat].isna().sum() == 0, f"NaN in {feat}"

def test_time_phase_values():
    """Проверяет что time_phase корректно нормализован."""
    df = pd.DataFrame({
        "mid_price": [0.5, 0.6],
        "spread": [0.02, 0.02],
        "time_left_min": [3.0, 15.0],  # 3/15=0.2, 15/15=1.0
        "volume_5min": [100, 100],
        "price_velocity": [0.0, 0.0],
        "hour_of_day": [10, 10],
        "market_id": ["m1", "m2"], # Инференс (нет дубликатов)
    })
    result = add_derived_features(df)
    assert abs(result["time_phase"].iloc[0] - 0.20) < 1e-4  # 3/15
    assert abs(result["time_phase"].iloc[1] - 1.00) < 1e-4  # 15/15
    assert result["is_final_phase"].iloc[0] == 1.0   # <= 0.20
    assert result["is_final_phase"].iloc[1] == 0.0
