import numpy as np
import pandas as pd
from polyflip.models.trainer import _fit_and_serialize

def test_c_grid_survives_single_class_folds():
    """_fit_and_serialize не должен падать если некоторые фолды имеют один класс."""
    rng = np.random.default_rng(42)
    # 5 рынков, 4 снапшота каждый, намеренно почти все target=1 (имбаланс)
    n_markets = 5
    snaps = 4
    rows = []
    for mid in range(n_markets):
        for t in range(snaps):
            rows.append({
                "mid_price": 0.5 + 0.01 * t,
                "time_left_min": 60 - t * 5,
                "spread": 0.01,
                "volume_5min": 100,
                "price_velocity": 0.001 * t,
                "hour_of_day": t,
                "day_of_week": t % 7,
                "price_distance_from_max": 0.0,
                "time_phase": 1.0 - t / (snaps - 1),
                "price_deviation": abs(0.5 + 0.01 * t - 0.5),
                "deviation_x_time": 0.0,
                "price_deviation_sq": 0.0,
                "spread_pct": 0.02,
                "log_time_left": np.log1p(60 - t * 5),
                "price_velocity_lag1": 0.0,
                "price_momentum": 0.0,
                "spread_trend": 1.0,
                "volume_trend": 1.0,
                "_group": f"market_{mid}",
                "_target": int(mid < 4),  # почти все 1
            })
    df = pd.DataFrame(rows)
    X = df[[c for c in df.columns if not c.startswith("_")]]
    y = pd.Series(df["_target"].values)
    groups = pd.Series(df["_group"].values)
    # Не должен бросить исключение
    result = _fit_and_serialize(X, y, groups)
    assert result is not None
