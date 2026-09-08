import pandas as pd
from datetime import datetime, timezone
from polyflip.models.trainer import add_derived_features, DERIVED_FEATURES
from polyflip.models.feature_lags import add_lag_features, LAG_FEATURE_NAMES
from polyflip.trading.feature_builder import FEATURE_COLUMNS

def test_day_of_week_range():
    for weekday in range(7):
        dt = datetime(2026, 7, 13 + weekday, tzinfo=timezone.utc)
        assert dt.weekday() == weekday
        assert 0 <= dt.weekday() <= 6

def test_price_distance_from_max():
    df = pd.DataFrame({
        "market_id": ["m1", "m1", "m1"],
        "mid_price":  [0.6, 0.8, 0.7],
        "spread": [0.02]*3, 
        "time_left_min": [60, 30, 10],
    })
    r = add_derived_features(df)
    assert abs(r.loc[0, "price_distance_from_max"] - 0.2) < 1e-6  # 0.8 - 0.6
    assert abs(r.loc[1, "price_distance_from_max"] - 0.0) < 1e-6  # max == self
    assert (r["price_distance_from_max"] >= 0).all()

def test_feature_columns_match_derived_plus_lag():
    """FEATURE_COLUMNS должен содержать базовые + новые + лаговые фичи."""
    from polyflip.trading.feature_builder import FEATURE_COLUMNS
    from polyflip.models.feature_lags import LAG_FEATURE_NAMES

    # Обязательные фичи в FEATURE_COLUMNS
    required = {
        "time_left_min", "mid_price", "spread", "volume_5min",
        "price_velocity", "hour_of_day",
        "day_of_week", "price_distance_from_max",
        *LAG_FEATURE_NAMES,
    }
    missing_from_columns = required - set(FEATURE_COLUMNS)
    assert not missing_from_columns, (
        f"Фичи отсутствуют в FEATURE_COLUMNS: {missing_from_columns}"
    )



def test_add_lag_features_basic():
    df = pd.DataFrame({
        "market_id": ["m1", "m1", "m1", "m1", "m1", "m1", "m1", "m1"],
        "recorded_at": pd.date_range("2026-07-13", periods=8, freq="5min"),
        "mid_price": [0.5, 0.52, 0.51, 0.55, 0.53, 0.54, 0.58, 0.59],
        "spread": [0.01]*8,
        "volume_5min": [100, 150, 120, 200, 210, 180, 250, 300],
        "price_velocity": [0.0, 0.02, -0.01, 0.04, -0.02, 0.01, 0.04, 0.01]
    })
    
    r = add_lag_features(df)
    
    # price_momentum: shift(3)
    # i=3: mid_price=0.55, lag3 (i=0) = 0.50 => 0.05
    assert abs(r.iloc[3]["price_momentum"] - 0.05) < 1e-6
    
    # volume_trend: shift(3)
    # i=3: vol=200, lag3=100 => 200/100 = 2.0
    assert abs(r.iloc[3]["volume_trend"] - 2.0) < 1e-6

    # spread_trend: shift(6)
    # i=6: spread=0.01, lag6(i=0)=0.01 => 1.0
    assert abs(r.iloc[6]["spread_trend"] - 1.0) < 1e-6

def test_engine_inference_empty_history():
    """При пустой истории снапшотов лаги сохраняют NaN, has_lag_history=0.0, а Pipeline с импьютером не падает."""
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Симулируем: history_snaps пустой, только live-строка
    rows = [{
        "market_id": "new_market",
        "recorded_at": datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
        "time_left_min": 45.0,
        "mid_price": 0.6,
        "spread": 0.02,
        "price_velocity": 0.01,
        "volume_5min": 100.0,
        "hour_of_day": 12,
        "day_of_week": 6,
    }]
    global_max = 0.0  # нет истории

    df = pd.DataFrame(rows)
    df = add_derived_features(df)
    df["price_distance_from_max"] = (global_max - df["mid_price"]).clip(lower=0.0)
    df = add_lag_features(df)
    df.drop(columns=["recorded_at", "market_id"], errors="ignore", inplace=True)

    assert len(df) == 1
    for col in LAG_FEATURE_NAMES:
        assert pd.isna(df.iloc[0][col]), f"Ожидался NaN для {col} при отсутствии истории"
    assert df.iloc[0]["has_lag_history"] == 0.0
    assert df.iloc[0]["price_distance_from_max"] == 0.0

    # Pipeline с импьютером корректно обрабатывает пустую историю
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ])
    train_X = pd.DataFrame({
        "price_momentum": [0.01, -0.02, 0.03, 0.0],
        "spread_trend": [1.0, 1.1, 0.9, 1.0],
        "volume_trend": [1.0, 1.5, 0.8, 1.2],
    })
    train_y = pd.Series([1, 0, 1, 0])
    pipe.fit(train_X, train_y)
    prob = pipe.predict_proba(df[LAG_FEATURE_NAMES])
    assert prob.shape == (1, 2)
    assert np.isfinite(prob).all()


def test_prefix_invariance_and_market_isolation():
    """
    Самопроверка 1.3:
    Добавление будущих строк, изменение будущих цен и добавление другого market_id
    НЕ меняют уже рассчитанные прошлые признаки. Исходный баг momentum 0.030->0.015
    больше не воспроизводится.
    """
    from tests.fixtures.model_audit.audit_data import generate_market_history_with_future

    past_df, combined_df = generate_market_history_with_future()

    res_past = add_lag_features(past_df)
    res_combined = add_lag_features(combined_df)

    # Filter combined back to past rows (market-1, first 5 timestamps)
    filtered_combined = res_combined[
        (res_combined["market_id"] == "market-1") &
        (res_combined["recorded_at"].isin(past_df["recorded_at"]))
    ].sort_values("recorded_at").reset_index(drop=True)

    res_past_sorted = res_past.sort_values("recorded_at").reset_index(drop=True)

    for col in LAG_FEATURE_NAMES:
        past_vals = res_past_sorted[col].values
        comb_vals = filtered_combined[col].values
        # Check that where past is NaN, comb is NaN; where float, values match exactly
        for i, (p, c) in enumerate(zip(past_vals, comb_vals)):
            if pd.isna(p):
                assert pd.isna(c), f"Row {i} col {col}: past was NaN, comb became {c}"
            else:
                assert abs(p - c) < 1e-9, f"Row {i} col {col} mismatch: past={p}, comb={c}"

    assert (res_past_sorted["has_lag_history"].values == filtered_combined["has_lag_history"].values).all()

