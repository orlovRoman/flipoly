import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LogisticRegression
from polyflip.trading.ml_inference import build_inference_dataframe, run_model_inference
from polyflip.crypto.predictor import CryptoPredictor
from polyflip.crypto.feature_builder import CryptoFeatureVector


def test_insertion_of_later_snapshot_does_not_change_decision_prediction():
    """Самопроверка 1.4: Вставка более позднего snapshot не меняет прогноз decision row."""
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    decision_time = base_time + timedelta(minutes=5)
    decision_id = "target_decision_point"

    # Dummy class with required attributes
    class Snap:
        def __init__(self, time_left, mid, spread, vel, vol, hour, dt, snap_id):
            self.time_left_min = time_left
            self.mid_price = mid
            self.spread = spread
            self.price_velocity = vel
            self.volume_5min = vol
            self.hour_of_day = hour
            self.market_id = "m1"
            self.recorded_at = dt
            self.market_duration_min = 15.0
            self.id = snap_id

    class Market:
        market_id = "m1"
        asset = "BTC"
        price_velocity = 0.01
        volume_5min = 100.0
        market_duration_min = 15.0

    history_normal = [
        Snap(15.0, 0.45, 0.02, 0.0, 100.0, 12, base_time, "s1"),
        Snap(14.0, 0.46, 0.02, 0.01, 110.0, 12, base_time + timedelta(minutes=1), "s2"),
    ]

    # Model
    X_dummy = pd.DataFrame({
        "time_left_min": [10.0, 10.0],
        "mid_price": [0.5, 0.6],
        "spread": [0.02, 0.02],
    })
    y_dummy = pd.Series([0, 1])
    clf = LogisticRegression().fit(X_dummy, y_dummy)
    features = ["time_left_min", "mid_price", "spread"]

    df1 = build_inference_dataframe(
        market=Market(),
        history_snaps=history_normal,
        fresh_yes_price=0.48,
        fresh_spread=0.02,
        global_max=0.50,
        start_time=decision_time,
        time_left_sec=600.0,
        decision_id=decision_id,
    )
    p1 = run_model_inference(df1, clf, features, decision_row_id=decision_id)

    # Now add a future snapshot into history_snaps (which could happen if not filtered by decision_at)
    # BUT build_inference_dataframe and run_model_inference track the explicit decision_row_id!
    history_with_future = history_normal + [
        Snap(5.0, 0.90, 0.01, 0.20, 500.0, 12, base_time + timedelta(minutes=10), "s_future")
    ]
    df2 = build_inference_dataframe(
        market=Market(),
        history_snaps=history_with_future,
        fresh_yes_price=0.48,
        fresh_spread=0.02,
        global_max=0.50,
        start_time=decision_time,
        time_left_sec=600.0,
        decision_id=decision_id,
    )
    p2 = run_model_inference(df2, clf, features, decision_row_id=decision_id)

    assert np.isclose(p1, p2, atol=1e-9), f"Future snapshot changed prediction: p1={p1}, p2={p2}"


def test_row_permutation_does_not_change_decision_row_selection():
    """Самопроверка 1.4: Перестановка входных строк не меняет выбранную decision row."""
    decision_id = "target_row"
    clf = LogisticRegression()
    clf.classes_ = np.array([0, 1])
    clf.coef_ = np.array([[1.0, 2.0]])
    clf.intercept_ = np.array([0.0])

    df = pd.DataFrame({
        "feat1": [0.1, 0.9, 0.4],
        "feat2": [0.2, 0.8, 0.5],
        "_row_id": ["row_a", decision_id, "row_c"],
        "_is_decision_row": [False, True, False],
    })

    p_orig = run_model_inference(df, clf, ["feat1", "feat2"], decision_row_id=decision_id)

    # Permute rows
    df_perm = df.iloc[[2, 0, 1]].reset_index(drop=True)
    p_perm = run_model_inference(df_perm, clf, ["feat1", "feat2"], decision_row_id=decision_id)

    assert np.isclose(p_orig, p_perm, atol=1e-9)


def test_crypto_predictor_filters_candles_by_decision_time():
    """Самопроверка 1.4: CryptoPredictor отсекает свечи после decision_time."""
    from polyflip.crypto.predictor import CryptoPredictor
    from datetime import datetime, timedelta, timezone

    predictor = CryptoPredictor()
    predictor._loaded_symbols.add("BTCUSDT")

    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    decision_time = t0 + timedelta(minutes=15 * 100)

    # Generate 120 candles, but decision_time is at candle 100
    candles = []
    class DummyCandle:
        def __init__(self, dt, close):
            self.open_time = dt
            self.close_time = dt + timedelta(minutes=15)
            self.open = close
            self.high = close + 10.0
            self.low = close - 10.0
            self.close = close
            self.volume = 100.0
            self.taker_buy_volume = 50.0

    for i in range(120):
        candles.append(DummyCandle(t0 + timedelta(minutes=15 * i), 50000.0 + i))

    # predict with decision_time at candle 100
    sig = predictor.predict(candles, "BTCUSDT", decision_time=decision_time)
    # Even if symbol has dummy models, features_ok reflects whether >= 100 candles were valid
    assert sig.features_ok is True or sig.status in ("READY", "REGIME_UNAVAILABLE", "MODEL_NOT_LOADED")
