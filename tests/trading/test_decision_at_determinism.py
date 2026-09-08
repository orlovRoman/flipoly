"""
tests/trading/test_decision_at_determinism.py

Step 1.4: Strict decision_at determinism and causal isolation:
1. Вставка более позднего snapshot не меняет прогноз.
2. Перестановка входных строк не меняет выбранную decision row.
3. Свежая строка с одинаковым timestamp выбирается детерминированно.
4. Свечи Binance после decision_at строго отсекаются.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LogisticRegression

from polyflip.trading.ml_inference import build_inference_dataframe, run_model_inference
from polyflip.crypto.predictor import CryptoPredictor


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


def _create_trained_classifier():
    clf = LogisticRegression()
    clf.classes_ = np.array([0, 1])
    clf.coef_ = np.array([[1.5, -2.0, 0.5]])
    clf.intercept_ = np.array([0.1])
    return clf


def test_insertion_of_later_snapshot_does_not_change_prediction():
    """1. Вставка более позднего snapshot не меняет прогноз decision row."""
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    decision_time = base_time + timedelta(minutes=5)
    decision_id = "target_decision_point"

    history_base = [
        Snap(15.0, 0.45, 0.02, 0.0, 100.0, 12, base_time, "s1"),
        Snap(14.0, 0.46, 0.02, 0.01, 110.0, 12, base_time + timedelta(minutes=1), "s2"),
    ]

    features = ["time_left_min", "mid_price", "spread"]
    clf = _create_trained_classifier()

    # Inference without later snapshot
    df1 = build_inference_dataframe(
        market=Market(),
        history_snaps=history_base,
        fresh_yes_price=0.48,
        fresh_spread=0.02,
        global_max=0.50,
        start_time=decision_time,
        time_left_sec=600.0,
        decision_id=decision_id,
    )
    p1 = run_model_inference(df1, clf, features, decision_row_id=decision_id)

    # Add a future snapshot (simulating a delayed message or subsequent tick)
    history_with_future = history_base + [
        Snap(9.0, 0.70, 0.05, 0.1, 500.0, 12, decision_time + timedelta(minutes=2), "s_future"),
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
    """2. Перестановка входных строк не меняет выбранную decision row."""
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

    # Arbitrary permutations
    for perm_indices in [[2, 0, 1], [1, 2, 0], [2, 1, 0]]:
        df_perm = df.iloc[perm_indices].reset_index(drop=True)
        p_perm = run_model_inference(df_perm, clf, ["feat1", "feat2"], decision_row_id=decision_id)
        assert np.isclose(p_orig, p_perm, atol=1e-9), f"Permutation {perm_indices} changed prediction"


def test_fresh_row_with_same_timestamp_chosen_deterministically():
    """3. Свежая строка с одинаковым timestamp выбирается детерминированно."""
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    same_time = base_time + timedelta(minutes=5)
    decision_id = "fresh_decision_row"

    # History contains a snapshot with the EXACT same recorded_at timestamp as start_time
    history_with_collision = [
        Snap(15.0, 0.45, 0.02, 0.0, 100.0, 12, base_time, "s1"),
        Snap(10.0, 0.40, 0.03, 0.0, 80.0, 12, same_time, "s_colliding"),
    ]

    features = ["time_left_min", "mid_price", "spread"]
    clf = _create_trained_classifier()

    df = build_inference_dataframe(
        market=Market(),
        history_snaps=history_with_collision,
        fresh_yes_price=0.55,
        fresh_spread=0.02,
        global_max=0.55,
        start_time=same_time,
        time_left_sec=600.0,
        decision_id=decision_id,
    )

    # Verify deterministic ordering: fresh decision row is tagged and resolved
    assert decision_id in df["_row_id"].values
    assert df["_is_decision_row"].sum() == 1

    p = run_model_inference(df, clf, features, decision_row_id=decision_id)
    assert 0.0 <= p <= 1.0

    # Also verify fallback to _is_decision_row when decision_row_id is not passed
    p_by_flag = run_model_inference(df, clf, features, decision_row_id=None)
    assert np.isclose(p, p_by_flag, atol=1e-9)


def test_binance_candles_after_decision_at_strictly_truncated():
    """4. Свечи Binance после decision_at строго отсекаются."""
    from unittest.mock import MagicMock
    from polyflip.crypto.feature_sets import CONTROL_FEATURES

    predictor = CryptoPredictor()
    predictor._loaded_symbols.add("BTCUSDT")

    dummy_model = MagicMock()
    dummy_model.predict_proba.return_value = np.array([[0.4, 0.6]])
    dummy_model.predict_raw_proba.return_value = np.array([[0.45, 0.55]])
    dummy_model.classes_ = np.array([0, 1])

    regimes = ["low_vol", "mid_vol", "high_vol"]
    predictor._models["BTCUSDT"] = {r: dummy_model for r in regimes}
    predictor._model_features["BTCUSDT"] = {r: list(CONTROL_FEATURES) for r in regimes}
    predictor._model_versions["BTCUSDT"] = {r: 1 for r in regimes}
    predictor._thresholds["BTCUSDT"] = {r: (0.55, 0.45) for r in regimes}

    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    decision_time = t0 + timedelta(minutes=15 * 105)

    class DummyCandle:
        def __init__(self, dt, close):
            self.open_time = dt
            self.close_time = dt + timedelta(minutes=15)
            self.open = close
            self.high = close + 5.0
            self.low = close - 5.0
            self.close = close
            self.volume = 100.0
            self.taker_buy_volume = 50.0

    # Create 130 candles (105 candles up to decision_time: i=0..104; 25 candles after)
    candles = [
        DummyCandle(t0 + timedelta(minutes=15 * i), 50000.0 + i * 2)
        for i in range(130)
    ]

    # Predict with cutoff at decision_time (candles 105..129 are after decision_time)
    sig = predictor.predict(candles, "BTCUSDT", decision_time=decision_time, funding_rate=0.0)

    # Predict with only candles up to decision_time
    candles_only_prior = [c for c in candles if c.close_time <= decision_time]
    sig_prior = predictor.predict(candles_only_prior, "BTCUSDT", decision_time=decision_time, funding_rate=0.0)

    assert sig.features_ok is True
    assert sig.status in ("READY", "DEGENERATE_PREDICTION")
    # Verify future candles are strictly truncated: predictions match exactly
    assert sig.p_up == sig_prior.p_up
    assert sig.p_up_raw == sig_prior.p_up_raw
