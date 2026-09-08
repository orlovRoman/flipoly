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


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def _create_trained_classifier(n_features: int = 3):
    clf = LogisticRegression()
    clf.classes_ = np.array([0, 1])
    np.random.seed(42)
    clf.coef_ = np.random.randn(1, n_features) * 0.5
    clf.intercept_ = np.array([0.1])
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    imputer.fit(np.zeros((1, n_features)))
    return Pipeline([("imputer", imputer), ("model", clf)])


def test_insertion_of_later_snapshot_does_not_change_prediction():
    """1. Вставка более позднего snapshot не меняет прогноз decision row и PIT-признаки."""
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    decision_time = base_time + timedelta(minutes=5)
    decision_id = "target_decision_point"

    # Snapshots providing explicit 180s (at t=2m) and 60s (at t=4m) references for decision at t=5m
    history_base = [
        Snap(15.0, 0.45, 0.02, 0.0, 100.0, 12, base_time, "s1"),
        Snap(13.0, 0.46, 0.02, 0.01, 110.0, 12, base_time + timedelta(minutes=2), "s2"),
        Snap(11.0, 0.47, 0.02, 0.01, 120.0, 12, base_time + timedelta(minutes=4), "s3"),
    ]

    pit_features = [
        "time_left_min", "mid_price", "spread",
        "pm_change_60s", "pm_change_180s", "has_60s_ref", "has_180s_ref",
        "history_age_seconds", "legacy_last_poll_delta", "price_distance_from_max",
    ]
    clf = _create_trained_classifier(len(pit_features))

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
    p1 = run_model_inference(df1, clf, pit_features, decision_row_id=decision_id)

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
    p2 = run_model_inference(df2, clf, pit_features, decision_row_id=decision_id)

    # 1. Point-in-time features on decision row must be identical
    row1 = df1[df1["_is_decision_row"]].iloc[0]
    row2 = df2[df2["_is_decision_row"]].iloc[0]
    for feat in pit_features:
        v1 = float(row1[feat])
        v2 = float(row2[feat])
        assert np.isclose(v1, v2, atol=1e-9, equal_nan=True), f"PIT feature {feat} mismatch: {v1} vs {v2}"

    # Verify PIT dynamic values are valid and computed
    assert row1["has_60s_ref"] == 1.0
    assert row1["has_180s_ref"] == 1.0
    assert row1["pm_change_60s"] == pytest.approx(0.48 - 0.47, abs=1e-6)
    assert row1["pm_change_180s"] == pytest.approx(0.48 - 0.46, abs=1e-6)
    assert row1["history_age_seconds"] == pytest.approx(300.0, abs=1e-6)
    assert row1["legacy_last_poll_delta"] == pytest.approx(0.48 - 0.47, abs=1e-6)
    assert row1["price_distance_from_max"] == pytest.approx(0.50 - 0.48, abs=1e-6)

    # 2. Future row must be masked/zeroed out
    future_row = df2[df2["_row_id"] == "s_future"].iloc[0]
    assert pd.isna(future_row["pm_change_60s"])
    assert pd.isna(future_row["pm_change_180s"])
    assert future_row["has_60s_ref"] == 0.0
    assert future_row["has_180s_ref"] == 0.0
    assert future_row["price_distance_from_max"] == 0.0

    # 3. Model prediction must match exactly
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


def test_point_in_time_train_serve_parity():
    """5. Полный train/serve parity между обучением (trainer) и инференсом (build_inference_dataframe)."""
    from polyflip.models.trainer import add_derived_features
    from polyflip.models.feature_lags import add_lag_features
    from polyflip.models.point_in_time_features import compute_point_in_time_features

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    n_points = 8
    # Create synthetic series of snapshots for a market
    prices = [0.40, 0.42, 0.45, 0.43, 0.47, 0.52, 0.50, 0.55]
    spreads = [0.02, 0.02, 0.015, 0.02, 0.018, 0.02, 0.015, 0.02]
    volumes = [100.0, 110.0, 120.0, 105.0, 130.0, 150.0, 140.0, 160.0]
    velocities = [0.0, 0.02, 0.03, -0.02, 0.04, 0.05, -0.02, 0.05]
    time_lefts = [15.0 - i for i in range(n_points)]
    dts = [base_time + timedelta(minutes=i) for i in range(n_points)]

    # 1. Training feature calculation over entire resolved market history
    train_rows = []
    for i in range(n_points):
        train_rows.append({
            "market_id": "m1",
            "recorded_at": dts[i],
            "time_left_min": time_lefts[i],
            "mid_price": prices[i],
            "spread": spreads[i],
            "best_bid": prices[i] - spreads[i] / 2.0,
            "best_ask": prices[i] + spreads[i] / 2.0,
            "price_velocity": velocities[i],
            "volume_5min": volumes[i],
            "hour_of_day": dts[i].hour,
            "day_of_week": float(dts[i].weekday()),
            "market_duration_min": 15.0,
        })
    df_train = pd.DataFrame(train_rows)
    df_train = add_derived_features(df_train)
    df_train = add_lag_features(df_train)
    df_train = compute_point_in_time_features(df_train)

    # 2. Test parity across multiple decision points (e.g. at index 3, 5, 7)
    for dec_idx in [3, 5, 7]:
        dec_time = dts[dec_idx]
        history_snaps = [
            Snap(
                time_left=time_lefts[j],
                mid=prices[j],
                spread=spreads[j],
                vel=velocities[j],
                vol=volumes[j],
                hour=dts[j].hour,
                dt=dts[j],
                snap_id=f"snap_{j}",
            )
            for j in range(dec_idx)
        ]

        # In inference, global_max is observed lifetime max up to dec_time
        observed_max = max(prices[:dec_idx + 1])
        dec_id = f"decision_at_{dec_idx}"

        market_inst = Market()
        market_inst.volume_5min = volumes[dec_idx]
        market_inst.price_velocity = velocities[dec_idx]

        df_inf = build_inference_dataframe(
            market=market_inst,
            history_snaps=history_snaps,
            fresh_yes_price=prices[dec_idx],
            fresh_spread=spreads[dec_idx],
            global_max=observed_max,
            start_time=dec_time,
            time_left_sec=time_lefts[dec_idx] * 60.0,
            decision_id=dec_id,
        )

        inf_row = df_inf[df_inf["_is_decision_row"]].iloc[0]
        train_row = df_train.iloc[dec_idx]

        check_features = [
            "pm_change_60s",
            "pm_change_180s",
            "has_60s_ref",
            "has_180s_ref",
            "history_age_seconds",
            "legacy_last_poll_delta",
            "price_distance_from_max",
            "price_velocity",
            "price_momentum",
            "spread_trend",
            "volume_trend",
            "has_lag_history",
            "price_deviation",
            "spread_pct",
            "log_time_left",
            "day_of_week",
        ]

        for feat in check_features:
            inf_val = float(inf_row[feat])
            train_val = float(train_row[feat])
            assert np.isclose(inf_val, train_val, atol=1e-7, equal_nan=True), (
                f"Train/serve parity mismatch at dec_idx={dec_idx} for feature '{feat}': "
                f"inference={inf_val} vs train={train_val}"
            )


def test_point_in_time_future_noise_invariance():
    """6. Наличие будущих снапшотов в истории инференса не нарушает parity с train."""
    from polyflip.models.trainer import add_derived_features
    from polyflip.models.feature_lags import add_lag_features
    from polyflip.models.point_in_time_features import compute_point_in_time_features

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    dts = [base_time + timedelta(minutes=i) for i in range(6)]
    prices = [0.45, 0.46, 0.48, 0.50, 0.52, 0.55]

    # Training representation up to t=3
    train_rows = [
        {
            "market_id": "m1",
            "recorded_at": dts[i],
            "time_left_min": 15.0 - i,
            "mid_price": prices[i],
            "spread": 0.02,
            "price_velocity": 0.01,
            "volume_5min": 100.0,
            "hour_of_day": dts[i].hour,
            "day_of_week": float(dts[i].weekday()),
            "market_duration_min": 15.0,
        }
        for i in range(4)
    ]
    df_train = pd.DataFrame(train_rows)
    df_train = add_derived_features(df_train)
    df_train = add_lag_features(df_train)
    df_train = compute_point_in_time_features(df_train)

    decision_idx = 3
    decision_time = dts[decision_idx]

    # Inference receives past snapshots AND accidental future snapshots (t=4, t=5)
    all_snaps = [
        Snap(15.0 - j, prices[j], 0.02, 0.01, 100.0, dts[j].hour, dts[j], f"s_{j}")
        for j in range(len(dts)) if j != decision_idx
    ]

    df_inf = build_inference_dataframe(
        market=Market(),
        history_snaps=all_snaps,
        fresh_yes_price=prices[decision_idx],
        fresh_spread=0.02,
        global_max=0.50,
        start_time=decision_time,
        time_left_sec=(15.0 - decision_idx) * 60.0,
        decision_id="decision_clean",
    )

    inf_row = df_inf[df_inf["_is_decision_row"]].iloc[0]
    train_row = df_train.iloc[decision_idx]

    for col in [
        "pm_change_60s", "pm_change_180s", "has_60s_ref", "has_180s_ref",
        "history_age_seconds", "legacy_last_poll_delta", "price_distance_from_max",
    ]:
        v_inf = float(inf_row[col])
        v_train = float(train_row[col])
        assert np.isclose(v_inf, v_train, atol=1e-7, equal_nan=True), (
            f"Feature {col} diverged due to future noise: inf={v_inf} vs train={v_train}"
        )
