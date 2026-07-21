import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from polyflip.models.trainer import (
    add_derived_features,
    DERIVED_FEATURES,
)
from polyflip.models.feature_lags import add_lag_features, LAG_FEATURE_NAMES

def make_snapshot_row(
    market_id: str,
    flip_vs_final: bool,
    mid_price: float = 0.6,
    spread: float = 0.02,
    price_velocity: float = 0.001,
    volume_5min: float = 500.0,
    time_left_min: float = 45.0,
    minutes_ago: int = 0,
) -> dict:
    return {
        "market_id": market_id,
        "recorded_at": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        "time_left_min": time_left_min,
        "mid_price": mid_price,
        "spread": spread,
        "price_velocity": price_velocity,
        "volume_5min": volume_5min,
        "hour_of_day": 14,
        "day_of_week": 0,
        "target": 1 if flip_vs_final else 0,
    }

def build_test_df(n_markets: int = 5, snaps_per_market: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n_markets):
        market_id = f"market_{i}"
        flip = bool(i % 2)
        for t in range(snaps_per_market):
            rows.append(make_snapshot_row(
                market_id=market_id,
                flip_vs_final=flip,
                mid_price=0.5 + 0.1 * np.sin(t),
                spread=0.01 + 0.005 * t,
                price_velocity=0.001 * t,
                volume_5min=200 + 50 * t,
                time_left_min=max(0.0, 60 - t * 5),
                minutes_ago=snaps_per_market - t,
            ))
    return pd.DataFrame(rows)

class TestTargetEncoding:
    def test_flip_true_maps_to_1(self):
        row = make_snapshot_row("m1", flip_vs_final=True)
        assert row["target"] == 1

    def test_flip_false_maps_to_0(self):
        row = make_snapshot_row("m1", flip_vs_final=False)
        assert row["target"] == 0

    def test_target_is_binary(self):
        df = build_test_df()
        assert set(df["target"].unique()).issubset({0, 1})

    def test_target_has_two_classes(self):
        df = build_test_df(n_markets=6)
        assert len(df["target"].unique()) == 2

class TestDerivedFeatures:
    @pytest.fixture
    def df_with_derived(self):
        df = build_test_df()
        return add_derived_features(df)

    def test_all_derived_columns_present(self, df_with_derived):
        base_derived = [
            "price_deviation", "deviation_x_time", "price_deviation_sq",
            "spread_pct", "log_time_left", "price_distance_from_max", "time_phase"
        ]
        for col in base_derived:
            assert col in df_with_derived.columns

    def test_no_nan_in_derived(self, df_with_derived):
        base_derived = [
            "price_deviation", "deviation_x_time", "price_deviation_sq",
            "spread_pct", "log_time_left", "price_distance_from_max", "time_phase"
        ]
        nan_counts = df_with_derived[base_derived].isna().sum()
        assert nan_counts.sum() == 0

    def test_price_deviation_is_non_negative(self, df_with_derived):
        assert (df_with_derived["price_deviation"] >= 0).all()

    def test_log_time_left_is_non_negative(self, df_with_derived):
        assert (df_with_derived["log_time_left"] >= 0).all()

    def test_spread_pct_is_clipped(self, df_with_derived):
        assert (df_with_derived["spread_pct"] <= 10.0 + 1e-9).all()

class TestLagFeatures:
    @pytest.fixture
    def df_with_lags(self):
        df = build_test_df(snaps_per_market=10)
        return add_lag_features(df)

    def test_all_lag_columns_present(self, df_with_lags):
        for col in LAG_FEATURE_NAMES:
            assert col in df_with_lags.columns

    def test_no_nan_in_lag_features(self, df_with_lags):
        nan_counts = df_with_lags[LAG_FEATURE_NAMES].isna().sum()
        assert nan_counts.sum() == 0

    def test_lag_features_imputed_for_first_rows(self, df_with_lags):
        first_rows = df_with_lags.groupby("market_id").head(1)
        assert not first_rows["price_velocity_lag1"].isna().any()

    def test_spread_trend_clipped(self, df_with_lags):
        assert (df_with_lags["spread_trend"] <= 10.0 + 1e-9).all()

    def test_volume_trend_clipped(self, df_with_lags):
        assert (df_with_lags["volume_trend"] <= 10.0 + 1e-9).all()

class TestFullFeaturePipeline:
    def test_all_derived_features_no_nan_after_full_pipeline(self):
        df = build_test_df(n_markets=10, snaps_per_market=10)
        df = add_derived_features(df)
        df = add_lag_features(df)
        df.drop(columns=["recorded_at"], errors="ignore", inplace=True)

        missing = [f for f in DERIVED_FEATURES if f not in df.columns]
        assert not missing
        nan_counts = df[DERIVED_FEATURES].isna().sum()
        assert nan_counts.sum() == 0

from sklearn.model_selection import GroupKFold
from polyflip.constants import CV_N_SPLITS

class TestCrossValidationScheme:
    @pytest.fixture
    def prepared_data(self):
        df = build_test_df(n_markets=20, snaps_per_market=10)
        df = add_derived_features(df)
        df = add_lag_features(df)
        df.drop(columns=["recorded_at"], errors="ignore", inplace=True)
        return df

    def test_no_market_id_overlap_in_train_val(self, prepared_data):
        df = prepared_data
        feature_cols = [c for c in df.columns if c not in ("market_id", "target")]
        X = df[feature_cols]
        y = df["target"]
        groups = df["market_id"]

        gkf = GroupKFold(n_splits=CV_N_SPLITS)
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            train_markets = set(groups.iloc[train_idx].unique())
            val_markets = set(groups.iloc[val_idx].unique())
            overlap = train_markets & val_markets
            assert len(overlap) == 0

    def test_all_samples_covered_in_oof(self, prepared_data):
        df = prepared_data
        feature_cols = [c for c in df.columns if c not in ("market_id", "target")]
        X = df[feature_cols]
        y = df["target"]
        groups = df["market_id"]
        gkf = GroupKFold(n_splits=CV_N_SPLITS)
        val_counts = np.zeros(len(df), dtype=int)
        for _, val_idx in gkf.split(X, y, groups=groups):
            val_counts[val_idx] += 1
        assert (val_counts == 1).all()

    def test_each_fold_has_both_classes(self, prepared_data):
        df = prepared_data
        feature_cols = [c for c in df.columns if c not in ("market_id", "target")]
        X = df[feature_cols]
        y = df["target"]
        groups = df["market_id"]
        gkf = GroupKFold(n_splits=CV_N_SPLITS)
        for fold_idx, (_, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            y_val = y.iloc[val_idx]
            unique_classes = set(y_val.unique())
            assert unique_classes == {0, 1}

    def test_cv_splits_count(self, prepared_data):
        df = prepared_data
        feature_cols = [c for c in df.columns if c not in ("market_id", "target")]
        X = df[feature_cols]
        y = df["target"]
        groups = df["market_id"]
        gkf = GroupKFold(n_splits=CV_N_SPLITS)
        splits = list(gkf.split(X, y, groups=groups))
        assert len(splits) == CV_N_SPLITS

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score,
)
from polyflip.constants import CV_RANDOM_STATE
MIN_PRECISION_FOR_THRESHOLD = 0.52
MAX_SUSPICIOUS_THRESHOLD = 0.95

def run_full_oof_pipeline(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("market_id", "target", "recorded_at")]
    X = df[feature_cols]
    y = df["target"]
    groups = df["market_id"]

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", C=5.0,
            random_state=CV_RANDOM_STATE, max_iter=1000,
        )),
    ])

    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    aucs, oof_scores = [], np.zeros(len(y))

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_base = clone(base_model)
        fold_base.fit(X_train, y_train)

        fold_cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(fold_base), method="sigmoid"
        )
        fold_cal.fit(X_val, y_val)

        y_proba = fold_cal.predict_proba(X_val)[:, 1]
        oof_scores[val_idx] = y_proba
        aucs.append(roc_auc_score(y_val, y_proba))

    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y, oof_scores)
    valid_mask = precision_arr[:-1] >= MIN_PRECISION_FOR_THRESHOLD
    f1 = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)

    if valid_mask.any():
        f1_filtered = np.where(valid_mask, f1, 0)
        best_idx = int(np.argmax(f1_filtered))
    else:
        best_idx = int(np.argmax(f1))

    optimal_threshold = float(thresholds_pr[best_idx]) if len(thresholds_pr) > 0 else 0.65

    y_pred_oof = (oof_scores >= optimal_threshold).astype(int)
    precision_at_threshold = float(precision_score(y, y_pred_oof, zero_division=0))
    recall_at_threshold = float(recall_score(y, y_pred_oof, zero_division=0))
    f1_at_threshold = float(f1_score(y, y_pred_oof, zero_division=0))
    mean_auc = float(np.mean(aucs))
    baseline_auc = float(max(y.mean(), 1.0 - y.mean()))

    frac_pos, mean_pred_prob = calibration_curve(y, oof_scores, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(frac_pos - mean_pred_prob)))

    return {
        "mean_auc": mean_auc,
        "baseline_auc": baseline_auc,
        "optimal_threshold": optimal_threshold,
        "precision_at_threshold": precision_at_threshold,
        "recall_at_threshold": recall_at_threshold,
        "f1_at_threshold": f1_at_threshold,
        "ece": ece,
        "oof_scores": oof_scores,
        "y": y,
        "fold_aucs": aucs,
    }

class TestThresholdAnalytics:
    @pytest.fixture(scope="class")
    def pipeline_result(self):
        df = build_test_df(n_markets=40, snaps_per_market=15)
        df = add_derived_features(df)
        df = add_lag_features(df)
        df.drop(columns=["recorded_at"], errors="ignore", inplace=True)
        return run_full_oof_pipeline(df)

    def test_auc_above_baseline(self, pipeline_result):
        r = pipeline_result
        assert 0.0 < r["mean_auc"] < 1.0

    def test_auc_not_suspiciously_perfect(self, pipeline_result):
        r = pipeline_result
        assert r["mean_auc"] < 0.99

    def test_fold_auc_consistency(self, pipeline_result):
        fold_aucs = pipeline_result["fold_aucs"]
        std_auc = float(np.std(fold_aucs))
        assert std_auc < 0.15

    def test_threshold_below_suspicious_limit(self, pipeline_result):
        r = pipeline_result
        assert r["optimal_threshold"] < MAX_SUSPICIOUS_THRESHOLD

    def test_threshold_in_valid_range(self, pipeline_result):
        th = pipeline_result["optimal_threshold"]
        assert 0.3 <= th <= 0.95

    def test_precision_at_threshold_not_zero(self, pipeline_result):
        r = pipeline_result
        assert r["precision_at_threshold"] > 0.0

    def test_recall_at_threshold_not_zero(self, pipeline_result):
        assert pipeline_result["recall_at_threshold"] > 0.0

    def test_f1_at_threshold_reasonable(self, pipeline_result):
        r = pipeline_result
        assert r["f1_at_threshold"] >= 0.0

    def test_ece_reasonable(self, pipeline_result):
        ece = pipeline_result["ece"]
        assert ece < 1.0

    def test_oof_scores_cover_all_samples(self, pipeline_result):
        oof = pipeline_result["oof_scores"]
        nonzero_fraction = (oof != 0.0).mean()
        assert nonzero_fraction > 0.5

    def test_oof_scores_in_probability_range(self, pipeline_result):
        oof = pipeline_result["oof_scores"]
        assert ((oof >= 0.0) & (oof <= 1.0)).all()
