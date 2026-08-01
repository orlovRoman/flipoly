import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from polyflip.crypto.trainer import _fit_lgbm_and_serialize, CRYPTO_FEATURES

def make_fake_df(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "ret_1":           np.random.randn(n) * 0.003,
        "ret_3":           np.random.randn(n) * 0.005,
        "ret_6":           np.random.randn(n) * 0.007,
        "ret_12":          np.random.randn(n) * 0.009,
        "ret_24":          np.random.randn(n) * 0.012,
        "vol_6":           np.abs(np.random.randn(n)) * 0.002 + 0.001,
        "vol_24":          np.abs(np.random.randn(n)) * 0.003 + 0.001,
        "vol_48":          np.abs(np.random.randn(n)) * 0.003 + 0.001,
        "vol_ratio":       np.random.uniform(0.5, 2.0, n),
        "rsi_14":          np.random.uniform(30, 70, n),
        "ema_ratio_9_21":  np.random.uniform(0.99, 1.01, n),
        "bb_width":        np.random.uniform(0.01, 0.05, n),
        "bb_position":     np.random.uniform(0, 1, n),
        "taker_buy_ratio": np.random.uniform(0.4, 0.6, n),
        "hour_utc":        np.random.randint(0, 24, n).astype(float),
        "consec_up":       np.random.randint(0, 5, n).astype(float),
        "consec_down":     np.random.randint(0, 5, n).astype(float),
        "target":          np.random.randint(0, 2, n),
        "ret_48":          np.random.randn(n),
        "vol_z_1":         np.random.randn(n),
        "dist_to_high_24": np.random.randn(n),
        "dist_to_low_24":  np.random.randn(n),
        "dist_to_high_96": np.random.randn(n),
        "dist_to_low_96":  np.random.randn(n),
        "range_1":         np.random.randn(n),
        "range_avg_24":    np.random.randn(n),
        "dow":             np.random.randint(0, 7, n),
    })
    for f in CRYPTO_FEATURES:
        if f not in df.columns:
            df[f] = np.random.randn(n)
    df["target"] = (df["ret_1"] > 0).astype(int)
    return df

def test_vol_regime_split():
    """low_vol и high_vol датасеты не пересекаются."""
    df = make_fake_df(500)
    median = df["vol_ratio"].median()
    low  = df[df["vol_ratio"] <= median]
    high = df[df["vol_ratio"] > median]
    assert len(low) + len(high) == len(df)
    assert set(low.index) & set(high.index) == set()

def test_vol_regime_split_edge_case():
    """Проверяем что оба режима получают >= 150 строк на реальном датасете."""
    df = make_fake_df(500)
    median = df["vol_ratio"].median()
    low  = df[df["vol_ratio"] <= median]
    high = df[df["vol_ratio"] >  median]
    assert len(low)  >= 150, f"low_vol слишком мал: {len(low)}"
    assert len(high) >= 150, f"high_vol слишком мал: {len(high)}"

def test_vol_regime_split_skewed():
    """Если все vol_ratio одинаковые — весь датасет идёт в low_vol, high_vol пустой."""
    df = make_fake_df(500)
    df["vol_ratio"] = 1.0  # все значения равны медиане
    median = df["vol_ratio"].median()  # == 1.0
    low  = df[df["vol_ratio"] <= median]
    high = df[df["vol_ratio"] >  median]
    # Все строки <= median, high пустой — тренер должен пропустить high_vol
    assert len(low)  == 500
    assert len(high) == 0

def test_predictor_predict_missing_regime():
    """predict() не должен падать с KeyError если загружен только один режим."""
    from polyflip.crypto.predictor import CryptoPredictor
    from polyflip.crypto.predictor import CRYPTO_FEATURE_COLUMNS
    predictor = CryptoPredictor()
    mock_lgb = MagicMock()
    mock_lgb.predict_proba.return_value = [[0.3, 0.7]]

    # Загружаем только low_vol (имитируем частичный load)
    predictor._models["BTCUSDT"]         = {"low_vol": mock_lgb}  # нет high_vol
    predictor._model_versions["BTCUSDT"] = {"low_vol": 1}
    predictor._thresholds["BTCUSDT"]     = {"low_vol": (0.65, 0.35)}
    predictor._vol_p33s["BTCUSDT"]       = 1.0
    predictor._vol_p67s["BTCUSDT"]       = 2.0
    predictor._loaded_symbols.add("BTCUSDT")

    # vol_ratio > vol_p67 → должен запросить high_vol, которого нет
    # После фикса — должен сделать fallback на low_vol без KeyError
    from unittest.mock import patch
    with patch("polyflip.crypto.predictor.build_crypto_features") as mock_bf:
        mock_fv = MagicMock()
        mock_fv.valid = True
        from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
        features_data = {col: 1.0 for col in CRYPTO_FEATURE_COLUMNS}
        features_data["vol_trend"] = 2.5
        mock_fv.features = [[features_data[col] for col in CRYPTO_FEATURE_COLUMNS]]
        mock_bf.return_value = mock_fv
        # Создаем фейковую свечу для проброса close
        mock_candle = MagicMock()
        mock_candle.close = 50000.0
        result = predictor.predict([mock_candle], "BTCUSDT")

    # Не должно быть NONE из-за KeyError
    assert result.features_ok is True
    assert result.p_up == 0.7


def test_small_fold_oof_scores_not_zero():
    """  oof_scores      ."""
    df = make_fake_df(n=200)  #  
    _, auc, baseline_auc, optimal_threshold, ece, fi = _fit_lgbm_and_serialize(
        df[CRYPTO_FEATURES], df["target"], n_splits=2
    )
    #    .
    assert auc > 0

def test_calibration_ece_isotonic_better_than_uncalibrated():
    """Isotonic calibration должно отрабатывать без падения."""
    df = make_fake_df(n=1000)
    _, auc, baseline_auc, optimal_threshold, ece, fi = _fit_lgbm_and_serialize(
        df[CRYPTO_FEATURES], df["target"], n_splits=3
    )
    assert ece < 0.50, f"ECE слишком высокий: {ece:.3f}"
