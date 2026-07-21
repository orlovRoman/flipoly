from polyflip.crypto.feature_builder import build_features, build_crypto_features
import pandas as pd
import numpy as np


def test_funding_extreme_computation():
    """Проверяет корректность расчета индикатора funding_extreme (> 0.001 / 0.1%)."""
    candles = pd.DataFrame({
        'open_time': pd.date_range('2026-01-01', periods=100, freq='5min'),
        'open': [100.0]*100,
        'high': [101.0]*100,
        'low': [99.0]*100,
        'close': [100.5]*100,
        'volume': [1000.0]*100,
        'taker_buy_volume': [500.0]*100,
        'quote_volume': [100500.0]*100
    })


    # При экстремальной ставке 0.002 (0.2%) funding_extreme должен равен 1.0
    df = build_features(candles, funding_rate=0.002, funding_rate_ma3=0.0015)
    assert df['funding_extreme'].iloc[-1] == 1.0, \
        f"funding_extreme не вычислен корректно: {df['funding_extreme'].iloc[-1]}"

    # При нормальной ставке 0.0001 (0.01%) funding_extreme должен равен 0.0
    df_normal = build_features(candles, funding_rate=0.0001, funding_rate_ma3=0.0001)
    assert df_normal['funding_extreme'].iloc[-1] == 0.0

    # Проверка для build_crypto_features (инференс)
    feat_obj = build_crypto_features(candles, funding_rate=0.002, funding_rate_ma3=0.0015)
    assert feat_obj.features.shape[1] > 20
    assert feat_obj.features[0, -1] == 1.0  # funding_extreme - последняя фича

