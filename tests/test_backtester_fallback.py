def test_fallback_when_no_high_vol_model():
    """Если high_vol не обучилась — low_vol применяется к high_mask свечам."""
    import numpy as np, pandas as pd
    from datetime import datetime, timedelta, timezone
    from polyflip.crypto.feature_builder import build_features
    from polyflip.crypto.backtester import run_backtest

    # Делаем датасет где ВЕСЬ train — low_vol, чтобы high_vol не обучился
    n = 600
    np.random.seed(7)
    base = 50000 + np.cumsum(np.random.randn(n) * 50)   # маленькая волатильность
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "open_time":        [t0 + timedelta(minutes=15*i) for i in range(n)],
        "open":             base,
        "high":             base * 1.0005,
        "low":              base * 0.9995,
        "close":            base,
        "volume":           np.ones(n) * 10,
        "taker_buy_volume": np.ones(n) * 5,
    })
    features_df = build_features(df)
    result = run_backtest(features_df, symbol="BTCUSDT")
    # Не должно падать и должен вернуться BacktestResult
    assert result is not None
    assert 0.0 <= result.win_rate <= 1.0
