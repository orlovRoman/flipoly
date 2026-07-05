def test_decide_crypto_trend_respects_max_bet_edge():
    """MAX_BET_EDGE из config должен блокировать слишком высокий edge."""
    from polyflip.crypto.predictor import CryptoSignal
    from polyflip.trading.decision_logic import decide_crypto_trend

    sig = CryptoSignal(
        symbol="BTCUSDT", p_up=0.92, p_down=0.08,
        direction="UP", edge=0.50,   # очень высокий edge
        strike=60000.0, threshold_up=0.55, threshold_down=0.45,
        model_version=1, features_ok=True,
    )
    config = {"CRYPTO_MIN_EDGE": 0.01, "MAX_BET_EDGE": "0.30"}  # строковое значение из RuntimeSettings
    decision = decide_crypto_trend(sig, entry_price=0.65, volume_5min=1000.0, config=config)
    assert decision.action == "SKIP", f"Должен быть SKIP, но action={decision.action}"
    assert "suspicious" in decision.reason
