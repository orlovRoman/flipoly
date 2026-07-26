import pytest
from unittest.mock import Mock, patch
from polyflip.trading.decision_logic import decide_crypto_trend
from polyflip.crypto.predictor import CryptoSignal

@pytest.mark.asyncio
async def test_symmetric_flip_rules():
    # 3.4. Написать параметризованный тест: YES фаворит + UP / DOWN, NO фаворит + UP / DOWN
    
    config = {
        "NO_FLIP_THRESHOLD": "0.35",
        "FLIP_THRESHOLD": "0.60",
        "MIN_EDGE": "0.05",
    }
    
    def make_signal(direction):
        return CryptoSignal(
            symbol="BTCUSDT",
            p_up=0.8 if direction == "UP" else 0.2,
            p_down=0.2 if direction == "UP" else 0.8,
            direction=direction,
            signal_strength=0.1,
            strike=60000.0,
            threshold_up=0.55,
            threshold_down=0.45,
            model_version=1,
            features_ok=True
        )
    
    # 1. YES фаворит (entry_price >= 0.5)
    # Покупка фаворита требует низкой вероятности разворота (p_flip_ml < 0.35)
    sig_up_ok = make_signal("UP")
    
    # YES фаворит + UP (Buying YES Fav)
    # Valid
    dec = decide_crypto_trend(sig_up_ok, 0.55, 1000.0, config, p_flip_ml=0.20)
    assert dec.action == "BUY_YES"
    # Blocked by high flip prob (>= 0.35)
    dec = decide_crypto_trend(sig_up_ok, 0.55, 1000.0, config, p_flip_ml=0.40)
    assert dec.action == "SKIP"
    assert "Fav trade blocked" in dec.reason

    # YES фаворит + DOWN (Buying NO Outsider)
    # Покупка аутсайдера требует высокой вероятности разворота (p_flip_ml >= 0.60)
    sig_down_ok = make_signal("DOWN")
    # Valid
    dec = decide_crypto_trend(sig_down_ok, 0.55, 1000.0, config, p_flip_ml=0.65, no_ask=0.5)
    assert dec.action == "BUY_NO"
    # Blocked by low flip prob (< 0.60)
    dec = decide_crypto_trend(sig_down_ok, 0.55, 1000.0, config, p_flip_ml=0.50, no_ask=0.5)
    assert dec.action == "SKIP"
    assert "Outsider trade blocked" in dec.reason

    # 2. NO фаворит (entry_price < 0.5)
    # Покупка фаворита (NO) (когда направление DOWN) требует низкой вероятности разворота (p_flip_ml < 0.35)
    # NO фаворит + DOWN (Buying NO Fav)
    # Valid
    dec = decide_crypto_trend(sig_down_ok, 0.45, 1000.0, config, p_flip_ml=0.20, no_ask=0.6)
    assert dec.action == "BUY_NO"
    # Blocked
    dec = decide_crypto_trend(sig_down_ok, 0.45, 1000.0, config, p_flip_ml=0.40, no_ask=0.6)
    assert dec.action == "SKIP"
    assert "Fav trade blocked" in dec.reason

    # NO фаворит + UP (Buying YES Outsider)
    # Покупка аутсайдера (YES) требует высокой вероятности разворота (p_flip_ml >= 0.60)
    # Valid
    dec = decide_crypto_trend(sig_up_ok, 0.45, 1000.0, config, p_flip_ml=0.65)
    assert dec.action == "BUY_YES"
    # Blocked
    dec = decide_crypto_trend(sig_up_ok, 0.45, 1000.0, config, p_flip_ml=0.50)
    assert dec.action == "SKIP"
    assert "Outsider trade blocked" in dec.reason

    # Проверка, что при ошибке p_flip_ml (когда оно None) блокируются обе стороны
    dec_none_up = decide_crypto_trend(sig_up_ok, 0.55, 1000.0, config, p_flip_ml=None)
    assert dec_none_up.action == "SKIP"
    assert "not provided" in dec_none_up.reason

    dec_none_down = decide_crypto_trend(sig_down_ok, 0.55, 1000.0, config, p_flip_ml=None)
    assert dec_none_down.action == "SKIP"
    assert "not provided" in dec_none_down.reason
