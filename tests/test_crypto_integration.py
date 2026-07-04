import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from polyflip.crypto.predictor import CryptoPredictor, CryptoSignal
from polyflip.trading.decision_logic import decide_crypto_trend

@pytest.mark.asyncio
async def test_crypto_predictor_flow():
    # 1. Мокаем сессию базы данных и модель
    db_mock = MagicMock()
    
    predictor = CryptoPredictor()
    
    # Мокаем саму обученную модель LightGBM
    mock_lgb = MagicMock()
    # predict_proba должна возвращать массив вида [[p_down, p_up]]
    mock_lgb.predict_proba.return_value = [[0.2, 0.8]]
    
    predictor._model = mock_lgb
    predictor._model_version = 42
    predictor._thresholds["BTCUSDT"] = (0.65, 0.35)
    predictor._loaded_symbols.add("BTCUSDT")

    # 2. Мокаем сборку фичей build_crypto_features, возвращая полностью валидные 26 полей
    mock_features = MagicMock()
    mock_features.valid = True
    # 26 значений (все поля Validator заполнены)
    mock_features.features = [np.array([0.01]*26)]

    # Создаем фейковую свечу для страйка
    class FakeCandle:
        def __init__(self):
            self.close = 60099.0

    fake_candles = [FakeCandle()]

    with patch("polyflip.crypto.predictor.build_crypto_features", return_value=mock_features):
        # 3. Вызываем predict
        signal = predictor.predict(fake_candles, "BTCUSDT")
        
    assert signal.features_ok is True
    assert signal.p_up == 0.8
    assert signal.direction == "UP"
    assert signal.edge == pytest.approx(0.15)
    assert signal.strike == 60099.0

    # 4. Проверяем принятие решения
    config = {
        "CRYPTO_MIN_EDGE": 0.05,
        "MAX_EDGE_FILTER": 0.40,
        "TRADE_BET_SIZE_USDC": 10.0,
        "MAX_BET_SIZE_USDC": 100.0,
        "LIQUIDITY_FRACTION": 0.05,
        "BYPASS_BET_SIZE_CHECK": "true"
    }
    
    decision = decide_crypto_trend(signal, entry_price=0.65, volume_5min=5000.0, config=config)
    assert decision.action == "BUY_YES"
    assert decision.strategy_type == "CRYPTO_TREND"
    assert decision.bet_size_usdc == 70.0
    assert decision.p_up == 0.8
    assert decision.strike == 60099.0
