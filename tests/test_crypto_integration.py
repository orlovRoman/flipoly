import pytest
from unittest.mock import MagicMock
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

    # 2. Создаем фейковые свечи
    class FakeCandle:
        def __init__(self, close, volume):
            self.open_time = "2026-07-04T12:00:00"
            self.open = close - 10
            self.high = close + 20
            self.low = close - 20
            self.close = close
            self.volume = volume
            self.taker_buy_volume = volume * 0.5

    # Нужно не менее 50 свечей для валидности фичей
    fake_candles = [FakeCandle(60000.0 + i, 10.0) for i in range(100)]

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
    assert decision.bet_size_usdc == 10.0
    assert decision.p_up == 0.8
    assert decision.strike == 60099.0
