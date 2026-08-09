"""
tests/crypto/test_predictor_canonical_guard.py

Тесты проверок CryptoPredictor:
  - Игнорирование устаревших моделей без target_source == "POLYMARKET_FINAL_OUTCOME".
  - Использование underlying_price в качестве страйка вместо candles[-1].open.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from polyflip.crypto.predictor import CryptoPredictor
from polyflip.db.models import ModelRegistry


@pytest.mark.asyncio
async def test_predictor_ignores_legacy_binance_model():
    db = AsyncMock()
    legacy_model = ModelRegistry(
        asset="BTCUSDT_low_vol",
        version=1,
        model_blob=b"fake",
        accuracy=0.6,
        is_active=True,
        decision_threshold=0.55,
        decision_threshold_down=0.45,
        training_params={"target_source": "BINANCE_RET_1"},  # устаревший таргет
    )
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = [legacy_model]
    db.execute.return_value = mock_res

    predictor = CryptoPredictor()
    await predictor.load(db, "BTCUSDT")

    # Модель с синтетическим таргетом Binance должна быть проигнорирована
    assert "low_vol" not in predictor._models.get("BTCUSDT", {})


def test_predictor_uses_underlying_price_as_strike():
    predictor = CryptoPredictor()
    predictor._loaded_symbols.add("BTCUSDT")
    predictor._vol_p33s["BTCUSDT"] = 0.8
    predictor._vol_p67s["BTCUSDT"] = 1.2
    
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    predictor._models["BTCUSDT"] = {"low_vol": mock_model, "mid_vol": mock_model, "high_vol": mock_model}
    predictor._model_versions["BTCUSDT"] = {"low_vol": 1, "mid_vol": 1, "high_vol": 1}
    predictor._thresholds["BTCUSDT"] = {"low_vol": (0.55, 0.45), "mid_vol": (0.55, 0.45), "high_vol": (0.55, 0.45)}

    fake_candle = MagicMock()
    fake_candle.open = 10000.0  # Не должно использоваться для strike!
    fake_candle.close = 10000.0
    fake_candle.high = 10000.0
    fake_candle.low = 10000.0
    fake_candle.volume = 100.0
    fake_candle.taker_buy_volume = 50.0

    # Вызываем predict с явным underlying_price=65123.45
    with pytest.MonkeyPatch.context() as m:
        m.setattr("polyflip.crypto.predictor.build_crypto_features", lambda c: MagicMock(valid=True, features=[[0.0]*22]))
        m.setattr("polyflip.crypto.predictor.CryptoFeaturesValidator", lambda **kw: MagicMock(**{f: 0.0 for f in ["ret_1", "ret_3", "ret_6", "vol_6", "vol_24", "vol_z_1", "taker_buy_ratio", "cvd_1", "cvd_6", "rsi_14", "ema_ratio_9_21", "bb_width", "bb_position", "dist_to_high_24", "dist_to_low_24", "range_1", "range_avg_24", "consec_balance", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]}))
        
        sig = predictor.predict([fake_candle]*100, "BTCUSDT", underlying_price=65123.45)
        assert sig.strike == 65123.45
