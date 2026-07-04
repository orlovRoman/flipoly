import pytest
import dataclasses
from polyflip.trading.decision_logic import TradeDecision
from polyflip.crypto.trainer import CRYPTO_FEATURES
from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.predictor import CryptoFeaturesValidator

def test_crypto_bet_size_not_overwritten():
    """TradeDecision сохраняет размер ставки."""
    decision = TradeDecision(
        action="BUY_YES", buy_price=0.6, bet_size_usdc=12.5,
        reason="test", strategy_type="CRYPTO_TREND",
        p_up=0.7, strike=65000.0, edge=0.15
    )
    assert decision.bet_size_usdc == 12.5

def test_dataclasses_replace_frozen():
    d = TradeDecision("BUY_YES", 0.6, 10.0, "test", "ML_TREND", p_flip=0.3, edge=0.1)
    d2 = dataclasses.replace(d, action="SKIP", reason="veto")
    assert d2.action == "SKIP"
    assert d2.strategy_type == "ML_TREND"
    assert d2.edge == 0.1

def test_crypto_features_import():
    assert isinstance(CRYPTO_FEATURES, list)
    assert len(CRYPTO_FEATURES) == 17   # В trainer.py определено 17 признаков
    assert "rsi_14" in CRYPTO_FEATURES

def test_feature_columns_match_validator():
    """CRYPTO_FEATURE_COLUMNS должен точно совпадать с полями CryptoFeaturesValidator."""
    validator_fields = set(CryptoFeaturesValidator.model_fields.keys())
    feature_columns = set(CRYPTO_FEATURE_COLUMNS)
    
    missing_in_validator = feature_columns - validator_fields
    missing_in_columns = validator_fields - feature_columns
    
    assert not missing_in_validator, f"В CRYPTO_FEATURE_COLUMNS нет в Validator: {missing_in_validator}"
    assert not missing_in_columns, f"В Validator нет в CRYPTO_FEATURE_COLUMNS: {missing_in_columns}"


def test_crypto_predictor_cache():
    """Повторный вызов load() для загруженного символа не делает запросы к БД."""
    from polyflip.crypto.predictor import CryptoPredictor
    predictor = CryptoPredictor()
    predictor._loaded_symbols.add("BTCUSDT")
    predictor._model = object()  # mock
    predictor._thresholds["BTCUSDT"] = (0.55, 0.45)
    
    # Проверяем, что load() вернет True без SQL-сессии
    import asyncio
    async def run_test():
        res = await predictor.load(None, "BTCUSDT")
        assert res is True
        
    asyncio.run(run_test())

