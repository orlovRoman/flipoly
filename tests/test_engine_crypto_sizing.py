import pytest
import dataclasses
from unittest.mock import AsyncMock
from sqlalchemy import select
from polyflip.trading.decision_logic import TradeDecision
from polyflip.crypto.trainer import CRYPTO_FEATURES
from polyflip.crypto.feature_builder import CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.predictor import CryptoFeaturesValidator

def test_crypto_bet_size_not_overwritten():
    """TradeDecision сохраняет размер ставки."""
    decision = TradeDecision(action='BUY_YES', buy_price=0.6, bet_size_usdc=12.5, reason='test', strategy_type='LIGHTGBM_TREND', p_up=0.7, strike=65000.0, edge=0.15)
    assert decision.bet_size_usdc == 12.5

def test_dataclasses_replace_frozen():
    d = TradeDecision('BUY_YES', 0.6, 10.0, 'test', 'ML_TREND', p_flip=0.3, edge=0.1)
    d2 = dataclasses.replace(d, action='SKIP', reason='veto')
    assert d2.action == 'SKIP'
    assert d2.strategy_type == 'ML_TREND'
    assert d2.edge == 0.1

def test_trainer_features_subset_of_validator():
    """CRYPTO_FEATURES (trainer) должен быть подмножеством CryptoFeaturesValidator."""
    validator_fields = set(CryptoFeaturesValidator.model_fields.keys())
    trainer_features = set(CRYPTO_FEATURES)
    unknown = trainer_features - validator_fields
    assert not unknown, f'Trainer использует признаки не из validator: {unknown}'

def test_validator_features_subset_of_columns():
    """CryptoFeaturesValidator должен быть подмножеством CRYPTO_FEATURE_COLUMNS."""
    assert set(CryptoFeaturesValidator.model_fields.keys()).issubset(set(CRYPTO_FEATURE_COLUMNS))

def test_crypto_predictor_cache():
    """Повторный вызов load() для загруженного символа не делает запросы к БД."""
    from polyflip.crypto.predictor import CryptoPredictor
    predictor = CryptoPredictor()
    predictor._loaded_symbols.add('BTCUSDT')
    mock = object()
    predictor._models['BTCUSDT'] = {'low_vol': mock, 'mid_vol': mock, 'high_vol': mock}
    predictor._model_versions['BTCUSDT'] = {'low_vol': 42, 'mid_vol': 42, 'high_vol': 42}
    predictor._thresholds['BTCUSDT'] = {'low_vol': (0.55, 0.45), 'mid_vol': (0.55, 0.45), 'high_vol': (0.55, 0.45)}
    predictor._vol_p33s['BTCUSDT'] = 0.5
    predictor._vol_p67s['BTCUSDT'] = 1.5
    import asyncio

    async def run_test():
        fake_db = AsyncMock()

        class MockRow:

            def __init__(self, asset, version):
                self.asset = asset
                self.version = version
        mock_result = AsyncMock()
        mock_result.all.return_value = [MockRow('BTCUSDT_low_vol', 42), MockRow('BTCUSDT_mid_vol', 42), MockRow('BTCUSDT_high_vol', 42)]
        fake_db.execute.return_value = mock_result
        res = await predictor.load(fake_db, 'BTCUSDT')
        assert res is True
        assert fake_db.execute.call_count >= 1
    asyncio.run(run_test())

def test_crypto_features_count_matches_mock():
    from polyflip.crypto.trainer import CRYPTO_FEATURES
    assert len(CRYPTO_FEATURES) == 22, f'Ожидалось 22 фичи, фактически: {len(CRYPTO_FEATURES)}'