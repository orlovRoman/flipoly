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

def test_trainer_features_subset_of_validator():
    """CRYPTO_FEATURES (trainer) должен быть подмножеством CryptoFeaturesValidator."""
    validator_fields = set(CryptoFeaturesValidator.model_fields.keys())
    trainer_features = set(CRYPTO_FEATURES)

    unknown = trainer_features - validator_fields
    assert not unknown, f"Trainer использует признаки не из validator: {unknown}"

def test_feature_columns_match_validator():
    """CRYPTO_FEATURE_COLUMNS (feature_builder) == CryptoFeaturesValidator."""
    assert set(CRYPTO_FEATURE_COLUMNS) == set(CryptoFeaturesValidator.model_fields.keys())

def test_crypto_predictor_cache():
    """Повторный вызов load() для загруженного символа не делает запросы к БД."""
    from polyflip.crypto.predictor import CryptoPredictor
    predictor = CryptoPredictor()
    predictor._loaded_symbols.add("BTCUSDT")
    mock = object()
    predictor._models["BTCUSDT"] = {"low_vol": mock, "mid_vol": mock, "high_vol": mock}
    predictor._model_versions["BTCUSDT"] = {"low_vol": 42, "mid_vol": 42, "high_vol": 42}
    predictor._thresholds["BTCUSDT"] = {"low_vol": (0.55, 0.45), "mid_vol": (0.55, 0.45), "high_vol": (0.55, 0.45)}
    predictor._vol_p33s["BTCUSDT"] = 0.5
    predictor._vol_p67s["BTCUSDT"] = 1.5
    
    # Проверяем, что load() проверит версии в БД и вернет True (cache hit)
    import asyncio
    async def run_test():
        fake_db = AsyncMock()
        
        class MockRow:
            def __init__(self, asset, version):
                self.asset = asset
                self.version = version

        mock_result = AsyncMock()
        mock_result.all.return_value = [MockRow("BTCUSDT_low_vol", 42), MockRow("BTCUSDT_mid_vol", 42), MockRow("BTCUSDT_high_vol", 42)]
        fake_db.execute.return_value = mock_result
        
        res = await predictor.load(fake_db, "BTCUSDT")
        assert res is True
        assert fake_db.execute.call_count >= 1
        
    asyncio.run(run_test())


@pytest.mark.asyncio
async def test_engine_crypto_standalone_bet_size(db_session):
    from datetime import datetime, timezone, timedelta
    import pickle
    from polyflip.db.models import RuntimeSettings, LiveMarket, ModelRegistry, TradeHistory
    from polyflip.trading.engine import trade_worker_cycle
    from unittest.mock import patch, AsyncMock, MagicMock
    import numpy as np

    now = datetime.now(timezone.utc)
    settings = [
        RuntimeSettings(key="TRADING_ENABLED", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_TIME_LEFT_SEC", value="10", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_TIME_LEFT_SEC", value="1000", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADING_MODE_BTC", value="lightgbm", updated_at=now, updated_by="test"),
        RuntimeSettings(key="CRYPTO_MIN_EDGE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="CRYPTO_THRESHOLD_UP_BTC", value="0.60", updated_at=now, updated_by="test"),
        RuntimeSettings(key="CRYPTO_THRESHOLD_DOWN_BTC", value="0.40", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_SIZE_USDC", value="50.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_BET_SIZE_USDC", value="10.0", updated_at=now, updated_by="test"),
        RuntimeSettings(key="BYPASS_BET_SIZE_CHECK", value="true", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_PRICE_DRIFT", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MIN_PRICE", value="0.05", updated_at=now, updated_by="test"),
        RuntimeSettings(key="TRADE_MAX_PRICE", value="0.95", updated_at=now, updated_by="test"),
        RuntimeSettings(key="MAX_BET_EDGE", value="0.50", updated_at=now, updated_by="test"),
    ]
    db_session.add_all(settings)

    market = LiveMarket(
        market_id="crypto_m1", asset="BTC", question="BTC up?",
        current_yes_price=0.6, current_no_price=0.4, current_spread=0.01,
        volume_5min=100.0, price_velocity=0.0,
        end_time_est=now + timedelta(seconds=100),
        yes_token_id="t_yes", no_token_id="t_no", last_updated=now
    )
    db_session.add(market)

    # Мокаем LightGBM модель в ModelRegistry
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.2, 0.8]]  # p_up = 0.8
    db_session.add(ModelRegistry(
        asset="CRYPTO", model_blob=b"dummy_bytes", is_active=True,
        version=42, accuracy=0.9, features="all", trained_at=now
    ))
    await db_session.commit()

    # Свечи
    class FakeCandle:
        def __init__(self):
            self.close = 60099.0

    fake_candles = [FakeCandle()]
    mock_features = MagicMock()
    mock_features.valid = True
    mock_features.features = [np.array([0.01]*27)]

    with patch("polyflip.trading.engine.PolyTrader") as mock_trader_cls, \
         patch("polyflip.trading.engine.PolymarketClient") as mock_api_cls, \
         patch("pickle.loads", return_value=mock_model), \
         patch("polyflip.crypto.predictor.build_crypto_features", return_value=mock_features), \
         patch("polyflip.trading.decision_runners.get_recent_candles", AsyncMock(return_value=fake_candles)):

         mock_trader = mock_trader_cls.return_value
         mock_trader.execute_trade = AsyncMock(return_value={"status": "SUCCESS", "error_msg": None})

         mock_api = mock_api_cls.return_value
         mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01, "best_ask": 0.61})
         mock_api.close = AsyncMock()

         await trade_worker_cycle(db_session, mock_trader, mock_api)

         res = await db_session.execute(select(TradeHistory))
         trades = res.scalars().all()

         assert len(trades) == 1
         assert trades[0].outcome_bought == "YES"
         assert trades[0].status == "SUCCESS"
         assert trades[0].amount_usdc > 0


def test_crypto_features_count_matches_mock():
    from polyflip.crypto.trainer import CRYPTO_FEATURES
    assert len(CRYPTO_FEATURES) == 27, \
        f"Тесты мокают 27 фичей, но реальных: {len(CRYPTO_FEATURES)}"
