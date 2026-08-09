"""
tests/crypto/test_market_direction_signal.py

Тесты сервиса зафиксированных сигналов LightGBM на 15-минутный рынок (market_direction_signals).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from polyflip.crypto.market_direction_service import get_or_create_market_direction_signal
from polyflip.crypto.predictor import CryptoSignal, CryptoPredictor
from polyflip.db.models import MarketDirectionSignal


@pytest.mark.asyncio
async def test_get_or_create_market_direction_signal_reuses_existing_record():
    db = AsyncMock()
    existing = MarketDirectionSignal(
        market_id="m100",
        asset="BTC",
        symbol="BTCUSDT",
        regime="mid_vol",
        direction="UP",
        p_up=0.75,
        p_down=0.25,
        signal_strength=0.5,
        strike=65000.0,
        threshold_up=0.55,
        threshold_down=0.45,
        model_key="BTCUSDT_mid_vol",
        model_version=10,
        features_ok=True,
        risk_vetoed=False,
        status="READY",
    )
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = existing
    db.execute.return_value = mock_res

    market = MagicMock()
    market.market_id = "m100"
    market.asset = "BTC"
    market.underlying_price = 65000.0

    predictor = MagicMock(spec=CryptoPredictor)

    sig = await get_or_create_market_direction_signal(db, market, [], predictor)
    
    assert sig.direction == "UP"
    assert sig.p_up == 0.75
    assert sig.model_version == 10
    # Predictor.predict не должен вызыветься, если сигнал уже в базе
    assert predictor.predict.call_count == 0


@pytest.mark.asyncio
async def test_get_or_create_market_direction_signal_creates_new_record():
    db = AsyncMock()
    db.add = MagicMock()
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    db.execute.return_value = mock_res

    market = MagicMock()
    market.market_id = "m200"
    market.asset = "ETH"
    market.binance_symbol = None
    market.underlying_price = 3500.0

    new_sig = CryptoSignal(
        symbol="ETHUSDT",
        model_key="ETHUSDT_low_vol",
        direction="DOWN",
        p_up=0.2,
        p_down=0.8,
        signal_strength=0.6,
        strike=3500.0,
        threshold_up=0.55,
        threshold_down=0.45,
        model_version=5,
        features_ok=True,
        risk_vetoed=False,
        regime="low_vol",
        status="READY",
    )

    predictor = MagicMock(spec=CryptoPredictor)
    predictor.predict.return_value = new_sig

    sig = await get_or_create_market_direction_signal(db, market, [], predictor)

    assert sig.direction == "DOWN"
    assert predictor.predict.call_count == 1
    assert db.add.call_count == 1
    db.commit.assert_awaited_once()
    predictor.predict.assert_called_once_with(
        [],
        "ETHUSDT",
        funding_rate=None,
        invert_lgbm_signal=False,
        underlying_price=3500.0,
    )
