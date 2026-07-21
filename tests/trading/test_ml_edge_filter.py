import pytest
import pickle
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from polyflip.db.models import LiveMarket
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.decision_runners import decide_ml_mode, DecisionResult
from polyflip.trading.ml_inference import ModelsCache


class SimpleMockModel:
    def __init__(self, prob_yes: float):
        self.prob_yes = prob_yes
        self.feature_names_in_ = ["mid_price"]

    def predict_proba(self, X):
        return [[1.0 - self.prob_yes, self.prob_yes]]


@pytest.mark.asyncio
async def test_ml_mode_respects_min_edge():
    """Тест: ML-режим должен проверять итоговый edge и пропускать сделку, если он ниже MIN_EDGE."""
    now = datetime.now(timezone.utc)
    
    # 1. Задаем конфиг с MIN_EDGE = 0.05
    cfg = MagicMock(spec=TradingConfig)
    cfg.trading_enabled = True
    cfg.trading_mode = "ml"
    cfg.min_edge = 0.05
    cfg.trade_max_price = 0.95
    cfg.favorite_threshold = 0.55
    cfg.favorite_min_price = 0.55
    cfg.favorite_max_price = 0.95
    cfg.dead_zone = 0.05
    cfg.auto_dead_zone = False
    cfg.no_flip_threshold = 0.35
    cfg.trade_on_favorite = True
    cfg.trade_on_flip = False
    cfg.active_features_str = "mid_price"
    cfg.use_crypto_confirm = False

    # 2. Создаем рынок с YES ценой 0.80
    market = LiveMarket(
        market_id="m_ml_edge", asset="BTC", question="BTC Up?",
        current_yes_price=0.80, current_no_price=0.20, current_spread=0.02,
        price_velocity=0.0, volume_5min=100.0,
        yes_token_id="tok_yes", no_token_id="tok_no",
        end_time_est=now
    )

    # 3. Мокаем API
    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.80, "current_spread": 0.02})

    # 4. Модель предсказывает P(flip) = 0.30 (то есть p_win_yes = 1 - 0.30 = 0.70)
    # Порог NO_FLIP_THRESHOLD = 0.35, значит 0.30 < 0.35 — проверка флипа пройдена.
    # Но цена входа 0.81 (yes_ask = mid_price + spread/2 = 0.81).
    # Edge = 0.70 / 0.81 - 1 = -0.135 (отрицательный).
    # При MIN_EDGE = 0.05 эта сделка обязана быть отсеяна!
    mock_model = SimpleMockModel(prob_yes=0.30)
    
    # 5. Подготавливаем кэш моделей
    models_cache = ModelsCache(
        models={"BTC": mock_model},
        versions={"BTC": 1},
        features={"BTC": ["mid_price"]}
    )

    db_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [MagicMock(asset="BTC", version=1)]
    db_session.execute = AsyncMock(return_value=mock_result)

    # Запускаем ML-режим
    result = await decide_ml_mode(
        db_session=db_session,
        api_client=mock_api,
        market=market,
        cfg=cfg,
        raw_settings={"MIN_EDGE": "0.05", "TRADE_NO_FLIP_THRESHOLD": "0.35"},
        models_cache=models_cache,
        crypto_predictor=None,
        start_time=now,
        time_left_sec=300.0
    )

    # Должен быть пропуск (SKIP) по причине недостаточного edge
    assert result.decision_obj is not None
    assert result.decision_obj.action == "SKIP"
    assert "Edge out of bounds" in result.decision_obj.reason


@pytest.mark.asyncio
async def test_ml_mode_enters_when_edge_is_sufficient():
    """Тест: ML-режим должен входить в сделку, если edge >= MIN_EDGE."""
    now = datetime.now(timezone.utc)
    
    cfg = MagicMock(spec=TradingConfig)
    cfg.trading_enabled = True
    cfg.trading_mode = "ml"
    cfg.min_edge = 0.05
    cfg.trade_max_price = 0.95
    cfg.favorite_threshold = 0.55
    cfg.favorite_min_price = 0.55
    cfg.favorite_max_price = 0.95
    cfg.dead_zone = 0.05
    cfg.auto_dead_zone = False
    cfg.no_flip_threshold = 0.35
    cfg.trade_on_favorite = True
    cfg.trade_on_flip = False
    cfg.active_features_str = "mid_price"
    cfg.use_crypto_confirm = False
    cfg.bet_size = 10.0
    cfg.max_bet_size_usdc = 50.0
    cfg.max_bet_edge = 0.30
    cfg.liquidity_fraction = 0.1
    cfg.bet_sizing_mode = "fixed"

    # YES цена 0.60
    market = LiveMarket(
        market_id="m_ml_edge", asset="BTC", question="BTC Up?",
        current_yes_price=0.60, current_no_price=0.40, current_spread=0.01,
        price_velocity=0.0, volume_5min=100.0,
        yes_token_id="tok_yes", no_token_id="tok_no",
        end_time_est=now
    )

    mock_api = MagicMock()
    mock_api.get_market_prices = AsyncMock(return_value={"current_yes_price": 0.60, "current_spread": 0.01})

    # P(flip) = 0.10 (p_win_yes = 0.90)
    # buy_price = 0.605
    # Edge = 0.90 / 0.605 - 1 = 0.487 > 0.05
    mock_model = SimpleMockModel(prob_yes=0.10)
    
    models_cache = ModelsCache(
        models={"BTC": mock_model},
        versions={"BTC": 1},
        features={"BTC": ["mid_price"]}
    )

    db_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [MagicMock(asset="BTC", version=1)]
    db_session.execute = AsyncMock(return_value=mock_result)

    result = await decide_ml_mode(
        db_session=db_session,
        api_client=mock_api,
        market=market,
        cfg=cfg,
        raw_settings={"MIN_EDGE": "0.05", "TRADE_NO_FLIP_THRESHOLD": "0.35"},
        models_cache=models_cache,
        crypto_predictor=None,
        start_time=now,
        time_left_sec=300.0
    )

    assert result.decision_obj is not None
    assert result.decision_obj.action == "BUY_YES"
