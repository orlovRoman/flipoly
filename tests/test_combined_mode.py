import pytest
import math
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from polyflip.trading.combined_voting import evaluate_combined_entry, CombinedEntryResult
from polyflip.crypto.predictor import CryptoSignal
from polyflip.trading.decision_runners import decide_combined_mode, DecisionResult
from polyflip.trading.trading_config import parse_trading_settings


# Вспомогательный cfg с "открытыми" ценовыми фильтрами для тестов
def _make_cfg(**overrides):
    base = {
        "MIN_EDGE": "0.03",
        "NO_MIN_EDGE": "0.03",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "FAVORITE_MIN_PRICE": "0.05",
        "FAVORITE_MAX_PRICE": "0.95",
        "OUTSIDER_MAX_PRICE": "0.49",
        "TRADE_ON_FLIP": "true",
        "FAVORITE_THRESHOLD": "0.50",
    }
    base.update(overrides)
    return parse_trading_settings(base)

def test_evaluate_combined_entry_direction_up_success():
    """LightGBM = UP, LogReg дает хороший net_edge -> BUY_YES"""
    sig = CryptoSignal(
        symbol="BTCUSDT", p_up=0.75, p_down=0.25, direction="UP",
        signal_strength=0.5, strike=65000.0, threshold_up=0.55, threshold_down=0.45,
        model_version=2, features_ok=True, risk_vetoed=False, regime="HIGH_VOL"
    )
    # p_flip = 0.20 (YES фаворит: fresh_yes_price=0.60 >= 0.50 -> p_candidate_win = 1 - 0.20 = 0.80)
    # yes_ask = 0.62
    # gross_edge = 0.80 - 0.62 = 0.18
    # cost_buffer = 0.03 -> net_edge = 0.15 >= 0.03
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="BTC_mid_vol",
        entry_model_version=5,
        entry_model_source="PHASE",
        p_flip=0.20,
        fresh_yes_price=0.60,
        yes_ask=0.62,
        no_ask=0.38,
        cost_buffer=0.03,
        cfg=_make_cfg(),
        config_dict={"TRADE_BET_SIZE_USDC": "10", "MAX_BET_SIZE_USDC": "50"},
    )
    assert res.action == "BUY_YES"
    assert res.candidate_side == "BUY_YES"
    assert res.direction_status == "READY"
    assert res.direction_value == "UP"
    assert res.entry_status == "READY"
    assert res.entry_model_source == "PHASE"
    expected_gross = 0.80 - 0.62
    expected_net = round(expected_gross - 0.03, 4)
    assert math.isclose(res.gross_edge, expected_gross, rel_tol=1e-4)
    assert math.isclose(res.net_edge, expected_net, rel_tol=1e-4)
    assert res.max_acceptable_price is not None
    assert res.bet_size_usdc >= 10.0


def test_evaluate_combined_entry_direction_down_success():
    """LightGBM = DOWN, LogReg дает хороший net_edge для NO -> BUY_NO"""
    sig = CryptoSignal(
        symbol="ETHUSDT", p_up=0.20, p_down=0.80, direction="DOWN",
        signal_strength=0.6, strike=3500.0, threshold_up=0.55, threshold_down=0.45,
        model_version=3, features_ok=True, risk_vetoed=False, regime="MID_VOL"
    )
    # fresh_yes_price = 0.50 >= 0.50, candidate = NO -> p_candidate_win = p_flip = 0.85
    # no_ask = 0.55
    # gross_edge = 0.85 - 0.55 = 0.30
    # cost_buffer = 0.03 -> net_edge = 0.27
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="mid_vol",
        entry_requested_key="ETH_mid_vol",
        entry_model_key="ETH_mid_vol",
        entry_model_version=2,
        entry_model_source="PHASE",
        p_flip=0.85,
        fresh_yes_price=0.50,
        yes_ask=0.52,
        no_ask=0.55,
        cost_buffer=0.03,
        cfg=_make_cfg(OUTSIDER_MAX_PRICE="0.60"),
        config_dict={"TRADE_BET_SIZE_USDC": "10", "MAX_BET_SIZE_USDC": "50"},
    )
    assert res.action == "BUY_NO"
    assert res.candidate_side == "BUY_NO"
    assert res.direction_status == "READY"
    assert res.direction_value == "DOWN"
    assert res.net_edge > 0.03


def test_evaluate_combined_entry_direction_invalid_features():
    """LightGBM features_ok=False -> SKIP"""
    sig = CryptoSignal(
        symbol="BTCUSDT", p_up=0.70, p_down=0.30, direction="UP",
        signal_strength=0.4, strike=65000.0, threshold_up=0.55, threshold_down=0.45,
        model_version=2, features_ok=False, risk_vetoed=False, status="INSUFFICIENT_CANDLES"
    )
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="BTC_mid_vol",
        entry_model_version=5,
        entry_model_source="PHASE",
        p_flip=0.20,
        fresh_yes_price=0.60,
        yes_ask=0.62,
        no_ask=0.38,
        cfg=_make_cfg(),
    )
    assert res.action == "SKIP"
    assert res.direction_status == "INSUFFICIENT_CANDLES"
    assert "direction model unavailable" in res.reason.lower()


def test_evaluate_combined_entry_direction_risk_vetoed():
    """LightGBM risk_vetoed=True -> SKIP"""
    sig = CryptoSignal(
        symbol="BTCUSDT", p_up=0.70, p_down=0.30, direction="UP",
        signal_strength=0.4, strike=65000.0, threshold_up=0.55, threshold_down=0.45,
        model_version=2, features_ok=True, risk_vetoed=True, risk_reason="Extreme funding rate"
    )
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="BTC_mid_vol",
        entry_model_version=5,
        entry_model_source="PHASE",
        p_flip=0.20,
        fresh_yes_price=0.60,
        yes_ask=0.62,
        no_ask=0.38,
        cfg=_make_cfg(),
    )
    assert res.action == "SKIP"
    assert res.direction_status == "FUNDING_VETOED"
    assert "funding veto" in res.reason.lower()


def test_evaluate_combined_entry_insufficient_net_edge():
    """net_edge < min_net_edge -> SKIP"""
    sig = CryptoSignal(
        symbol="BTCUSDT", p_up=0.75, p_down=0.25, direction="UP",
        signal_strength=0.5, strike=65000.0, threshold_up=0.55, threshold_down=0.45,
        model_version=2, features_ok=True, risk_vetoed=False
    )
    # p_flip = 0.35, fresh_yes_price = 0.64 -> p_candidate_win = 1 - 0.35 = 0.65
    # yes_ask = 0.64
    # gross_edge = 0.65 - 0.64 = 0.01
    # cost_buffer = 0.03 -> net_edge = -0.02 < 0.03
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="BTC_mid_vol",
        entry_model_version=5,
        entry_model_source="PHASE",
        p_flip=0.35,
        fresh_yes_price=0.64,
        yes_ask=0.64,
        no_ask=0.36,
        cost_buffer=0.03,
        cfg=_make_cfg(MIN_EDGE="0.03"),
    )
    assert res.action == "SKIP"
    assert res.entry_status == "INSUFFICIENT_NET_EDGE"
    assert "insufficient net edge" in res.reason.lower()


def test_evaluate_combined_entry_model_fallback_global():
    """Entry model fallback to GLOBAL"""
    sig = CryptoSignal(
        symbol="BTCUSDT", p_up=0.80, p_down=0.20, direction="UP",
        signal_strength=0.6, strike=65000.0, threshold_up=0.55, threshold_down=0.45,
        model_version=2, features_ok=True, risk_vetoed=False
    )
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="GLOBAL",
        entry_model_version=1,
        entry_model_source="GLOBAL",
        fallback_reason="Base model BTC not found, fell back to GLOBAL",
        p_flip=0.10,
        fresh_yes_price=0.50,
        yes_ask=0.52,
        no_ask=0.48,
        cost_buffer=0.03,
        cfg=_make_cfg(),
        config_dict={"TRADE_BET_SIZE_USDC": "10"},
    )
    assert res.action == "BUY_YES"
    assert res.entry_model_source == "GLOBAL"
    assert res.fallback_reason == "Base model BTC not found, fell back to GLOBAL"


def test_decide_combined_mode_full_flow():
    """Тест decide_combined_mode с проверкой единого вызова log_funnel и всех полей"""
    db_session = AsyncMock()
    api_client = AsyncMock()
    api_client.get_market_prices.return_value = {
        "current_yes_price": "0.60",
        "best_ask": "0.62",
        "current_spread": "0.02",
    }
    market = MagicMock()
    market.market_id = 123
    market.asset = "BTC"
    market.yes_token_id = "token_yes_123"
    market.no_token_id = "token_no_123"
    market.volume_5min = 500.0
    market.underlying_price = 65000.0

    cfg = parse_trading_settings({"COMBINED_COST_BUFFER": "0.03"})
    raw_settings = {}
    
    mock_model = MagicMock()
    models_cache = MagicMock()
    models_cache.models = {"BTC_leaning": mock_model}
    models_cache.versions = {"BTC_leaning": 4}
    models_cache.features = {"BTC_leaning": ["f1", "f2"]}

    crypto_sig = CryptoSignal(
        symbol="BTCUSDT", model_key="BTCUSDT_mid_vol", p_up=0.85, p_down=0.15, direction="UP",
        signal_strength=0.7, strike=65000.0, threshold_up=0.55, threshold_down=0.45,
        model_version=10, features_ok=True, risk_vetoed=False, regime="MID_VOL", status="OK"
    )
    crypto_predictor = MagicMock()

    with patch("polyflip.trading.decision_runners._fetch_lgbm_signal", AsyncMock(return_value=crypto_sig)), \
         patch("polyflip.trading.decision_runners.infer_flip_for_market", AsyncMock(return_value=0.15)), \
         patch("polyflip.trading.decision_runners.log_funnel", AsyncMock()) as mock_log_funnel:

        res = asyncio.run(decide_combined_mode(
            db_session=db_session,
            api_client=api_client,
            market=market,
            cfg=cfg,
            raw_settings=raw_settings,
            models_cache=models_cache,
            crypto_predictor=crypto_predictor,
            start_time=MagicMock(),
            time_left_sec=300.0,
            execution_mode="PAPER"
        ))

        assert res.decision_obj.action == "BUY_YES"
        assert res.decision_obj.strategy_type == "COMBINED"
        assert res.used_model_key == "BTC_leaning"
        assert res.confirm_model_key == "BTCUSDT_mid_vol"
        assert res.confirm_model_version == 10
        
        # Проверяем, что log_funnel вызван ровно 1 раз
        assert mock_log_funnel.call_count == 1
        call_kwargs = mock_log_funnel.call_args.kwargs
        assert call_kwargs["trading_mode"] == "COMBINED"
        assert call_kwargs["direction_status"] == "OK"
        assert call_kwargs["direction_value"] == "UP"
        assert call_kwargs["entry_status"] == "READY"
        assert call_kwargs["entry_model_source"] == "PHASE"
        assert call_kwargs["gross_edge"] is not None
        assert call_kwargs["net_edge"] is not None
