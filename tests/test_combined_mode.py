import pytest
import math
from polyflip.trading.combined_voting import CryptoSignalProxy, combine_votes

def test_combined_mode_agreement():
    """ML и LightGBM согласны — берем сделку с бустом confidence"""
    crypto = CryptoSignalProxy(direction="UP", features_ok=True)
    res = combine_votes("BUY_YES", 0.10, crypto, "BTC")
    assert res.action == "BUY_YES"
    assert math.isclose(res.confidence, 0.12, rel_tol=1e-9)  # 0.10 * 1.2

    crypto = CryptoSignalProxy(direction="DOWN", features_ok=True)
    res = combine_votes("BUY_NO", 0.05, crypto, "ETH")
    assert res.action == "BUY_NO"
    assert math.isclose(res.confidence, 0.06, rel_tol=1e-9)

def test_combined_mode_veto():
    """ML и LightGBM не согласны — вето (SKIP)"""
    # ML говорит YES, LGBM говорит DOWN (вето)
    crypto = CryptoSignalProxy(direction="DOWN", features_ok=True)
    res = combine_votes("BUY_YES", 0.15, crypto, "BTC")
    assert res.action == "SKIP"
    assert "veto" in res.reason.lower()

    # ML говорит NO, LGBM говорит UP (вето)
    crypto = CryptoSignalProxy(direction="UP", features_ok=True)
    res = combine_votes("BUY_NO", 0.10, crypto, "ETH")
    assert res.action == "SKIP"
    assert "veto" in res.reason.lower()

    # ML говорит YES, LGBM говорит NONE (нет тренда) -> НЕ вето, а уменьшение ставки!
    crypto = CryptoSignalProxy(direction="NONE", features_ok=True)
    res = combine_votes("BUY_YES", 0.10, crypto, "BTC")
    assert res.action == "BUY_YES"
    assert res.bet_size_multiplier == 0.5
    
    # Если LGBM вернул None при features_ok=True (баг предиктора),
    # combine_votes обработает это как вето (None != "UP")
    crypto = CryptoSignalProxy(direction=None, features_ok=True)
    res = combine_votes("BUY_YES", 0.10, crypto, "BTC")
    assert res.action == "SKIP"
    assert "veto" in res.reason.lower()

def test_combined_mode_ml_skip():
    """ML уже SKIP -> оставляем SKIP, независим от LightGBM"""
    crypto = CryptoSignalProxy(direction="UP", features_ok=True)
    res = combine_votes("SKIP", 0.0, crypto, "BTC")
    assert res.action == "SKIP"
    assert "ML (in Combined mode) voted SKIP" in res.reason

def test_combined_mode_fallback():
    """LightGBM features_ok = False -> Fallback на ML"""
    crypto = CryptoSignalProxy(direction=None, features_ok=False)
    
    # ML говорит YES
    res = combine_votes("BUY_YES", 0.15, crypto, "BTC")
    assert res.action == "BUY_YES"
    assert res.confidence == 0.15  # без буста
    assert res.lgbm_features_ok is False
    assert "fallback" in res.reason.lower()

    # ML говорит NO
    res = combine_votes("BUY_NO", 0.10, crypto, "ETH")
    assert res.action == "BUY_NO"
    assert res.confidence == 0.10
    assert res.lgbm_features_ok is False
    assert "fallback" in res.reason.lower()

def test_combined_flat_reduced_bet():
    """LGBM в состоянии NONE -> берется решение ML, но с уменьшенным множителем"""
    crypto = CryptoSignalProxy(direction="NONE", features_ok=True)
    
    # С дефолтным множителем 0.5
    res = combine_votes("BUY_YES", 0.15, crypto, "BTC", none_bet_multiplier=0.5)
    assert res.action == "BUY_YES"
    assert res.bet_size_multiplier == 0.5
    assert "flat (NONE)" in res.reason

    # С кастомным множителем 0.3
    res2 = combine_votes("BUY_NO", 0.15, crypto, "ETH", none_bet_multiplier=0.3)
    assert res2.action == "BUY_NO"
    assert res2.bet_size_multiplier == 0.3

@pytest.mark.asyncio
async def test_combined_skip_preserves_edge(mocker):
    """decide_combined_mode должен сохранять edge при SKIP (вето)"""
    from unittest.mock import AsyncMock, MagicMock
    from polyflip.trading.decision_runners import decide_combined_mode, DecisionResult
    from polyflip.trading.decision_logic import TradeDecision

    # Мокаем decide_ml_mode, чтобы он возвращал ML-сигнал с положительным edge
    ml_dec = TradeDecision(action="BUY_YES", buy_price=0.55, bet_size_usdc=10.0, reason="ML ok", strategy_type="ML_TREND", p_up=0.60, strike=0.5)
    ml_res = DecisionResult(decision_obj=ml_dec, p_flip=0.12, model_ver=4, edge=0.08, skip_reason=None)
    mocker.patch("polyflip.trading.decision_runners.decide_ml_mode", AsyncMock(return_value=ml_res))

    # Мокаем _fetch_lgbm_signal, чтобы он возвращал вето (LGBM=DOWN при ML=BUY_YES)
    from polyflip.trading.combined_voting import CryptoSignalProxy
    lgbm_proxy = CryptoSignalProxy(direction="DOWN", features_ok=True, model_version=10)
    mocker.patch("polyflip.trading.decision_runners._fetch_lgbm_signal", AsyncMock(return_value=lgbm_proxy))

    # Запускаем decide_combined_mode
    db_session = AsyncMock()
    api_client = MagicMock()
    market = MagicMock()
    market.asset = "BTC"
    cfg = MagicMock()
    cfg.bet_size = 10.0
    raw_settings = {"COMBINED_NONE_BET_MULTIPLIER": "0.5"}
    models_cache = MagicMock()
    crypto_predictor = MagicMock()

    res = await decide_combined_mode(
        db_session, api_client, market, cfg, raw_settings,
        models_cache, crypto_predictor, start_time=MagicMock(), time_left_sec=300
    )

    # Должен быть SKIP (вето)
    assert res.decision_obj.action == "SKIP"
    # Но edge от ML должен сохраниться!
    assert res.edge == 0.08
    assert "veto" in res.skip_reason.lower()
    assert res.lgbm_metadata is not None
    
    # Проверяем lgbm_metadata
    import json
    meta = json.loads(res.lgbm_metadata)
    assert meta["lgbm_direction"] == "DOWN"
    assert meta["vote_action"] == "SKIP"
    assert meta["bet_size_multiplier"] == 0.0

