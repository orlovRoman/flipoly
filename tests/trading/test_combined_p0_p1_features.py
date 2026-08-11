import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from polyflip.trading.combined_voting import evaluate_combined_entry
from polyflip.crypto.predictor import CryptoSignal
from polyflip.trading.trading_config import TradingConfig
from polyflip.trading.ml_inference import run_model_inference


def make_test_cfg(**kwargs) -> TradingConfig:
    defaults = {
        "trading_enabled": True,
        "trading_mode": "combined",
        "favor_min_time_left": 60,
        "favor_max_time_left": 600,
        "outs_min_time_left": 30,
        "outs_max_time_left": 300,
        "bet_size": 10.0,
        "dead_zone": 0.05,
        "daily_limit": -100.0,
        "trade_min_price": 0.05,
        "trade_max_price": 0.95,
        "capital": 100.0,
        "active_features_str": "",
        "trade_on_favorite": True,
        "trade_on_flip": True,
        "flip_threshold": 0.60,
        "outs_min_edge": 0.04,
        "favorite_threshold": 0.70,
        "trade_assets": ["BTC"],
        "bet_sizing_mode": "fixed",
        "max_bet_size_usdc": 50.0,
        "favorite_min_price": 0.55,
        "favorite_max_price": 0.95,
        "favorite_min_edge": 0.05,
        "outsider_max_price": 0.40,
        "liquidity_fraction": 0.1,
        "bypass_bet_size_check": False,
        "stop_loss_enabled": False,
        "take_profit_enabled": False,
        "take_profit_multiplier": 2.0,
        "max_price_drift": 0.03,
        "stop_loss_pct_favorite": 40.0,
        "stop_loss_pct_outsider": 60.0,
        "fee_rate": 0.0,
        "slippage_rate": 0.0,
        "max_exposure_pct": 15.0,
        "min_direction_prob": 0.505,
        "min_win_prob": 0.51,
        "combined_dir_discount_weight": 0.0,
        "combined_dir_strong_threshold": 0.65,
        "combined_require_consensus": True,
        "combined_fallback_to_logreg_on_none": True,
        "combined_logreg_abstain_band": 0.05,
        "invert_lgbm_signal": False,
        "max_bet_edge": 0.40,
        "outsider_pwin_discount": 0.65,
        "max_spread_pct": 0.08,
        "combined_cost_buffer": 0.02,
        "lgbm_unavailable_policy": "SKIP",
    }
    defaults.update(kwargs)
    return TradingConfig(**defaults)


def make_test_sig(**kwargs) -> CryptoSignal:
    defaults = {
        "symbol": "BTC",
        "direction": "UP",
        "p_up": 0.70,
        "p_down": 0.30,
        "signal_strength": 0.20,
        "strike": 0.50,
        "threshold_up": 0.55,
        "threshold_down": 0.45,
        "status": "READY",
        "features_ok": True,
        "model_version": 1,
    }
    defaults.update(kwargs)
    return CryptoSignal(**defaults)


def test_lgbm_unavailable_policy_skip():
    cfg = make_test_cfg(lgbm_unavailable_policy="SKIP", lightgbm_decision_mode="ACTIVE")
    sig = make_test_sig(
        direction="NONE",
        p_up=0.5,
        p_down=0.5,
        status="MODEL_NOT_LOADED",
        features_ok=False,
        model_version=None,
    )
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="contested",
        entry_requested_key="BTC_contested",
        entry_model_key="BTC_contested",
        entry_model_version=1,
        entry_model_source="PHASE",
        p_flip=0.70,
        fresh_yes_price=0.60,
        yes_ask=0.60,
        no_ask=0.40,
        cfg=cfg,
        time_left_sec=300.0,
    )
    assert res.action == "SKIP"
    assert res.entry_status == "DIRECTION_UNAVAILABLE"
    assert res.would_live_accept is False


def test_lgbm_unavailable_policy_logreg_fallback():
    cfg = make_test_cfg(lgbm_unavailable_policy="LOGREG_FALLBACK", lightgbm_decision_mode="ACTIVE")
    sig = make_test_sig(
        direction="NONE",
        p_up=0.5,
        p_down=0.5,
        status="MODEL_NOT_LOADED",
        features_ok=False,
        model_version=None,
    )
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="contested",
        entry_requested_key="BTC_contested",
        entry_model_key="BTC_contested",
        entry_model_version=1,
        entry_model_source="PHASE",
        p_flip=0.70,
        fresh_yes_price=0.60,
        yes_ask=0.60,
        no_ask=0.40,
        cfg=cfg,
        time_left_sec=300.0,
    )
    # With LOGREG_FALLBACK, evaluation proceeds via PARTIAL_LR consensus
    assert res.consensus_type == "PARTIAL_LR"
    assert res.direction_status == "DIRECTION_NONE_FALLBACK_LR"
    assert res.would_live_accept is False


def test_ece_correction_on_p_flip():
    cfg = make_test_cfg()
    sig = make_test_sig()
    res = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="contested",
        entry_requested_key="BTC_contested",
        entry_model_key="BTC_contested",
        entry_model_version=1,
        entry_model_source="PHASE",
        p_flip=0.80,
        fresh_yes_price=0.60,
        yes_ask=0.60,
        no_ask=0.40,
        cfg=cfg,
        time_left_sec=300.0,
        entry_model_ece=0.10,
    )
    assert res.p_flip_raw == 0.80
    assert res.entry_model_ece == 0.10
    assert res.p_flip_effective is not None and res.p_flip_effective < 0.80


def test_feature_mismatch_detection():
    model = MagicMock(spec=[])
    features = ["mid_price", "spread", "required_custom_feature"]

    df = pd.DataFrame([{"mid_price": 0.5, "spread": 0.01}])

    with pytest.raises(ValueError, match="MODEL_FEATURE_MISMATCH"):
        run_model_inference(df, model, features)


def test_allowed_zero_default_features():
    model = MagicMock(spec=[])
    model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
    features = ["mid_price", "spread", "volume_5min"]

    df = pd.DataFrame([{"mid_price": 0.5, "spread": 0.01}])

    p = run_model_inference(df, model, features)
    assert p == 0.7


def test_would_live_accept_flag():
    cfg = make_test_cfg()
    sig = make_test_sig(direction="UP", p_up=0.70)
    
    # Matching agreement: LGBM=UP (BUY_YES), LogReg=p_flip=0.25 with fresh_yes_price=0.58 (lr_vote=BUY_YES)
    res_phase = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="contested",
        entry_requested_key="BTC_contested",
        entry_model_key="BTC_contested",
        entry_model_version=1,
        entry_model_source="PHASE",
        p_flip=0.25,
        fresh_yes_price=0.58,
        yes_ask=0.58,
        no_ask=0.42,
        cfg=cfg,
        time_left_sec=300.0,
    )
    assert res_phase.action == "BUY_YES"
    assert res_phase.would_live_accept is True

    # BASE fallback model -> would_live_accept False
    res_base = evaluate_combined_entry(
        crypto_sig=sig,
        market_phase="contested",
        entry_requested_key="BTC_contested",
        entry_model_key="BTC",
        entry_model_version=1,
        entry_model_source="BASE",
        p_flip=0.25,
        fresh_yes_price=0.58,
        yes_ask=0.58,
        no_ask=0.42,
        cfg=cfg,
        time_left_sec=300.0,
    )
    assert res_base.action == "BUY_YES"
    assert res_base.would_live_accept is False
