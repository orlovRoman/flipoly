import pytest
from polyflip.trading.trading_config import parse_trading_settings
from polyflip.trading.combined_voting import evaluate_combined_entry
from polyflip.crypto.predictor import CryptoSignal

def test_lightgbm_mode_defaults_to_shadow():
    cfg = parse_trading_settings({})
    assert cfg.lightgbm_decision_mode == "SHADOW"

def test_lightgbm_mode_active_parsing():
    cfg = parse_trading_settings({"LIGHTGBM_DECISION_MODE": "ACTIVE"})
    assert cfg.lightgbm_decision_mode == "ACTIVE"

def test_shadow_ignores_invalid_lgbm_and_uses_logreg():
    cfg = parse_trading_settings({"LIGHTGBM_DECISION_MODE": "SHADOW"})
    
    # LightGBM fails / disabled
    crypto_sig = CryptoSignal(
        symbol="BTCUSDT",
        p_up=0.0,
        p_down=0.0,
        direction="NONE",
        signal_strength=0.0,
        strike=0.0,
        threshold_up=0.0,
        threshold_down=0.0,
        model_version=-1,
        features_ok=False,
        risk_vetoed=False,
        regime="UNKNOWN",
        status="INFERENCE_FAILED",
    )

    result = evaluate_combined_entry(
        crypto_sig=crypto_sig,
        market_phase="contested",
        entry_requested_key="BTC_contested",
        entry_model_key="BTC_contested_v5",
        entry_model_version=5,
        entry_model_source="phase_matched",
        p_flip=0.75,
        fresh_yes_price=0.50,
        yes_ask=0.51,
        no_ask=0.51,
        cfg=cfg,
        cost_buffer=0.02,
        time_left_sec=120.0,
        underlying_price=100000.0,
    )

    assert result.consensus_type == "LOGREG_ONLY"
    assert result.direction_status == "SHADOW_NOT_APPLIED"
    assert result.p_candidate_win == result.p_logreg_win
    assert result.direction_discount_applied == 1.0
