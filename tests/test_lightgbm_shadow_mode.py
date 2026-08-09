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

def test_shadow_mode_opposing_lgbm_signal_and_veto_ignored():
    cfg = parse_trading_settings({"LIGHTGBM_DECISION_MODE": "SHADOW"})

    # LightGBM predicts UP and has funding veto
    crypto_sig = CryptoSignal(
        symbol="BTCUSDT",
        p_up=0.90,
        p_down=0.10,
        direction="UP",
        signal_strength=0.30,
        strike=100000.0,
        threshold_up=0.60,
        threshold_down=0.40,
        model_version=10,
        features_ok=True,
        risk_vetoed=True,
        risk_reason="Funding rate spike",
        regime="low_vol",
        status="READY",
    )

    # LogReg predicts p_flip=0.75 at fresh_yes_price=0.50 (fav YES) -> votes BUY_NO
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
    assert result.candidate_side == "BUY_NO"
    assert result.direction_status == "SHADOW_NOT_APPLIED"
    assert result.p_candidate_win == result.p_logreg_win
    assert result.direction_discount_applied == 1.0

def test_off_mode_parsing_and_disabled_status():
    cfg = parse_trading_settings({"LIGHTGBM_DECISION_MODE": "OFF"})
    assert cfg.lightgbm_decision_mode == "OFF"

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
        status="DISABLED_BY_OPERATOR",
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
    assert result.direction_status == "DISABLED_BY_OPERATOR"

def test_ece_correction_toggle_disabled():
    cfg_enabled = parse_trading_settings({"ENABLE_ECE_CORRECTION": True})
    cfg_disabled = parse_trading_settings({"ENABLE_ECE_CORRECTION": False})
    
    crypto_sig = CryptoSignal(
        symbol="BTCUSDT", p_up=0.0, p_down=0.0, direction="NONE",
        signal_strength=0.0, strike=0.0, threshold_up=0.0, threshold_down=0.0,
        model_version=-1, features_ok=False, risk_vetoed=False, regime="UNKNOWN",
        status="DISABLED_BY_OPERATOR",
    )

    res_enabled = evaluate_combined_entry(
        crypto_sig=crypto_sig, market_phase="decided", entry_requested_key="BTC_decided",
        entry_model_key="BTC_decided_v10", entry_model_version=10, entry_model_source="phase_matched",
        p_flip=0.75, fresh_yes_price=0.50, yes_ask=0.51, no_ask=0.51,
        cfg=cfg_enabled, cost_buffer=0.02, time_left_sec=120.0, underlying_price=100000.0,
        entry_model_ece=0.25,
    )
    # High ECE (0.25) shrinks p_flip from 0.75 down to 0.50 when enabled
    assert res_enabled.p_flip == 0.50

    res_disabled = evaluate_combined_entry(
        crypto_sig=crypto_sig, market_phase="decided", entry_requested_key="BTC_decided",
        entry_model_key="BTC_decided_v10", entry_model_version=10, entry_model_source="phase_matched",
        p_flip=0.75, fresh_yes_price=0.50, yes_ask=0.51, no_ask=0.51,
        cfg=cfg_disabled, cost_buffer=0.02, time_left_sec=120.0, underlying_price=100000.0,
        entry_model_ece=0.25,
    )
    # When disabled, p_flip is kept raw at 0.75
    assert res_disabled.p_flip == 0.75

