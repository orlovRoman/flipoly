import pytest
from polyflip.trading.combined_voting import apply_direction_confidence_discount, evaluate_combined_entry
from polyflip.crypto.predictor import CryptoSignal
from polyflip.trading.trading_config import parse_trading_settings


def test_apply_direction_confidence_discount_disabled():
    # When discount_weight is 0.0, probability is untouched
    res = apply_direction_confidence_discount(
        p_logreg_win=0.70,
        dir_prob=0.51,
        min_direction_prob=0.505,
        strong_threshold=0.65,
        discount_weight=0.0,
    )
    assert res == pytest.approx(0.70, abs=1e-4)


def test_apply_direction_confidence_discount_strong_confidence():
    # When dir_prob >= strong_threshold, multiplier is 1.0 (no discount)
    res = apply_direction_confidence_discount(
        p_logreg_win=0.80,
        dir_prob=0.70,
        min_direction_prob=0.505,
        strong_threshold=0.65,
        discount_weight=0.25,
    )
    assert res == pytest.approx(0.80, abs=1e-4)


def test_apply_direction_confidence_discount_min_confidence():
    # When dir_prob <= min_direction_prob, maximum discount is applied: 0.80 * (1 - 0.20) = 0.64
    res = apply_direction_confidence_discount(
        p_logreg_win=0.80,
        dir_prob=0.505,
        min_direction_prob=0.505,
        strong_threshold=0.65,
        discount_weight=0.20,
    )
    assert res == pytest.approx(0.64, abs=1e-4)


def test_apply_direction_confidence_discount_halfway():
    # Band = 0.65 - 0.505 = 0.145. Halfway = 0.5775. Weakness = 0.5.
    # Multiplier = 1.0 - (0.20 * 0.5) = 0.90.
    # 0.70 * 0.90 = 0.63.
    res = apply_direction_confidence_discount(
        p_logreg_win=0.70,
        dir_prob=0.5775,
        min_direction_prob=0.505,
        strong_threshold=0.65,
        discount_weight=0.20,
    )
    assert res == pytest.approx(0.63, abs=1e-4)


def test_apply_direction_confidence_discount_invalid_band():
    # When strong_threshold <= min_direction_prob, fallback returns untouched
    res = apply_direction_confidence_discount(
        p_logreg_win=0.60,
        dir_prob=0.55,
        min_direction_prob=0.65,
        strong_threshold=0.60,
        discount_weight=0.20,
    )
    assert res == pytest.approx(0.60, abs=1e-4)


def test_evaluate_combined_entry_with_discount():
    crypto_sig = CryptoSignal(
        symbol="BTCUSDT",
        p_up=0.52,
        p_down=0.48,
        direction="UP",
        signal_strength=0.015,
        strike=65000.0,
        threshold_up=0.505,
        threshold_down=0.505,
        model_version=1,
        features_ok=True,
        model_key="BTC_LGBM",
        regime="low_vol",
        risk_vetoed=False,
    )

    raw_config = {
        "TRADING_ENABLED": "true",
        "TRADING_MODE": "combined",
        "TRADE_ASSETS": "BTC",
        "MIN_DIRECTION_PROB": "0.505",
        "MIN_WIN_PROB": "0.51",
        "OUTS_MIN_EDGE": "0.02",
        "COMBINED_COST_BUFFER": "0.01",
        "COMBINED_DIR_DISCOUNT_WEIGHT": "0.20",
        "COMBINED_DIR_STRONG_THRESHOLD": "0.65",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "TRADE_BET_SIZE_USDC": "10.0",
        "MAX_BET_SIZE_USDC": "50.0",
        "BET_SIZING_MODE": "fixed",
    }
    cfg = parse_trading_settings(raw_config)

    # p_flip = 0.30 -> Candidate YES (price 0.60 >= 0.50, favorite) -> p_logreg_win = 1 - 0.30 = 0.70
    # dir_prob = 0.52.
    # Band = 0.65 - 0.505 = 0.145.
    # Weakness = (0.65 - 0.52) / 0.145 = 0.13 / 0.145 = ~0.89655.
    # Multiplier = 1.0 - (0.20 * 0.89655) = ~0.82069.
    # p_candidate_win = round(0.70 * 0.82069, 4) = round(0.57448, 4) = 0.5745.

    # Case 1: Raw p_logreg_win (0.70) had positive edge (0.70 - 0.60 - 0.01 = +0.09 > 0.02),
    # but discount reduces p_candidate_win to 0.5745, resulting in net_edge -0.0355 < 0.02 -> SKIP!
    result = evaluate_combined_entry(
        crypto_sig=crypto_sig,
        market_phase="leaning",
        entry_requested_key="BTC_leaning",
        entry_model_key="BTC_leaning",
        entry_model_version=1,
        entry_model_source="PHASE",
        p_flip=0.30,
        fresh_yes_price=0.60,
        yes_ask=0.60,
        no_ask=0.40,
        cost_buffer=0.01,
        cfg=cfg,
        volume_5min=1000.0,
        config_dict=raw_config, time_left_sec=300,
    )

    assert result.p_logreg_win == pytest.approx(0.70, abs=1e-4)
    assert result.p_candidate_win < 0.70
    assert result.p_candidate_win == pytest.approx(0.5745, abs=1e-4)
    assert result.direction_discount_applied < 1.0
    assert result.combined_dir_discount_weight == pytest.approx(0.20, abs=1e-4)
    assert result.action == "SKIP"
    assert "Insufficient net edge" in result.reason

    # Case 2: Stronger LogReg signal (p_flip=0.10 -> p_logreg_win=0.90) -> discounted p_candidate_win = 0.90 * 0.82069 = 0.7386
    # Net edge = 0.7386 - 0.60 - 0.01 = +0.1286 >= 0.02 -> BUY_YES!
    result_buy = evaluate_combined_entry(
        crypto_sig=crypto_sig,
        market_phase="leaning",
        entry_requested_key="BTC_leaning",
        entry_model_key="BTC_leaning",
        entry_model_version=1,
        entry_model_source="PHASE",
        p_flip=0.10,
        fresh_yes_price=0.60,
        yes_ask=0.60,
        no_ask=0.40,
        cost_buffer=0.01,
        cfg=cfg,
        volume_5min=1000.0,
        config_dict=raw_config, time_left_sec=300,
    )
    assert result_buy.action == "BUY_YES"
    assert result_buy.p_candidate_win == pytest.approx(0.7386, abs=1e-4)
    assert result_buy.net_edge == pytest.approx(0.1286, abs=1e-4)
