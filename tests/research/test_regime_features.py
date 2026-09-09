"""
tests/research/test_regime_features.py

Comprehensive tests for Stage 2 features:
- Efficiency Ratio (monotonic move -> 1.0, symmetric saw -> 0.0, constant price -> division by zero safe)
- Normalized slope
- Return sign change frequency (zero returns filtered out, not creating artificial saw)
- Lag-1 return autocorrelation
- 4-state regime classifier (TREND, REVERSION, QUIET, UNCERTAIN)
- Strike moneyness z_strike and reversion benefit check (ending on losing side is NOT marked favorable)
- Mirror symmetry invariance under price reflection and UP/DOWN swap
"""
import math
import numpy as np
import pytest

from polyflip.research.regime_features import (
    compute_efficiency_ratio,
    compute_normalized_slope,
    compute_return_sign_changes,
    compute_return_autocorrelation,
    compute_local_mean_deviation,
    classify_local_regime,
    compute_strike_context,
    verify_mirror_symmetry,
)


def test_efficiency_ratio_monotonic_and_saw():
    # Monotonic strictly increasing
    mono = [100.0, 101.0, 102.0, 103.0, 105.0]
    er_mono, status_mono = compute_efficiency_ratio(mono)
    assert status_mono == "VALID"
    assert math.isclose(er_mono, 1.0, abs_tol=1e-6)

    # Perfect symmetric saw (whipsaw oscillating around 100)
    saw = [100.0, 105.0, 100.0, 105.0, 100.0]
    er_saw, status_saw = compute_efficiency_ratio(saw)
    assert status_saw == "VALID"
    assert math.isclose(er_saw, 0.0, abs_tol=1e-6)

    # Constant flat price: division by zero safe
    flat = [100.0, 100.0, 100.0, 100.0]
    er_flat, status_flat = compute_efficiency_ratio(flat)
    assert status_flat == "QUIET_FLAT"
    assert er_flat == 0.0

    # Insufficient history
    short = [100.0, 101.0]
    er_short, status_short = compute_efficiency_ratio(short, min_periods=3)
    assert status_short == "INSUFFICIENT_HISTORY"
    assert np.isnan(er_short)


def test_normalized_slope():
    prices = [100.0, 102.0, 104.0, 106.0]
    slope, status = compute_normalized_slope(prices, sigma=1.0, time_horizon_min=4.0)
    assert status == "VALID"
    assert slope == (106.0 - 100.0) / (1.0 * 2.0)  # 6.0 / 2.0 = 3.0

    # Division by zero safety
    slope_zero, status_zero = compute_normalized_slope([100.0, 100.0], sigma=0.0)
    assert status_zero == "INSUFFICIENT_VOLATILITY"
    assert np.isnan(slope_zero)


def test_sign_change_frequency_filters_zero_returns():
    # Alternating prices: returns are +5, -5, +5, -5 (all flips)
    oscillating = [100.0, 105.0, 100.0, 105.0, 100.0]
    freq, status = compute_return_sign_changes(oscillating)
    assert status == "VALID"
    assert math.isclose(freq, 1.0, abs_tol=1e-6)

    # Monotonic: returns are all positive (0 flips)
    trending = [100.0, 102.0, 104.0, 106.0]
    freq_tr, status_tr = compute_return_sign_changes(trending)
    assert status_tr == "VALID"
    assert math.isclose(freq_tr, 0.0, abs_tol=1e-6)

    # Flat prices interspersed: [100, 100, 105, 105, 100, 100]
    # Active diffs are +5, -5 (1 flip out of 1 transition)
    with_zeros = [100.0, 100.0, 105.0, 105.0, 100.0, 100.0, 105.0]
    freq_z, status_z = compute_return_sign_changes(with_zeros, min_active_returns=3)
    assert status_z == "VALID"
    assert freq_z == 1.0  # +5 -> -5 -> +5 (2 flips out of 2 transitions)

    # All flat: zero returns do NOT create artificial saw
    all_flat = [100.0, 100.0, 100.0, 100.0]
    freq_flat, status_flat = compute_return_sign_changes(all_flat)
    assert status_flat == "INSUFFICIENT_ACTIVE_RETURNS"
    assert np.isnan(freq_flat)


def test_return_autocorrelation():
    # Oscillating: +5, -5, +5, -5 -> negative autocorrelation
    saw = [100.0, 105.0, 100.0, 105.0, 100.0, 105.0, 100.0]
    ac, status = compute_return_autocorrelation(saw, lag=1)
    assert status == "VALID"
    assert ac < -0.8  # Strong negative autocorrelation

    # Trend: +2, +2, +2, +2 -> zero variance or positive
    trend = [100.0, 102.0, 105.0, 109.0, 114.0, 120.0]
    ac_tr, status_tr = compute_return_autocorrelation(trend, lag=1)
    assert status_tr == "VALID"
    assert ac_tr > -0.1


def test_classify_local_regime_4_states():
    # 1. TREND
    trend_prices = [100.0, 102.0, 104.5, 107.0, 110.0]
    res_trend = classify_local_regime(trend_prices, min_observations=4)
    assert res_trend["state"] == "TREND"
    assert res_trend["status"] == "CONFIRMED_TREND"

    # 2. REVERSION
    saw_prices = [100.0, 105.0, 99.0, 104.5, 100.2, 105.1, 99.8]
    res_rev = classify_local_regime(saw_prices, min_observations=4)
    assert res_rev["state"] == "REVERSION"
    assert res_rev["status"] == "CONFIRMED_REVERSION"

    # 3. QUIET (flat market with micro-noise, must NOT become REVERSION)
    quiet_prices = [100.0, 100.00001, 100.0, 100.00002, 100.0]
    res_quiet = classify_local_regime(quiet_prices, min_observations=4)
    assert res_quiet["state"] == "QUIET"
    assert res_quiet["status"] == "QUIET_MARKET"

    # 4. UNCERTAIN (too short)
    short_prices = [100.0, 101.0]
    res_short = classify_local_regime(short_prices, min_observations=4)
    assert res_short["state"] == "UNCERTAIN"
    assert res_short["status"] == "INSUFFICIENT_HISTORY"


def test_strike_context_reversion_favorable_and_unfavorable():
    strike = 68000.0
    sigma_min = 0.001
    time_left_min = 5.0

    # Case A: Candidate DOWN, spot = 68200 (out-of-the-money), local_mean = 67900 (< strike)
    # Reversion towards 67900 moves downward AND reaches winning side of strike!
    ctx_a = compute_strike_context(
        spot=68200.0,
        strike=strike,
        sigma_min=sigma_min,
        time_left_min=time_left_min,
        local_mean=67900.0,
        candidate_side="DOWN",
    )
    assert ctx_a["status"] == "VALID"
    assert ctx_a["reversion_helps_strike"] is True
    assert ctx_a["reversion_reason"] == "FAVORABLE_REVERSION_CROSSES_OR_SETTLES_IN_MONEY"

    # Case B: Candidate DOWN, spot = 68500, local_mean = 68300 (> strike)
    # Reversion moves downward towards 68300, BUT local mean is STILL above strike (> 68000).
    # Self-check Item 16: Ending on losing side of strike is NEVER marked favorable!
    ctx_b = compute_strike_context(
        spot=68500.0,
        strike=strike,
        sigma_min=sigma_min,
        time_left_min=time_left_min,
        local_mean=68300.0,
        candidate_side="DOWN",
    )
    assert ctx_b["status"] == "VALID"
    assert ctx_b["reversion_helps_strike"] is False
    assert ctx_b["reversion_reason"] == "REVERSION_STAYS_ON_LOSING_SIDE"

    # Case C: Candidate DOWN, spot = 68200, local_mean = 68400 (drifts further away)
    ctx_c = compute_strike_context(
        spot=68200.0,
        strike=strike,
        sigma_min=sigma_min,
        time_left_min=time_left_min,
        local_mean=68400.0,
        candidate_side="DOWN",
    )
    assert ctx_c["status"] == "VALID"
    assert ctx_c["reversion_helps_strike"] is False
    assert ctx_c["reversion_reason"] == "REVERSION_DRIFTS_FURTHER_LOSING"

    # Case D: Candidate UP, spot = 67800 (out-of-the-money), local_mean = 68100 (> strike)
    ctx_d = compute_strike_context(
        spot=67800.0,
        strike=strike,
        sigma_min=sigma_min,
        time_left_min=time_left_min,
        local_mean=68100.0,
        candidate_side="UP",
    )
    assert ctx_d["status"] == "VALID"
    assert ctx_d["reversion_helps_strike"] is True


def test_strike_context_invalid_inputs_no_fake_constants():
    # Zero or negative sigma
    ctx_zero_sig = compute_strike_context(
        spot=68000.0,
        strike=68000.0,
        sigma_min=0.0,
        time_left_min=5.0,
        local_mean=68000.0,
    )
    assert ctx_zero_sig["status"] == "INVALID_OR_MISSING_INPUTS"
    assert np.isnan(ctx_zero_sig["z_strike"])
    assert ctx_zero_sig["reversion_helps_strike"] is False

    # Negative time left
    ctx_neg_time = compute_strike_context(
        spot=68000.0,
        strike=68000.0,
        sigma_min=0.001,
        time_left_min=-1.0,
    )
    assert ctx_neg_time["status"] == "INVALID_OR_MISSING_INPUTS"
    assert np.isnan(ctx_neg_time["z_strike"])


def test_mirror_symmetry_verification():
    # Oscillating around strike 68000
    p = [68100.0, 68300.0, 68050.0, 68250.0, 68120.0]
    res_sym = verify_mirror_symmetry(
        price_path=p,
        strike=68000.0,
        sigma_min=0.001,
        time_left_min=5.0,
    )
    assert res_sym["passed"] is True
    assert res_sym["slope_symmetry"] is True
    assert res_sym["er_symmetry"] is True
    assert res_sym["z_symmetry"] is True
    assert res_sym["reversion_help_symmetry"] is True
