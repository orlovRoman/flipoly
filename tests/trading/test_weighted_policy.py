import math

from polyflip.trading.weighted_policy import (
    WeightedPolicyConfig,
    estimate_trade_cost,
    score_weighted_probability,
    select_weighted_side,
)


def test_direct_blend_uses_configured_weights():
    result = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.70,
        p_lgbm_yes=0.80,
        config=WeightedPolicyConfig(
            market_weight=0.90,
            logreg_weight=0.05,
            lgbm_weight=0.05,
            fee_rate=0.0,
            slippage_rate=0.0,
        ),
    )

    assert math.isclose(result.p_final_yes, 0.615, rel_tol=1e-9)
    assert result.missing_components == ()
    assert math.isclose(result.market_weight, 0.90, rel_tol=1e-9)


def test_missing_model_is_removed_and_remaining_weights_are_renormalized():
    result = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.70,
        p_lgbm_yes=None,
        config=WeightedPolicyConfig(),
    )

    assert math.isclose(
        result.p_final_yes,
        (0.90 * 0.60 + 0.05 * 0.70) / 0.95,
        rel_tol=1e-7,
    )
    assert result.missing_components == ("lgbm",)
    assert math.isclose(result.lgbm_weight, 0.0)


def test_all_zero_available_weights_fail_closed():
    result = select_weighted_side(
        p_market_yes=0.90,
        p_logreg_yes=None,
        p_lgbm_yes=None,
        yes_ask=0.10,
        no_ask=0.90,
        config=WeightedPolicyConfig(
            market_weight=0.0,
            logreg_weight=0.0,
            lgbm_weight=0.0,
        ),
    )

    assert result.selected is None
    assert result.reason == "NO_CONFIGURED_COMPONENT_WEIGHT"


def test_taker_fee_is_price_dependent_and_maker_fee_is_zero():
    taker = estimate_trade_cost(0.77, fee_rate=0.07, slippage_rate=0.0, role="TAKER")
    maker = estimate_trade_cost(0.77, fee_rate=0.07, slippage_rate=0.0, role="MAKER")

    assert math.isclose(taker.fee_per_share, 0.07 * 0.77 * 0.23, rel_tol=1e-8)
    assert maker.fee_per_share == 0.0
    assert taker.total_per_share > maker.total_per_share


def test_taker_fee_uses_market_fee_curve_exponent():
    estimate = estimate_trade_cost(
        0.77, fee_rate=0.07, fee_exponent=2.0, slippage_rate=0.0, role="TAKER"
    )

    assert math.isclose(
        estimate.fee_per_share, round(0.07 * (0.77 * 0.23) ** 2, 8), rel_tol=1e-8
    )


def test_selection_is_based_on_cost_aware_expected_value():
    result = select_weighted_side(
        p_market_yes=0.80,
        p_logreg_yes=0.80,
        p_lgbm_yes=0.80,
        yes_ask=0.75,
        no_ask=0.25,
        config=WeightedPolicyConfig(fee_rate=0.0, slippage_rate=0.0),
        min_net_ev=0.01,
    )

    assert result.selected is not None
    assert result.selected.side == "BUY_YES"
    assert result.selected.net_ev_per_share == 0.05
    assert result.no_quote is not None
    assert result.no_quote.net_ev_per_share < 0.0


def test_spread_is_included_in_both_policy_quotes():
    result = select_weighted_side(
        p_market_yes=0.80,
        p_logreg_yes=0.80,
        p_lgbm_yes=0.80,
        yes_ask=0.75,
        no_ask=0.25,
        config=WeightedPolicyConfig(fee_rate=0.0, slippage_rate=0.0),
        spread=0.03,
        min_net_ev=0.01,
    )

    assert result.selected is not None
    assert result.selected.side == "BUY_YES"
    assert result.selected.cost.spread_per_share == 0.03
    assert result.selected.net_ev_per_share == 0.02


def test_mrf_evidence_adjusts_log_odds_only_when_enabled():
    base = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.60,
        p_lgbm_yes=0.60,
        config=WeightedPolicyConfig(mrf_beta=0.0),
        mrf_evidence=-1.0,
    )
    adjusted = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.60,
        p_lgbm_yes=0.60,
        config=WeightedPolicyConfig(mrf_beta=0.5),
        mrf_evidence=-1.0,
    )

    assert base.p_final_yes == 0.60
    assert adjusted.p_final_yes < base.p_final_yes
