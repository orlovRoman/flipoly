import math

from polyflip.trading.weighted_sizing import stepped_bet_size
from polyflip.trading.weighted_policy import (
    WeightedPolicyConfig,
    compute_net_ev_per_share,
    estimate_trade_cost,
    logit,
    market_yes_probability,
    score_weighted_probability,
    select_weighted_side,
    sigmoid,
)


def test_stepped_bet_size_keeps_base_for_weak_lower_bound_and_caps_levels():
    assert stepped_bet_size(-0.10) == 1.0
    assert stepped_bet_size(0.03) == 1.5
    assert stepped_bet_size(0.06) == 2.0
    assert stepped_bet_size(0.10) == 3.0
    assert stepped_bet_size(0.20, cap_usdc=2.25) == 2.25
    assert stepped_bet_size(float("nan")) == 1.0




def test_logit_residual_formula_uses_market_as_prior():
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

    expected = sigmoid(
        logit(0.60)
        + 0.05 * (logit(0.70) - logit(0.60))
        + 0.05 * (logit(0.80) - logit(0.60))
    )
    assert math.isclose(result.p_final_yes, expected, rel_tol=1e-7)
    assert result.missing_components == ()
    assert math.isclose(result.market_weight, 0.90, rel_tol=1e-9)


def test_missing_model_weight_is_absorbed_by_market_prior():
    result = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.70,
        p_lgbm_yes=None,
        config=WeightedPolicyConfig(),
    )

    expected = sigmoid(logit(0.60) + 0.05 * (logit(0.70) - logit(0.60)))
    assert math.isclose(result.p_final_yes, expected, rel_tol=1e-7)
    assert result.missing_components == ("lgbm",)
    assert math.isclose(result.lgbm_weight, 0.0)
    assert math.isclose(result.market_weight, 0.95)


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


def test_mrf_stake_application_never_changes_probability():
    result = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.60,
        p_lgbm_yes=0.60,
        config=WeightedPolicyConfig(mrf_beta=0.5, mrf_application="STAKE"),
        mrf_evidence=-1.0,
    )

    assert result.p_final_yes == 0.60
    assert result.mrf_adjustment_logodds == 0.0


def test_mrf_stake_application_disables_learned_stacker_mrf_term():
    feature_names = (
        "intercept",
        "market_logit",
        "logreg_residual",
        "lgbm_residual",
        "mrf_evidence",
        "role_outsider",
        "models_agree",
        "outsider_agree",
        "outsider_logreg_residual",
        "outsider_lgbm_residual",
    )
    coefficients = (0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    probability = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.60,
        p_lgbm_yes=0.60,
        config=WeightedPolicyConfig(
            stacker_feature_names=feature_names,
            stacker_coefficients=coefficients,
            mrf_application="PROBABILITY",
        ),
        mrf_evidence=-1.0,
    )
    stake = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.60,
        p_lgbm_yes=0.60,
        config=WeightedPolicyConfig(
            stacker_feature_names=feature_names,
            stacker_coefficients=coefficients,
            mrf_application="STAKE",
        ),
        mrf_evidence=-1.0,
    )
    assert probability.p_final_yes < 0.60
    assert stake.p_final_yes == 0.60
    assert stake.mrf_evidence == -1.0
    assert stake.mrf_adjustment_logodds == 0.0


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


def test_market_prior_normalizes_both_executable_asks():
    result = market_yes_probability(yes_ask=0.55, no_ask=0.47, fallback_yes=0.90)

    assert math.isclose(result, 0.55 / (0.55 + 0.47), rel_tol=1e-7)


def test_contributions_sum_to_final_log_odds_and_record_agreement():
    result = score_weighted_probability(
        p_market_yes=0.55,
        p_logreg_yes=0.70,
        p_lgbm_yes=0.40,
        config=WeightedPolicyConfig(mrf_beta=0.2, intercept=0.1),
        mrf_evidence=-0.5,
    )

    assert result.models_agree is False
    assert math.isclose(
        sum(result.contributions.values()),
        logit(result.p_final_yes),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )


def test_net_ev_and_compatibility_edge_use_per_share_usdc():
    costs = estimate_trade_cost(
        0.60, fee_rate=0.0, slippage_rate=0.0, latency_buffer=0.01
    )
    assert compute_net_ev_per_share(0.70, 0.60, costs) == 0.09

    selection = select_weighted_side(
        p_market_yes=0.70,
        p_logreg_yes=0.70,
        p_lgbm_yes=0.70,
        yes_ask=0.60,
        no_ask=0.40,
        config=WeightedPolicyConfig(
            fee_rate=0.0, slippage_rate=0.0, latency_buffer=0.01
        ),
    )
    assert selection.selected is not None
    assert selection.selected.net_edge == selection.selected.net_ev_per_share


def test_models_agree_beta_is_a_separate_auditable_contribution():
    base = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.70,
        p_lgbm_yes=0.80,
        config=WeightedPolicyConfig(models_agree_beta=0.0),
    )
    boosted = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.70,
        p_lgbm_yes=0.80,
        config=WeightedPolicyConfig(models_agree_beta=0.25),
    )
    disagree = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.70,
        p_lgbm_yes=0.40,
        config=WeightedPolicyConfig(models_agree_beta=0.25),
    )
    assert base.models_agree is True
    assert boosted.models_agree_adjustment_logodds == 0.25
    assert boosted.p_final_yes > base.p_final_yes
    assert disagree.models_agree is False
    assert disagree.models_agree_adjustment_logodds == 0.0


def test_hierarchical_segment_coefficients_override_global_model():
    names = (
        "intercept",
        "market_logit",
        "logreg_residual",
        "lgbm_residual",
        "mrf_evidence",
        "role_outsider",
        "models_agree",
        "outsider_agree",
        "outsider_logreg_residual",
        "outsider_lgbm_residual",
    )
    global_coefficients = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    segment_coefficients = (-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    config = WeightedPolicyConfig(
        stacker_feature_names=names,
        stacker_coefficients=global_coefficients,
        stacker_segment_models=(
            ("BTCUSDT|MID_VOL|FAVORITE|AGREE", segment_coefficients),
        ),
    )

    result = score_weighted_probability(
        p_market_yes=0.60,
        p_logreg_yes=0.60,
        p_lgbm_yes=0.60,
        config=config,
        asset="BTCUSDT",
        phase="mid_vol",
        role="FAVORITE",
    )

    assert result.p_final_yes == 0.35559502
    assert result.p_final_yes != sigmoid(1.0 + logit(0.60))
