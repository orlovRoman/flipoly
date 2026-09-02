from datetime import datetime, timedelta, timezone

import pytest

from polyflip.trading.policy_artifact import (
    ActivationEvidence,
    activation_gate,
    create_policy_artifact,
    load_policy_artifact,
    save_policy_artifact,
    weighted_policy_config_from_artifact,
)
from polyflip.trading.weighted_benchmark import (
    BenchmarkConfig,
    MarketObservation,
    benchmark,
    cluster_bootstrap_ci,
    deduplicate_observations,
    evaluate_arm,
    estimate_oof_standard_error,
    evaluate_sizing_steps,
    filter_fixed_horizons,
    fit_ridge_logistic_stacker,
    fit_hierarchical_stackers,
    observation_segment_key,
    optimize_min_net_ev,
    optimize_price_cap,
    optimize_time_window,
    compare_mrf_application,
    compare_kelly_fractions,
    optimize_mrf_beta,
    compare_outsider_agreement,
    create_policy_artifact_from_benchmark,
    parameter_sensitivity,
    stability_by_segment,
    fingerprint_observations,
    purged_walk_forward_folds,
)
from polyflip.trading.weighted_policy import WeightedPolicyConfig
from polyflip.trading.weighted_sizing import (
    conservative_size,
    probability_lower_bound,
)


def _row(index: int, outcome: bool = True) -> MarketObservation:
    return MarketObservation(
        market_id=f"m-{index}",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index),
        asset="BTC",
        yes_ask=0.60,
        no_ask=0.40,
        outcome_yes=outcome,
        p_market_yes=0.70,
        p_logreg_yes=0.72,
        p_lgbm_yes=0.74,
        mrf_evidence=-0.5,
        group=f"day-{index // 4}",
    )


def test_mapping_preserves_signed_mrf_evidence():
    item = MarketObservation.from_mapping(
        {
            "market_id": "m-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "asset": "BTC",
            "yes_ask": 0.6,
            "no_ask": 0.4,
            "outcome_yes": "YES",
            "weighted_mrf_evidence": -0.75,
        }
    )
    assert item.mrf_evidence == -0.75
    assert item.outcome_yes is True


def test_walk_forward_folds_have_purge_gap():
    rows = [_row(i, i % 2 == 0) for i in range(12)]
    folds = purged_walk_forward_folds(rows, train_min_rows=4, test_size=3, purge_gap=2)
    assert folds[0].train_indices == (0, 1, 2, 3)
    assert folds[0].test_indices == (6, 7, 8)
    assert set(folds[0].train_indices).isdisjoint(folds[0].test_indices)


def test_stacker_is_bounded_and_benchmark_reports_cost_aware_arms():
    rows = [_row(i, i % 3 != 0) for i in range(16)]
    model = fit_ridge_logistic_stacker(rows, coefficient_bound=0.5)
    assert model.coefficients[1] == 1.0
    assert all(
        abs(value) <= 0.5
        for index, value in enumerate(model.coefficients)
        if index != 1
    )
    result = benchmark(
        rows,
        config=BenchmarkConfig(
            policy_config=WeightedPolicyConfig(
                fee_rate=0.0,
                slippage_rate=0.0,
            ),
            train_min_rows=4,
            test_size=4,
            bootstrap_iterations=20,
        ),
    )
    assert result.resolved_observations == 16
    assert result.stacker is not None
    assert result.stacker.coefficients[1] == 1.0
    assert all(
        abs(value) <= 5.0
        for index, value in enumerate(result.stacker.coefficients)
        if index != 1
    )
    assert {item.arm for item in result.arms} >= {"MARKET_ONLY", "FULL_WEIGHTED_MRF"}
    assert [item["stake_usdc"] for item in result.sizing_steps] == [1.0, 1.5, 2.0, 3.0]
    assert [item["fraction_percent"] for item in result.kelly_fractions] == [2.5, 5.0, 10.0]
    assert {item["parameter"] for item in result.tuning} >= {
        "min_net_ev_favorite",
        "min_net_ev_outsider",
        "favorite_max_price",
        "outsider_max_price",
        "time_left_favorite",
        "time_left_outsider",
        "mrf_application",
        "mrf_beta",
        "outsider_agreement",
    }


def test_mrf_beta_and_outsider_agreement_are_compared_on_oot_rows():
    rows = [
        MarketObservation(
            **{
                **_row(index, index % 2 == 0).__dict__,
                "market_role": "OUTSIDER",
                "yes_ask": 0.40,
                "no_ask": 0.60,
            }
        )
        for index in range(8)
    ]
    folds = purged_walk_forward_folds(rows, train_min_rows=4, test_size=2)
    beta = optimize_mrf_beta(
        rows,
        candidate_values=(0.0, 0.2),
        folds=folds,
        minimum_stable_folds=1,
    )
    agreement = compare_outsider_agreement(rows, folds=folds)
    assert beta.parameter == "mrf_beta"
    assert beta.selected in {0.0, 0.2}
    assert agreement["parameter"] == "outsider_agreement"
    assert {item["value"] for item in agreement["candidates"]} == {
        "FULL_WEIGHTED_MRF",
        "OUTSIDER_AGREE_ONLY",
    }
    assert "difference" in agreement
    assert agreement["selected"] == "FULL_WEIGHTED_MRF"
    assert agreement["difference"]["statistically_better"] is False


def test_legacy_arm_replays_persisted_action_instead_of_reselecting_weighted_side():
    rows = [
        MarketObservation(
            **{
                **_row(0, True).__dict__,
                "p_legacy_yes": 0.55,
                "legacy_action": "BUY_NO",
                "legacy_ask": 0.30,
                "yes_ask": 0.70,
                "no_ask": 0.30,
            }
        )
    ]
    metrics = evaluate_arm(
        rows,
        "LEGACY",
        config=WeightedPolicyConfig(fee_rate=0.0, slippage_rate=0.0),
    )
    assert metrics.trades == 1
    assert metrics.evaluations[0].side == "BUY_NO"
    assert metrics.evaluations[0].ask == 0.30


def test_legacy_action_is_normalized_from_exported_final_action():
    item = MarketObservation.from_mapping(
        {
            "market_id": "legacy-action",
            "timestamp": "2026-01-01T00:00:00Z",
            "asset": "BTC",
            "yes_ask": 0.60,
            "no_ask": 0.40,
            "outcome_yes": "YES",
            "p_logreg_yes": 0.70,
            "final_action": "BUY_YES",
            "candidate_ask": 0.60,
        }
    )
    assert item.legacy_action == "BUY_YES"
    assert item.legacy_ask == 0.60


def test_evaluate_arm_uses_observed_cost_and_ci_is_reproducible():
    rows = [_row(0, True)]
    rows = [
        MarketObservation(**{**item.__dict__, "observed_cost_per_share": 0.05})
        for item in rows
    ]
    metrics = evaluate_arm(
        rows,
        "MARKET_ONLY",
        config=WeightedPolicyConfig(fee_rate=0.0, slippage_rate=0.0),
    )
    assert metrics.net_pnl == 0.35
    assert cluster_bootstrap_ci(metrics.evaluations, iterations=20, seed=7) == cluster_bootstrap_ci(
        metrics.evaluations, iterations=20, seed=7
    )


def test_sizing_sanitizes_non_finite_uncertainty_and_cost_inputs():
    assert probability_lower_bound(0.70, float("nan")) == 0.0
    assert probability_lower_bound(0.70, float("inf")) == 0.0
    decision = conservative_size(
        0.70,
        price=float("nan"),
        cost_per_share=float("nan"),
        standard_error=float("nan"),
    )
    assert decision.p_lower == 0.0
    assert decision.edge_lower == -0.5
    assert decision.size_multiplier == 0.0


def test_artifact_is_immutable_and_sizing_uses_lower_bound(tmp_path):
    artifact = create_policy_artifact(
        version="test-v1",
        created_at="2026-01-01T00:00:00+00:00",
        training_window={"rows": 10},
        stacker=None,
        policy_config=WeightedPolicyConfig(),
        thresholds={"min_net_ev": 0.03},
    )
    path = tmp_path / "policy.json"
    save_policy_artifact(path, artifact)
    assert load_policy_artifact(path).artifact_id == artifact.artifact_id
    with pytest.raises(ValueError):
        path.write_text(path.read_text().replace("test-v1", "test-v2"), encoding="utf-8")
        load_policy_artifact(path)
    assert probability_lower_bound(0.70, 0.05) < 0.70
    decision = conservative_size(
        0.70,
        price=0.60,
        cost_per_share=0.01,
        standard_error=0.01,
    )
    assert decision.p_lower < decision.p_estimate


def test_artifact_config_is_immutable_and_overrides_runtime_fallback():
    artifact = create_policy_artifact(
        version="test-config-v1",
        created_at="2026-01-01T00:00:00+00:00",
        training_window={"rows": 10},
        stacker=None,
        policy_config=WeightedPolicyConfig(
            market_weight=0.75,
            logreg_weight=0.15,
            lgbm_weight=0.10,
            fee_rate=0.02,
        ),
        thresholds={"min_net_ev_favorite": 0.04},
    )
    loaded = artifact
    runtime = weighted_policy_config_from_artifact(
        loaded,
        fallback=WeightedPolicyConfig(market_weight=0.90, fee_rate=0.07),
    )
    assert runtime.policy_id == artifact.artifact_id
    assert runtime.market_weight == 0.75
    assert runtime.logreg_weight == 0.15
    assert runtime.fee_rate == 0.02


def test_activation_gate_requires_plan_evidence():
    assert not activation_gate(ActivationEvidence()).eligible
    assert activation_gate(
        ActivationEvidence(
            shadow_days=14,
            shadow_resolved_markets=1000,
            shadow_candidate_trades=300,
            repeat_oot_reports=1,
            live_fills=300,
            pnl_ci_lower=0.01,
            weighted_brier=0.10,
            market_brier=0.103,
            legacy_brier=0.104,
            weighted_net_pnl=10.0,
            market_net_pnl=5.0,
            legacy_net_pnl=4.0,
            execution_drag=0.005,
            calibration_error=0.02,
        )
    ).eligible


def test_activation_gate_rejects_missing_quality_evidence():
    evidence = ActivationEvidence(
        shadow_days=14,
        shadow_resolved_markets=1000,
        shadow_candidate_trades=300,
        repeat_oot_reports=1,
        live_fills=300,
        pnl_ci_lower=0.01,
    )
    gate = activation_gate(evidence)
    assert not gate.eligible
    assert "BRIER_EVIDENCE_MISSING" in gate.reasons
    assert "PNL_COMPARISON_MISSING" in gate.reasons
    assert "EXECUTION_DRAG_MISSING" not in gate.reasons
    assert "CALIBRATION_ERROR_MISSING" not in gate.reasons


def test_activation_gate_allows_first_fixed_bet_before_live_validation():
    gate = activation_gate(
        ActivationEvidence(
            shadow_days=14,
            shadow_resolved_markets=1000,
            shadow_candidate_trades=300,
            repeat_oot_reports=1,
            live_fills=0,
            pnl_ci_lower=0.01,
            weighted_brier=0.10,
            market_brier=0.103,
            legacy_brier=0.104,
            weighted_net_pnl=10.0,
            market_net_pnl=5.0,
            legacy_net_pnl=4.0,
        )
    )
    assert gate.eligible


def test_activation_gate_requires_t57_live_validation_when_requested():
    gate = activation_gate(
        ActivationEvidence(
            shadow_days=14,
            shadow_resolved_markets=1000,
            shadow_candidate_trades=300,
            repeat_oot_reports=1,
            live_fills=299,
            pnl_ci_lower=0.01,
            weighted_brier=0.10,
            market_brier=0.103,
            legacy_brier=0.104,
            weighted_net_pnl=10.0,
            market_net_pnl=5.0,
            legacy_net_pnl=4.0,
            execution_drag=0.005,
            calibration_error=0.02,
        ),
        require_live_validation=True,
    )
    assert not gate.eligible
    assert "LIVE_FILLS_BELOW_MINIMUM" in gate.reasons


def test_deduplicate_keeps_one_row_per_market_and_horizon():
    first = MarketObservation(
        **{**_row(0).__dict__, "horizon": "15M"}
    )
    later = MarketObservation(
        **{
            **first.__dict__,
            "timestamp": first.timestamp + timedelta(minutes=1),
        }
    )
    other_horizon = MarketObservation(
        **{**first.__dict__, "horizon": "1H"}
    )
    result = deduplicate_observations([later, other_horizon, first])
    assert len(result) == 2
    assert first in result
    assert later not in result


def test_hierarchical_stacker_has_role_phase_and_agreement_segment():
    rows = [
        MarketObservation(
            **{
                **_row(index, index % 2 == 0).__dict__,
                "market_role": "UNDERDOG",
                "phase": "DECIDED",
            }
        )
        for index in range(12)
    ]
    model = fit_hierarchical_stackers(
        rows,
        min_segment_rows=4,
        shrinkage=300.0,
    )
    key = observation_segment_key(rows[0])
    assert key in model.segment_models
    assert "DECIDED" in key
    assert "CONTESTED" not in key
    assert model.segment_models[key].training_markets == 12
    assert len(model.global_model.coefficients) == 10
    assert model.global_model.feature_names[-2:] == (
        "outsider_logreg_residual",
        "outsider_lgbm_residual",
    )
    assert model.predict_one(rows[0]) is not None


def test_hierarchical_minimum_uses_independent_markets_not_horizon_rows():
    rows = []
    for market_index in range(2):
        for horizon in ("10M", "5M", "2M"):
            rows.append(
                MarketObservation(
                    **{
                        **_row(len(rows), market_index % 2 == 0).__dict__,
                        "market_id": f"same-{market_index}",
                        "horizon": horizon,
                        "market_role": "UNDERDOG",
                        "phase": "DECIDED",
                    }
                )
            )
    model = fit_hierarchical_stackers(rows, min_segment_rows=3, shrinkage=300.0)
    assert observation_segment_key(rows[0]) not in model.segment_models


def test_walk_forward_never_splits_one_market_between_train_and_test():
    rows = []
    for market_index in range(6):
        for horizon in ("10M", "5M"):
            rows.append(
                MarketObservation(
                    **{
                        **_row(len(rows), market_index % 2 == 0).__dict__,
                        "market_id": f"market-{market_index}",
                        "horizon": horizon,
                    }
                )
            )
    folds = purged_walk_forward_folds(
        rows,
        train_min_rows=4,
        test_size=2,
        purge_gap=1,
    )
    assert folds
    for fold in folds:
        train_markets = {rows[index].market_id for index in fold.train_indices}
        test_markets = {rows[index].market_id for index in fold.test_indices}
        assert train_markets.isdisjoint(test_markets)


def test_benchmark_arm_metrics_are_out_of_time_only():
    rows = [_row(index, index % 2 == 0) for index in range(12)]
    result = benchmark(
        rows,
        config=BenchmarkConfig(
            policy_config=WeightedPolicyConfig(
                fee_rate=0.0,
                slippage_rate=0.0,
            ),
            train_min_rows=4,
            test_size=4,
            bootstrap_iterations=10,
        ),
    )
    market_only = next(item for item in result.arms if item.arm == "MARKET_ONLY")
    assert market_only.observations == 8
    assert all(
        evaluation.market_id in {f"m-{index}" for index in range(4, 12)}
        for evaluation in market_only.evaluations
    )


def test_fixed_horizon_filter_accepts_aliases_and_excludes_other_windows():
    rows = [
        MarketObservation(**{**_row(0).__dict__, "horizon": "10"}),
        MarketObservation(**{**_row(1).__dict__, "horizon": "5min"}),
        MarketObservation(**{**_row(2).__dict__, "horizon": "1H"}),
    ]
    filtered = filter_fixed_horizons(rows)
    assert [item.horizon for item in filtered] == ["10", "5min"]


def test_tuning_helpers_use_oot_folds_and_role_specific_constraints():
    rows = [
        MarketObservation(
            **{
                **_row(index, True).__dict__,
                "market_id": f"tune-{index}",
                "market_role": "FAVORITE",
                "time_left_sec": 60.0 + index * 30.0,
            }
        )
        for index in range(12)
    ]
    folds = purged_walk_forward_folds(
        rows,
        train_min_rows=4,
        test_size=2,
        purge_gap=0,
    )
    threshold = optimize_min_net_ev(
        rows,
        role="FAVORITE",
        candidate_values=(0.01, 0.03),
        folds=folds,
        minimum_stable_folds=1,
    )
    cap = optimize_price_cap(
        rows,
        role="FAVORITE",
        candidate_values=(0.75, 0.90),
        folds=folds,
        minimum_stable_folds=1,
    )
    window = optimize_time_window(
        rows,
        role="FAVORITE",
        windows=((30.0, 300.0), (60.0, 600.0)),
        folds=folds,
        minimum_stable_folds=1,
    )
    assert threshold.parameter == "min_net_ev_favorite"
    assert threshold.selected in {0.01, 0.03}
    assert cap.parameter == "favorite_max_price"
    assert cap.selected in {0.75, 0.90}
    assert window.parameter == "time_left_favorite"
    assert window.selected in ([30.0, 300.0], [60.0, 600.0])
    assert threshold.candidates[0]["folds"] == len(folds)


def test_mrf_application_comparison_reports_both_methods():
    rows = [_row(index, index % 2 == 0) for index in range(8)]
    folds = purged_walk_forward_folds(rows, train_min_rows=4, test_size=2)
    comparison = compare_mrf_application(rows, folds=folds, beta=0.2, gamma=0.2)
    assert comparison["selected"] in {"probability_adjustment", "stake_adjustment"}
    assert comparison["folds"] == len(folds)
    assert comparison["probability_adjustment"]["arm"] == "FULL_WEIGHTED_MRF"
    assert comparison["stake_adjustment"]["arm"] == "FULL_WEIGHTED"


def test_observation_mapping_tolerates_invalid_time_and_preserves_horizon():
    item = MarketObservation.from_mapping(
        {
            "market_id": "m-invalid-time",
            "timestamp": "2026-01-01T00:00:00Z",
            "asset": "BTC",
            "yes_ask": "0.6",
            "no_ask": "0.4",
            "outcome_yes": "YES",
            "horizon": "5min",
            "time_left_sec": "unknown",
        }
    )
    assert item.horizon == "5M"
    assert item.time_left_sec is None


def test_parameter_sensitivity_uses_fixed_oot_indices_and_records_shifts():
    rows = [
        MarketObservation(
            **{
                **_row(index, index % 2 == 0).__dict__,
                "market_role": "OUTSIDER" if index % 2 else "FAVORITE",
                "phase": "SIDEWAYS" if index % 2 else "STRONG_UP",
            }
        )
        for index in range(8)
    ]
    result = parameter_sensitivity(
        rows,
        config=WeightedPolicyConfig(fee_rate=0.01),
        parameters=("market_weight", "mrf_beta"),
        deltas=(-0.10, 0.10),
        evaluation_indices=(4, 5, 6, 7),
    )
    assert len(result) == 4
    assert {item["parameter"] for item in result} == {"market_weight", "mrf_beta"}
    assert {item["delta"] for item in result} == {-0.1, 0.1}
    assert all(item["trades"] >= 0 for item in result)


def test_stability_report_splits_required_dimensions_and_adds_ci():
    rows = [
        MarketObservation(
            **{
                **_row(index, index % 2 == 0).__dict__,
                "asset": "BTC" if index < 4 else "ETH",
                "market_role": "OUTSIDER" if index % 2 else "FAVORITE",
                "phase": "SIDEWAYS" if index % 2 else "STRONG_UP",
                "asset_phase": "BTC_SIDEWAYS" if index < 4 else "ETH_STRONG_UP",
                "horizon": "10M" if index % 2 else "5M",
                "execution_role": "MAKER" if index % 2 else "TAKER",
            }
        )
        for index in range(8)
    ]
    result = stability_by_segment(
        rows,
        arm="FULL_WEIGHTED_MRF",
        config=WeightedPolicyConfig(fee_rate=0.0, slippage_rate=0.0),
        evaluation_indices=range(4, 8),
    )
    dimensions = {item["dimension"] for item in result}
    assert dimensions == {
        "asset",
        "market_role",
        "phase",
        "asset_phase",
        "horizon",
        "execution_role",
        "consensus",
        "week",
    }
    assert all("pnl_ci_low" in item and "pnl_ci_high" in item for item in result)
    assert all("roi" in item for item in result)
    assert all(item["observations"] > 0 for item in result)


def test_mapping_preserves_phase_fields_for_stability_reports():
    item = MarketObservation.from_mapping(
        {
            "market_id": "m-phase",
            "timestamp": "2026-01-01T00:00:00Z",
            "asset": "BTC",
            "yes_ask": 0.6,
            "no_ask": 0.4,
            "outcome_yes": "YES",
            "market_phase": "strong_up",
            "mrf_asset_phase": "btc_strong_up",
        }
    )
    assert item.phase == "STRONG_UP"
    assert item.asset_phase == "BTC_STRONG_UP"


def test_oof_standard_error_is_reproducible_and_bounded():
    rows = [_row(index, index % 2 == 0) for index in range(8)]
    estimate = estimate_oof_standard_error(
        rows,
        {index: 0.75 for index in range(8)},
        evaluation_indices=(4, 5, 6, 7),
    )
    assert estimate is not None
    assert 0.0 < estimate <= 0.5




def test_sizing_step_report_uses_same_oot_sample_for_each_level():
    rows = [
        MarketObservation(
            **{
                **_row(index, index % 2 == 0).__dict__,
                "yes_ask": 0.48,
                "no_ask": 0.52,
            }
        )
        for index in range(8)
    ]
    result = evaluate_sizing_steps(
        rows,
        config=WeightedPolicyConfig(fee_rate=0.0, slippage_rate=0.0),
        levels=(1.0, 1.5, 2.0, 3.0),
        evaluation_indices=(4, 5, 6, 7),
        bootstrap_iterations=10,
    )
    assert [item["stake_usdc"] for item in result] == [1.0, 1.5, 2.0, 3.0]
    assert {item["trades"] for item in result} == {4}
    assert result[1]["net_pnl"] == result[0]["net_pnl"] * 1.5
    assert all(item["pnl_ci_low"] is not None for item in result)




def test_kelly_fraction_report_compares_fixed_oot_sample():
    rows = [
        MarketObservation(
            **{
                **_row(index, index % 2 == 0).__dict__,
                "yes_ask": 0.48,
                "no_ask": 0.52,
            }
        )
        for index in range(8)
    ]
    result = compare_kelly_fractions(
        rows,
        config=WeightedPolicyConfig(fee_rate=0.0, slippage_rate=0.0),
        fractions=(0.025, 0.05, 0.10),
        evaluation_indices=(4, 5, 6, 7),
        bootstrap_iterations=10,
    )
    assert [item["fraction"] for item in result] == [0.025, 0.05, 0.1]
    assert [item["fraction_percent"] for item in result] == [2.5, 5.0, 10.0]
    assert all("max_drawdown" in item for item in result)
    assert all("pnl_ci_low" in item for item in result)


def test_benchmark_fingerprint_binds_artifact_to_exact_dataset():
    rows = [_row(index, index % 2 == 0) for index in range(12)]
    report = benchmark(
        rows,
        config=BenchmarkConfig(
            train_min_rows=4,
            test_size=4,
            bootstrap_iterations=10,
        ),
    )
    artifact = create_policy_artifact_from_benchmark(
        rows,
        report,
        version="benchmark-v1",
        policy_config=WeightedPolicyConfig(fee_rate=0.01),
        thresholds={"min_net_ev_favorite": 0.03},
    )
    assert report.dataset_fingerprint == fingerprint_observations(rows)
    assert artifact.dataset_fingerprint == report.dataset_fingerprint
    assert artifact.training_window["observations"] == len(rows)
    assert artifact.model["feature_names"]


def test_benchmark_filters_labeled_non_fixed_horizons_before_fingerprint():
    rows = [
        MarketObservation(**{**_row(0).__dict__, "horizon": "10M"}),
        MarketObservation(**{**_row(1).__dict__, "horizon": "1H"}),
    ]
    report = benchmark(
        rows,
        config=BenchmarkConfig(train_min_rows=1, test_size=1, bootstrap_iterations=5),
    )
    assert report.observations == 1
    assert report.dataset_fingerprint == fingerprint_observations(
        (rows[0],)
    )
