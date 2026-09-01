from datetime import datetime, timedelta, timezone

import pytest

from polyflip.trading.policy_artifact import (
    ActivationEvidence,
    activation_gate,
    create_policy_artifact,
    load_policy_artifact,
    save_policy_artifact,
)
from polyflip.trading.weighted_benchmark import (
    BenchmarkConfig,
    MarketObservation,
    benchmark,
    cluster_bootstrap_ci,
    deduplicate_observations,
    evaluate_arm,
    fit_ridge_logistic_stacker,
    fit_hierarchical_stackers,
    observation_segment_key,
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
    assert all(abs(value) <= 0.5 for value in model.coefficients)
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
    assert {item.arm for item in result.arms} >= {"MARKET_ONLY", "FULL_WEIGHTED_MRF"}


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
        )
    ).eligible


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


def test_hierarchical_stacker_has_role_and_agreement_segment():
    rows = [
        MarketObservation(
            **{
                **_row(index, index % 2 == 0).__dict__,
                "market_role": "UNDERDOG",
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
    assert len(model.global_model.coefficients) == 8
    assert model.predict_one(rows[0]) is not None
