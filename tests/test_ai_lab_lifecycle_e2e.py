import json

import pytest
from sqlalchemy import select

from polyflip.ai_lab.executor import AdapterRegistry, AdapterResult, execute_steps
from polyflip.ai_lab.orchestrator import evaluate_run, finalize_run, plan_run
from polyflip.ai_lab.service import create_experiment_config, create_run
from polyflip.db.models import AIModelArtifact, AIPermission, ModelRegistry


WINDOWS = [
    {
        "start": "2026-08-01T00:00:00+00:00",
        "end": "2026-08-01T01:00:00+00:00",
        "net_pnl": 1.0,
        "trade_count": 5,
        "max_drawdown": 1.0,
    },
    {
        "start": "2026-08-02T00:00:00+00:00",
        "end": "2026-08-02T01:00:00+00:00",
        "net_pnl": 1.5,
        "trade_count": 5,
        "max_drawdown": 1.0,
    },
    {
        "start": "2026-08-03T00:00:00+00:00",
        "end": "2026-08-03T01:00:00+00:00",
        "net_pnl": 2.0,
        "trade_count": 5,
        "max_drawdown": 1.0,
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_family", ["LogisticRegression", "LightGBM"])
async def test_research_lifecycle_dispatches_family_and_promotes_valid_candidate(
    db_session, model_family
):
    permission = AIPermission(
        profile_name=f"e2e-{model_family.lower()}",
        version=1,
        is_current=True,
        enabled=True,
        allowed_actions=[
            "CREATE_EXPERIMENT",
            "TRAIN_MODEL",
            "RUN_OOT_BACKTEST",
            "RUN_POLYMARKET_OOT",
            "PROMOTE_TO_SHADOW",
        ],
        scope={"assets": ["BTCUSDT"]},
        limits={"max_runs": 1},
        updated_by="test",
    )
    db_session.add(permission)
    await db_session.flush()

    run = await create_run(
        db_session,
        objective=f"AI Lab lifecycle {model_family}",
        scope={"asset": "BTCUSDT", "min_trades": 50, "min_windows": 3},
        autonomy_level="AUTONOMOUS_SHADOW",
        budget_experiments=1,
        permission=permission,
        mode="RESEARCH",
        created_by="test",
    )
    config = await create_experiment_config(
        db_session,
        name=f"e2e-{model_family.lower()}",
        asset="BTCUSDT",
        regime="DEFAULT",
        model_family=model_family,
        feature_set="FS_D0",
        feature_pipeline_version="1.0",
        model_params={},
        strategy_params={},
        backtest_params={},
        config_hash=f"{model_family.lower()}-e2e-config-hash",
        created_by="test",
    )
    steps = await plan_run(db_session, run.id, [config.id])
    assert [step.action for step in steps] == [
        "TRAIN_MODEL",
        "RUN_OOT_BACKTEST",
        "RUN_POLYMARKET_OOT",
    ]
    await db_session.flush()

    artifact = AIModelArtifact(
        config_id=config.id,
        run_id=run.id,
        step_id=steps[0].id,
        artifact_bytes=b"offline-e2e-artifact",
        artifact_hash=f"{model_family.lower()}-e2e-artifact-hash",
        sha256=f"{model_family.lower()}-e2e-sha256",
        schema_version="1",
        feature_pipeline_version="1.0",
        artifact_metadata={"config_id": config.id, "model_family": model_family},
        loadability_status="VALID",
    )
    db_session.add(artifact)
    await db_session.flush()

    seen_families = []

    async def train(context):
        seen_families.append(context.model_family)
        return AdapterResult(
            evaluation_kind="TRAIN",
            artifact_id=artifact.id,
            metrics={"auc": 0.61, "ece": 0.02},
            summary=f"saved {model_family} artifact",
        )

    async def generic_oot(context):
        seen_families.append(context.model_family)
        return AdapterResult(
            evaluation_kind="OOT",
            artifact_id=artifact.id,
            metrics={"auc": 0.61, "ece": 0.02},
            summary="diagnostic OOT",
        )

    async def polymarket_oot(context):
        seen_families.append(context.model_family)
        return AdapterResult(
            evaluation_kind="POLYMARKET_OOT",
            artifact_id=artifact.id,
            metrics={"oot_windows": WINDOWS},
            slices={"oot_windows": WINDOWS},
            trade_count=15,
            net_pnl=4.5,
            max_drawdown=1.0,
            summary="saved-artifact Polymarket OOT",
        )

    registry = (
        AdapterRegistry()
        .register("TRAIN_MODEL", train)
        .register("RUN_OOT_BACKTEST", generic_oot)
        .register("RUN_POLYMARKET_OOT", polymarket_oot)
    )
    outcomes = await execute_steps(
        db_session,
        run.id,
        registry,
        max_steps=3,
        owner_token=f"e2e-{model_family}",
    )

    assert len(outcomes) == 3
    assert all(outcome.status == "SUCCEEDED" for outcome in outcomes)
    assert seen_families == [model_family] * 3

    report = await evaluate_run(db_session, run.id)
    assert run.status == "EVALUATING"
    assert report["recommendation_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["shadow_recommendation_status"] == "RESEARCH_PROVISIONAL"
    assert report["eligible_for_shadow"] is True

    result = await finalize_run(db_session, run.id)
    assert result["assignment"] is not None
    assert run.status == "SHADOW"
    assert result["assignment"].candidate_artifact_id == artifact.id
    summary = json.loads(run.summary)
    assert summary["status"] == "RESEARCH_PROVISIONAL"
    assert summary["report"]["recommendation_status"] == "INSUFFICIENT_EVIDENCE"

    active_models = (
        await db_session.execute(
            select(ModelRegistry).where(ModelRegistry.is_active.is_(True))
        )
    ).scalars().all()
    assert active_models == []
