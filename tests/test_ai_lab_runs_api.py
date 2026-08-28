import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from polyflip.api.auth import verify_api_key
from polyflip.api.main import app
from polyflip.db.connection import get_db_session
from polyflip.db.models import AIOptimizationRun, AIRunStep, ExperimentResult


async def _request(db_session, method, path):
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[verify_api_key] = lambda: True
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


@pytest.mark.asyncio
async def test_ai_runs_list_supports_page_pagination(db_session):
    for index in range(3):
        db_session.add(
            AIOptimizationRun(
                objective=f"pagination-{index}",
                scope={"asset": "BTCUSDT"},
                mode="RESEARCH",
                autonomy_level="EXPERIMENT",
                status="FAILED",
                budget_experiments=1,
            )
        )
    await db_session.commit()

    first = await _request(db_session, "GET", "/api/ai-lab/runs?page=1&page_size=2")
    second = await _request(db_session, "GET", "/api/ai-lab/runs?page=2&page_size=2")

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["total"] == 3
    assert first_payload["pages"] == 2
    assert first_payload["page"] == 1
    assert first_payload["has_prev"] is False
    assert first_payload["has_next"] is True
    assert len(first_payload["runs"]) == 2
    assert second_payload["page"] == 2
    assert second_payload["has_prev"] is True
    assert second_payload["has_next"] is False
    assert len(second_payload["runs"]) == 1
    assert first_payload["runs"][0]["id"] > second_payload["runs"][0]["id"]


@pytest.mark.asyncio
async def test_ai_run_delete_removes_run_owned_rows(db_session):
    run = AIOptimizationRun(
        objective="delete me",
        scope={"asset": "BTCUSDT"},
        mode="RESEARCH",
        autonomy_level="EXPERIMENT",
        status="FAILED",
        budget_experiments=1,
    )
    db_session.add(run)
    await db_session.flush()
    step = AIRunStep(
        run_id=run.id,
        step_index=0,
        step_type="TRAIN_MODEL",
        status="FAILED",
    )
    db_session.add(step)
    await db_session.flush()
    result = ExperimentResult(
        run_id=run.id,
        step_id=step.id,
        evaluation_kind="TRAIN",
        status="FAILED",
    )
    db_session.add(result)
    await db_session.commit()

    response = await _request(db_session, "DELETE", f"/api/ai-lab/runs/{run.id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "run_id": run.id}
    assert await db_session.get(AIOptimizationRun, run.id) is None
    assert (
        await db_session.execute(select(AIRunStep).where(AIRunStep.run_id == run.id))
    ).scalars().all() == []
    assert (
        await db_session.execute(
            select(ExperimentResult).where(ExperimentResult.run_id == run.id)
        )
    ).scalars().all() == []


@pytest.mark.asyncio
async def test_ai_run_delete_rejects_active_run(db_session):
    run = AIOptimizationRun(
        objective="keep me",
        scope={"asset": "BTCUSDT"},
        mode="RESEARCH",
        autonomy_level="EXPERIMENT",
        status="RUNNING",
        budget_experiments=1,
    )
    db_session.add(run)
    await db_session.commit()

    response = await _request(db_session, "DELETE", f"/api/ai-lab/runs/{run.id}")

    assert response.status_code == 409
    assert await db_session.get(AIOptimizationRun, run.id) is not None
