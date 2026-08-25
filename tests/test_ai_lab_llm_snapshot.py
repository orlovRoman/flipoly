"""Immutable LLM selection snapshot on AI Lab run creation (T05)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from types import SimpleNamespace

from polyflip.ai_lab.llm_catalog import refresh_model_catalog
from polyflip.ai_lab.service import AILabError, create_run
from polyflip.db.models import AILLMModelCatalog, AIOptimizationRun


def _cfg():
    return SimpleNamespace(
        AI_LAB_LLM_PROVIDER="opencode",
        AI_LAB_MODEL_RESEARCH="gpt-5.6",
        AI_LAB_MODEL_SUMMARY="gpt-5.6-mini",
    )


async def _seed_available(db_session, model_id: str):
    db_session.add(
        AILLMModelCatalog(
            provider="opencode",
            model_id=model_id,
            display_name=model_id,
            is_available=True,
            discovered_at=datetime.now(timezone.utc),
        )
    )
    await db_session.flush()


async def _create(db_session, *, provider, research=None, summary=None):
    return await create_run(
        db_session,
        objective="snapshot contract test",
        scope={},
        autonomy_level="OBSERVE",
        budget_experiments=1,
        permission=None,
        llm_provider=provider,
        llm_research_model=research,
        llm_summary_model=summary,
    )


@pytest.mark.asyncio
async def test_dynamic_models_persist_immutable_snapshot(db_session, monkeypatch):
    async def fake_fetch(endpoint_url, api_key, **kwargs):
        return {
            "data": [
                {"id": "research-x", "name": "Research X", "protocol": "responses"},
                {"id": "summary-y", "name": "Summary Y", "protocol": "responses"},
            ]
        }

    monkeypatch.setattr("polyflip.ai_lab.llm_catalog.fetch_opencode_models", fake_fetch)
    cfg = SimpleNamespace(
        AI_LAB_OPENCODE_MODELS_ENDPOINT="http://opencode.test/models",
        AI_LAB_OPENCODE_CATALOG_TTL_SECONDS=3600,
        AI_LAB_OPENCODE_MODELS_FALLBACK="",
        AI_LAB_LLM_API_KEY="k",
        OPENAI_API_KEY="",
    )
    await refresh_model_catalog(
        db_session, provider="opencode", refresh=True, settings_obj=cfg
    )

    run = await _create(
        db_session, provider="opencode", research="research-x", summary="summary-y"
    )
    assert run.llm_snapshot["catalog_model_status"] == "available"
    assert run.llm_snapshot["protocol"] == "responses"
    assert run.llm_snapshot["catalog_checked_at"]
    stored = await db_session.get(AIOptimizationRun, run.id)
    assert stored.llm_research_model == "research-x"
    assert stored.llm_snapshot["summary_model"] == "summary-y"


@pytest.mark.asyncio
async def test_missing_or_unavailable_model_rejected(db_session):
    await _seed_available(db_session, "good-model")
    unavailable = AILLMModelCatalog(
        provider="opencode",
        model_id="broken-model",
        is_available=False,
        discovered_at=datetime.now(timezone.utc),
    )
    db_session.add(unavailable)
    await db_session.flush()

    with pytest.raises(AILabError, match="not present"):
        await _create(
            db_session, provider="opencode",
            research="unknown-model", summary="good-model",
        )
    with pytest.raises(AILabError, match="not available"):
        await _create(
            db_session, provider="opencode",
            research="good-model", summary="broken-model",
        )


@pytest.mark.asyncio
async def test_empty_catalog_falls_back_to_static_defaults(db_session):
    run = await _create(
        db_session, provider="opencode",
        research="big-pickle", summary="gpt-5.6-sol",
    )
    assert run.llm_snapshot["catalog_model_status"] == "legacy_static"
    # big-pickle is a Chat Completions model in the legacy default list.
    assert run.llm_snapshot["protocol"] == "chat_completions"


@pytest.mark.asyncio
async def test_snapshot_does_not_change_when_catalog_flips_later(db_session):
    await _seed_available(db_session, "stable-model")
    run = await _create(
        db_session, provider="opencode",
        research="stable-model", summary="stable-model",
    )
    before = dict(run.llm_snapshot)

    row = (
        await db_session.execute(
            sa.select(AILLMModelCatalog).where(
                AILLMModelCatalog.model_id == "stable-model"
            )
        )
    ).scalar_one()
    row.is_available = False
    row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await db_session.flush()

    stored = await db_session.get(AIOptimizationRun, run.id)
    assert stored.llm_snapshot == before


@pytest.mark.asyncio
async def test_mock_provider_defaults_snapshot(db_session):
    run = await _create(db_session, provider="mock")
    assert run.llm_snapshot["provider"] == "mock"
    assert run.llm_snapshot["catalog_model_status"] == "available"
