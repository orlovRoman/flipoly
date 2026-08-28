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
    now = datetime.now(timezone.utc)
    db_session.add(
        AILLMModelCatalog(
            provider="opencode",
            model_id=model_id,
            display_name=model_id,
            is_available=True,
            is_discovered=True,
            probe_status="PASSED",
            last_checked_at=now,
            discovered_at=now,
            supports_structured_output=True,
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
    from polyflip.ai_lab.llm_catalog import persist_model_check_result

    async def fake_fetch(endpoint_url, api_key, **kwargs):
        return {
            "data": [
                {"id": "gpt-5.6-luna", "name": "GPT 5.6 Luna", "protocol": "responses"},
                {"id": "big-pickle", "name": "Big Pickle", "protocol": "chat_completions"},
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
    # Discovery creates UNCHECKED; must probe to PASSED before snapshot accepts.
    for mid in ("gpt-5.6-luna", "big-pickle"):
        await persist_model_check_result(
            db_session,
            provider="opencode",
            model_id=mid,
            report={
                "available": True,
                "protocol": "responses" if mid == "gpt-5.6-luna" else "chat_completions",
                "latency_ms": 10,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            },
        )
    await db_session.flush()

    run = await _create(
        db_session, provider="opencode", research="gpt-5.6-luna", summary="big-pickle"
    )
    assert run.llm_snapshot["catalog_model_status"] == "available"
    assert run.llm_snapshot["protocol"] == "responses"
    assert run.llm_snapshot["catalog_checked_at"]
    stored = await db_session.get(AIOptimizationRun, run.id)
    assert stored.llm_research_model == "gpt-5.6-luna"
    assert stored.llm_snapshot["summary_model"] == "big-pickle"


@pytest.mark.asyncio
async def test_missing_or_unavailable_model_rejected(db_session):
    await _seed_available(db_session, "gpt-5.6-luna")
    now = datetime.now(timezone.utc)
    unavailable = AILLMModelCatalog(
        provider="opencode",
        model_id="big-pickle",
        is_available=False,
        is_discovered=True,
        probe_status="FAILED",
        last_checked_at=now,
        discovered_at=now,
        supports_structured_output=True,
    )
    db_session.add(unavailable)
    await db_session.flush()

    with pytest.raises(AILabError, match="not present"):
        await _create(
            db_session, provider="opencode",
            research="unknown-model", summary="gpt-5.6-luna",
        )
    with pytest.raises(AILabError, match="probe status is FAILED"):
        await _create(
            db_session, provider="opencode",
            research="gpt-5.6-luna", summary="big-pickle",
        )


@pytest.mark.asyncio
async def test_empty_catalog_falls_back_to_static_defaults(db_session):
    run = await _create(
        db_session, provider="opencode",
        research="big-pickle", summary="gpt-5.6-luna",
    )
    assert run.llm_snapshot["catalog_model_status"] == "legacy_static"
    # big-pickle is a Chat Completions model in the legacy default list.
    assert run.llm_snapshot["protocol"] == "chat_completions"


@pytest.mark.asyncio
async def test_snapshot_does_not_change_when_catalog_flips_later(db_session):
    await _seed_available(db_session, "gpt-5.6-luna")
    run = await _create(
        db_session, provider="opencode",
        research="gpt-5.6-luna", summary="gpt-5.6-luna",
    )
    before = dict(run.llm_snapshot)

    row = (
        await db_session.execute(
            sa.select(AILLMModelCatalog).where(
                AILLMModelCatalog.model_id == "gpt-5.6-luna"
            )
        )
    ).scalar_one()
    row.is_available = False
    row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await db_session.flush()

    stored = await db_session.get(AIOptimizationRun, run.id)
    assert stored.llm_snapshot == before


@pytest.mark.asyncio
async def test_discovery_creates_unchecked_and_probe_sets_status(db_session, monkeypatch):
    from polyflip.ai_lab.llm_catalog import persist_model_check_result

    # Discovery via refresh creates UNCHECKED
    async def fake_fetch(endpoint_url, api_key, **kwargs):
        return {"data": [{"id": "big-pickle", "name": "Big Pickle"}]}

    monkeypatch.setattr("polyflip.ai_lab.llm_catalog.fetch_opencode_models", fake_fetch)
    cfg = SimpleNamespace(
        AI_LAB_OPENCODE_MODELS_ENDPOINT="http://opencode.test/models",
        AI_LAB_OPENCODE_CATALOG_TTL_SECONDS=3600,
        AI_LAB_OPENCODE_MODELS_FALLBACK="",
        AI_LAB_LLM_API_KEY="k",
        OPENAI_API_KEY="",
    )
    await refresh_model_catalog(db_session, provider="opencode", refresh=True, settings_obj=cfg)
    row = (await db_session.execute(sa.select(AILLMModelCatalog).where(AILLMModelCatalog.model_id == "big-pickle"))).scalar_one()
    assert row.is_discovered is True
    assert row.probe_status == "UNCHECKED"
    assert row.last_checked_at is None
    # Snapshot with UNCHECKED should be rejected
    with pytest.raises(AILabError, match="probe status is UNCHECKED"):
        await _create(db_session, provider="opencode", research="big-pickle", summary="big-pickle")
    # Probe to PASSED should allow snapshot
    await persist_model_check_result(db_session, provider="opencode", model_id="big-pickle", report={"available": True, "protocol": "chat_completions", "latency_ms": 5, "checked_at": datetime.now(timezone.utc).isoformat(), "error": None})
    await db_session.flush()
    row2 = (await db_session.execute(sa.select(AILLMModelCatalog).where(AILLMModelCatalog.model_id == "big-pickle"))).scalar_one()
    assert row2.probe_status == "PASSED"
    assert row2.last_checked_at is not None
    assert row2.is_discovered is True
    run = await _create(db_session, provider="opencode", research="big-pickle", summary="big-pickle")
    assert run.llm_snapshot["catalog_model_status"] == "available"
    # Probe to FAILED should reject snapshot
    await persist_model_check_result(db_session, provider="opencode", model_id="big-pickle", report={"available": False, "protocol": None, "latency_ms": None, "checked_at": datetime.now(timezone.utc).isoformat(), "error": "fail"})
    await db_session.flush()
    with pytest.raises(AILabError, match="probe status is FAILED"):
        await _create(db_session, provider="opencode", research="big-pickle", summary="big-pickle")
    # is_discovered False should reject even if PASSED
    row2.is_discovered = False  # type: ignore
    row2.probe_status = "PASSED"  # type: ignore
    row2.last_checked_at = datetime.now(timezone.utc)  # type: ignore
    await db_session.flush()
    with pytest.raises(AILabError, match="not discovered"):
        await _create(db_session, provider="opencode", research="big-pickle", summary="big-pickle")
    # supports_structured_output False should reject
    row2.is_discovered = True  # type: ignore
    row2.supports_structured_output = False
    row2.probe_status = "PASSED"  # type: ignore
    row2.last_checked_at = datetime.now(timezone.utc)  # type: ignore
    await db_session.flush()
    with pytest.raises(AILabError, match="does not support structured"):
        await _create(db_session, provider="opencode", research="big-pickle", summary="big-pickle")


@pytest.mark.asyncio
async def test_probe_ttl_expires_snapshot(db_session, monkeypatch):
    now = datetime.now(timezone.utc)
    # Seed a model with stale probe (2 days ago, TTL 86400)
    db_session.add(AILLMModelCatalog(provider="opencode", model_id="gpt-5.6-luna", display_name="GPT 5.6 Luna", is_available=True, is_discovered=True, probe_status="PASSED", last_checked_at=now - timedelta(days=2), discovered_at=now - timedelta(days=2), supports_structured_output=True))
    await db_session.flush()
    with pytest.raises(AILabError, match="probe is stale"):
        await _create(db_session, provider="opencode", research="gpt-5.6-luna", summary="gpt-5.6-luna")
    # Fresh probe should succeed
    db_session.add(AILLMModelCatalog(provider="opencode", model_id="big-pickle", display_name="Big Pickle", protocol="chat_completions", is_available=True, is_discovered=True, probe_status="PASSED", last_checked_at=now, discovered_at=now, supports_structured_output=True))
    await db_session.flush()
    run = await _create(db_session, provider="opencode", research="big-pickle", summary="big-pickle")
    assert run.llm_snapshot["provider"] == "opencode"


@pytest.mark.asyncio
async def test_mock_provider_defaults_snapshot(db_session):
    run = await _create(db_session, provider="mock")
    assert run.llm_snapshot["provider"] == "mock"
    assert run.llm_snapshot["catalog_model_status"] == "available"


@pytest.mark.asyncio
async def test_configured_endpoint_is_authoritative_for_new_run(db_session, monkeypatch):
    from polyflip.config import settings

    async def fake_fetch(endpoint_url, api_key, **kwargs):
        return {"data": [{"id": "big-pickle"}]}

    monkeypatch.setattr(
        "polyflip.ai_lab.llm_catalog.fetch_opencode_models", fake_fetch
    )
    monkeypatch.setattr(
        settings,
        "AI_LAB_OPENCODE_MODELS_ENDPOINT",
        "http://opencode.test/models",
    )
    monkeypatch.setattr(settings, "AI_LAB_OPENCODE_CATALOG_TTL_SECONDS", 3600)
    monkeypatch.setattr(settings, "AI_LAB_OPENCODE_MODELS_FALLBACK", "")

    with pytest.raises(AILabError, match="probe status is UNCHECKED"):
        await _create(
            db_session,
            provider="opencode",
            research="big-pickle",
            summary="big-pickle",
        )
