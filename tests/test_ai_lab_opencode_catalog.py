"""Dynamic OpenCode model catalog behavior (T02)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

import polyflip.ai_lab.llm_catalog as llm_catalog
from polyflip.ai_lab.llm_catalog import normalize_models, refresh_model_catalog
from polyflip.config import settings
from polyflip.db.models import AILLMModelCatalog


def _cfg(**overrides):
    base = {
        "AI_LAB_OPENCODE_MODELS_ENDPOINT": "http://opencode.test/models",
        "AI_LAB_OPENCODE_CATALOG_TTL_SECONDS": 3600,
        "AI_LAB_OPENCODE_MODELS_FALLBACK": "",
        "AI_LAB_LLM_API_KEY": "test-key",
        "OPENAI_API_KEY": "",
        "AI_LAB_LLM_PROVIDER": "opencode",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_fetch(monkeypatch, result):
    calls: list[str] = []

    async def fake_fetch(endpoint_url, api_key, **kwargs):
        calls.append(endpoint_url)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(llm_catalog, "fetch_opencode_models", fake_fetch)
    return calls


def _payload(*models: dict) -> dict:
    return {"data": list(models)}


@pytest.mark.asyncio
async def test_normalize_models_supports_data_and_models_shapes():
    data_shape = _payload(
        {"id": "model-a"},
        {"id": "", "name": "named-only"},
        {"id": "", "name": ""},
    )
    models_shape = {
        "models": [
            {"id": "model-b", "name": "Model B"},
            {"id": "model-a"},
        ]
    }

    normalized_data = normalize_models(data_shape)
    normalized_models = normalize_models(models_shape)
    # Per the plan contract, an empty id falls back to the row name.
    assert [item["model_id"] for item in normalized_data] == [
        "model-a",
        "named-only",
    ]
    assert [item["model_id"] for item in normalized_models] == ["model-a", "model-b"]
    assert normalized_models[1]["display_name"] == "Model B"
    assert all(item["protocol"] == "responses" for item in normalized_models)
    metadata = normalize_models(
        _payload(
            {
                "id": "model-meta",
                "owned_by": "team",
                "capabilities": {"json": True},
                "api_key": "must-not-persist",
            }
        )
    )[0]["raw_metadata"]
    assert metadata["owned_by"] == "team"
    assert metadata["capabilities"] == {"json": True}
    assert "api_key" not in metadata
    assert normalize_models({"data": "not-a-list"}) == []
    assert normalize_models(None) == []


@pytest.mark.asyncio
async def test_refresh_live_upsert_marks_missing_rows_unavailable(
    db_session, monkeypatch
):
    _make_fetch(
        monkeypatch,
        _payload({"id": "model-b", "name": "Model B"}, {"id": "model-a"}),
    )
    cfg = _cfg()

    first = await refresh_model_catalog(
        db_session, provider="opencode", refresh=True, settings_obj=cfg
    )
    assert first["source"] == "live"
    assert first["stale"] is False
    ids_first = {item["id"]: item for item in first["models"]}
    assert set(ids_first) == {"model-a", "model-b"}
    assert ids_first["model-b"]["label"] == "Model B"

    # Next discovery drops model-a: the cached row must become unavailable.
    _make_fetch(monkeypatch, _payload({"id": "model-b", "name": "Model B"}))
    second = await refresh_model_catalog(
        db_session, provider="opencode", refresh=True, settings_obj=cfg
    )
    ids_second = {item["id"]: item for item in second["models"]}
    assert set(ids_second) == {"model-a", "model-b"}
    assert ids_second["model-a"]["is_available"] is False

    rows = (
        (await db_session.execute(sa.select(AILLMModelCatalog)))
        .scalars().all()
    )
    assert {row.model_id: row.is_available for row in rows} == {
        "model-a": False,
        "model-b": True,
    }


@pytest.mark.asyncio
async def test_refresh_serves_stale_cache_when_endpoint_fails(
    db_session, monkeypatch
):
    _make_fetch(monkeypatch, _payload({"id": "cached-model"}))
    cfg = _cfg()
    await refresh_model_catalog(
        db_session, provider="opencode", refresh=True, settings_obj=cfg
    )

    # Expire the cache so the next refresh must hit the failing endpoint.
    now = datetime.now(timezone.utc)
    for row in (
        (await db_session.execute(sa.select(AILLMModelCatalog)))
        .scalars().all()
    ):
        row.expires_at = now - timedelta(seconds=1)

    _make_fetch(monkeypatch, RuntimeError("endpoint down"))
    result = await refresh_model_catalog(
        db_session, provider="opencode", refresh=True, settings_obj=cfg
    )
    assert result["source"] == "cache"
    assert result["stale"] is True
    assert [item["id"] for item in result["models"]] == ["cached-model"]
    assert result["error"]


@pytest.mark.asyncio
async def test_refresh_uses_fallback_list_without_cache(db_session, monkeypatch):
    _make_fetch(monkeypatch, RuntimeError("endpoint down"))
    cfg = _cfg(AI_LAB_OPENCODE_MODELS_FALLBACK="big-pickle,nemotron-3-ultra-free")

    result = await refresh_model_catalog(
        db_session, provider="opencode", refresh=True, settings_obj=cfg
    )
    assert result["source"] == "fallback"
    assert result["stale"] is True
    assert [item["id"] for item in result["models"]] == [
        "big-pickle",
        "nemotron-3-ultra-free",
    ]


@pytest.mark.asyncio
async def test_llm_models_endpoint_returns_dynamic_catalog(db_session, monkeypatch):
    """GET /api/ai-lab/llm/models exposes live OpenCode models (T02 criterion)."""
    from polyflip.api.ai_lab import list_llm_models

    _make_fetch(
        monkeypatch,
        _payload({"id": "real-account-model", "name": "Real Model"}),
    )
    monkeypatch.setattr(
        settings, "AI_LAB_OPENCODE_MODELS_ENDPOINT", "http://opencode.test/models"
    )

    payload = await list_llm_models(
        provider="opencode", refresh=True, db=db_session
    )

    assert payload["source"] == "live"
    assert payload["stale"] is False
    assert [item["id"] for item in payload["models"]] == ["real-account-model"]
    assert payload["defaults"]["research_model"] == "real-account-model"
    assert payload["checked_at"]
