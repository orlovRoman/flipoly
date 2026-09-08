"""OpenCode model availability probe behavior (T03)."""
from __future__ import annotations

import pytest
import sqlalchemy as sa
from types import SimpleNamespace

from polyflip.ai_lab.llm_catalog import (
    check_model_availability,
    persist_model_check_result,
)
from polyflip.db.models import AILLMModelCatalog


def _cfg(**overrides):
    base = {
        "AI_LAB_LLM_ENDPOINT": "",
        "AI_LAB_LLM_API_KEY": "test-key",
        "OPENAI_API_KEY": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _responses_payload(ok: bool = True) -> dict:
    return {"output_text": f'{{"ok": {str(ok).lower()}}}'}


def _chat_payload(ok: bool = True) -> dict:
    return {
        "choices": [
            {"message": {"content": f'{{"ok": {str(ok).lower()}}}'}}
        ]
    }


def _messages_payload(ok: bool = True) -> dict:
    return {
        "content": [
            {"type": "tool_use", "name": "model_probe", "input": {"ok": ok}}
        ]
    }


@pytest.mark.asyncio
async def test_mock_provider_is_available_without_network(db_session):
    report = await check_model_availability("mock", "mock-gpt-5")
    assert report["available"] is True
    assert report["protocol"] == "mock"


@pytest.mark.asyncio
async def test_responses_probe_success_is_persisted(db_session, monkeypatch):
    seen_urls: list[str] = []

    async def sender(*, url, headers, body):
        seen_urls.append(url)
        assert body["model"] == "probe-model"
        assert headers["Authorization"] == "Bearer test-key"
        return _responses_payload(True)

    report = await check_model_availability(
        "opencode", "probe-model", settings_obj=_cfg(), sender=sender
    )
    assert report["available"] is True
    assert report["protocol"] == "responses"
    assert report["latency_ms"] is not None and report["latency_ms"] >= 0

    row = await persist_model_check_result(
        db_session,
        provider="opencode",
        model_id="probe-model",
        report=report,
    )
    await db_session.commit()
    assert row.is_available is True
    stored = (
        await db_session.execute(sa.select(AILLMModelCatalog))
    ).scalar_one()
    assert stored.raw_metadata["last_check"]["available"] is True


@pytest.mark.asyncio
async def test_invalid_structured_output_marks_unavailable(db_session, monkeypatch):
    async def sender(*, url, headers, body):
        return _responses_payload(False)

    report = await check_model_availability(
        "opencode", "bad-model", settings_obj=_cfg(), sender=sender
    )
    assert report["available"] is False
    assert report["protocol"] is None
    assert report["error"]

    row = await persist_model_check_result(
        db_session, provider="opencode", model_id="bad-model", report=report
    )
    assert row.is_available is False


@pytest.mark.asyncio
async def test_chat_completions_fallback_when_responses_fails(monkeypatch):
    attempted: list[str] = []

    async def sender(*, url, headers, body):
        attempted.append(url)
        if url.endswith("/chat/completions"):
            return _chat_payload(True)
        raise RuntimeError("responses endpoint down")

    report = await check_model_availability(
        "opencode", "chat-only-model", settings_obj=_cfg(), sender=sender
    )
    assert len(attempted) == 2
    assert report["available"] is True
    assert report["protocol"] == "chat_completions"


@pytest.mark.asyncio
async def test_go_messages_probe_uses_go_endpoint_and_tool_output(monkeypatch):
    async def sender(*, url, headers, body):
        assert url == "https://opencode.ai/zen/go/v1/messages"
        assert body["model"] == "minimax-m3"
        assert body["tool_choice"] == {"type": "tool", "name": "model_probe"}
        assert headers["anthropic-version"] == "2023-06-01"
        return _messages_payload(True)

    report = await check_model_availability(
        "opencode",
        "minimax-m3",
        settings_obj=_cfg(),
        sender=sender,
    )

    assert report["available"] is True
    assert report["protocol"] == "messages"


@pytest.mark.asyncio
async def test_custom_endpoint_override_is_respected_first(monkeypatch):
    async def sender(*, url, headers, body):
        assert url == "http://custom.test/v1/chat/completions"
        return _chat_payload(True)

    report = await check_model_availability(
        "opencode",
        "override-model",
        settings_obj=_cfg(AI_LAB_LLM_ENDPOINT="http://custom.test/v1/chat/completions"),
        sender=sender,
    )
    assert report["available"] is True


@pytest.mark.asyncio
async def test_unknown_provider_rejected(db_session):
    with pytest.raises(ValueError):
        await check_model_availability("unknown-provider", "x")


@pytest.mark.asyncio
async def test_check_endpoint_rejects_unknown_provider(db_session):
    """Route maps ValueError from probe to HTTP 422 without network calls."""
    import pytest
    from fastapi import HTTPException

    from polyflip.api.ai_lab import check_llm_model

    with pytest.raises(HTTPException) as exc_info:
        await check_llm_model(
            provider="unknown-provider", model_id="x", db=db_session
        )
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_opencode_probe_sends_session_header(db_session):
    import uuid

    captured_headers = {}

    async def sender(*, url, headers, body):
        captured_headers.update(headers)
        return _responses_payload(True)

    report = await check_model_availability(
        "opencode", "grok-4.6", settings_obj=_cfg(), sender=sender
    )
    assert report["available"] is True
    assert "x-opencode-session" in captured_headers
    # Verify it is a valid UUID
    session_id = captured_headers["x-opencode-session"]
    uuid.UUID(session_id)


@pytest.mark.asyncio
async def test_default_probe_sender_extracts_error_detail(monkeypatch):
    import httpx
    from polyflip.ai_lab.llm_catalog import _default_probe_sender

    class _FakeResponse:
        status_code = 400
        text = '{"error": {"type": "MissingSessionID", "message": "Missing session header"}}'

        def json(self):
            return {"error": {"type": "MissingSessionID", "message": "Missing session header"}}

        def raise_for_status(self):
            req = httpx.Request("POST", "https://opencode.ai/zen/go/v1/responses")
            resp = httpx.Response(400, request=req, text=self.text)
            raise httpx.HTTPStatusError("Client error '400 Bad Request'", request=req, response=resp)

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, headers=None, json=None):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: _FakeClient())

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _default_probe_sender("https://opencode.ai/zen/go/v1/responses", {}, {})

    assert "Missing session header" in str(exc_info.value)
