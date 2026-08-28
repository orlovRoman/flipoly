from __future__ import annotations

import os
import sys

import pytest

SERVICES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "ai_research_agent")
)
if os.path.isdir(SERVICES_DIR):
    sys.path.insert(0, SERVICES_DIR)

from opencode_client import OpenCodeClient

from polyflip.ai_lab.llm import (
    DEFAULT_OPENROUTER_MODELS,
    get_llm_model_catalog,
)
from polyflip.ai_lab.llm_catalog import (
    check_model_availability,
    fetch_opencode_models,
    normalize_models,
)


def test_openrouter_static_catalog_contains_go_models(monkeypatch):
    from polyflip.config import settings

    monkeypatch.setattr(
        settings,
        "AI_LAB_LLM_AVAILABLE_PROVIDERS",
        "mock,openrouter",
    )
    monkeypatch.setattr(settings, "AI_LAB_ALLOWED_MODELS", "")
    monkeypatch.setattr(settings, "AI_LAB_OPENROUTER_MODELS", "")
    catalog = get_llm_model_catalog("openrouter")

    assert tuple(item["id"] for item in catalog["models"]) == DEFAULT_OPENROUTER_MODELS
    assert catalog["models"][0]["label"] == "Grok 4.6"
    longcat = next(item for item in catalog["models"] if item["id"] == "meituan/longcat-2.0")
    assert longcat["supports_structured_output"] is False
    assert all(item["protocol"] == "chat_completions" for item in catalog["models"])


def test_openrouter_normalizes_supported_parameters():
    rows = normalize_models(
        {
            "data": [
                {
                    "id": "meituan/longcat-2.0",
                    "name": "LongCat-2.0",
                    "supported_parameters": ["temperature", "tools"],
                },
                {
                    "id": "x-ai/grok-4.6",
                    "supported_parameters": ["response_format"],
                },
            ]
        },
        default_protocol="chat_completions",
    )

    by_id = {row["model_id"]: row for row in rows}
    assert by_id["x-ai/grok-4.6"]["supports_structured_output"] is True
    assert by_id["meituan/longcat-2.0"]["supports_structured_output"] is False
    assert all(row["protocol"] == "chat_completions" for row in rows)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeAsyncClient:
    def __init__(self, response):
        self.response = response
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, endpoint, *, headers, json):
        self.request = (endpoint, headers, json)
        return _FakeResponse(self.response)

    async def get(self, endpoint, *, headers):
        self.request = (endpoint, headers, None)
        return _FakeResponse(self.response)


@pytest.mark.asyncio
async def test_catalog_discovery_skips_non_ascii_api_key(monkeypatch):
    from polyflip.ai_lab import llm_catalog

    fake = _FakeAsyncClient({"data": [{"id": "x-ai/grok-4.6"}]})
    monkeypatch.setattr(llm_catalog.httpx, "AsyncClient", lambda **_: fake)

    payload = await fetch_opencode_models(
        "https://openrouter.ai/api/v1/models",
        "ключ-с-ошибкой",
    )

    assert payload["data"][0]["id"] == "x-ai/grok-4.6"
    assert "Authorization" not in fake.request[1]


@pytest.mark.asyncio
async def test_model_probe_reports_non_ascii_api_key_without_http_call():
    class _Config:
        OPENROUTER_API_KEY = "ключ-с-ошибкой"
        AI_LAB_LLM_API_KEY = ""
        AI_LAB_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    called = False

    async def sender(**kwargs):
        nonlocal called
        called = True
        return {}

    report = await check_model_availability(
        "openrouter",
        "x-ai/grok-4.6",
        settings_obj=_Config(),
        sender=sender,
    )

    assert report["available"] is False
    assert "non-ASCII" in report["error"]
    assert called is False


@pytest.mark.asyncio
async def test_external_client_reads_parsed_chat_message(monkeypatch):
    import opencode_client

    monkeypatch.setenv("AI_LAB_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("AI_LAB_LLM_API_KEY", "test-key")
    fake = _FakeAsyncClient(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "parsed": {
                            "action": "FINALIZE_NO_WINNER",
                            "rationale": "ok",
                        },
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    )
    monkeypatch.setattr(opencode_client.httpx, "AsyncClient", lambda **_: fake)
    client = OpenCodeClient()

    payload, telemetry = await client._structured_json(
        model="x-ai/grok-4.6",
        instructions="return JSON",
        context={},
        schema_name="agent_decision",
        schema={"type": "object"},
    )

    assert payload["action"] == "FINALIZE_NO_WINNER"
    assert telemetry["total_tokens"] == 5
    assert fake.request[0].endswith("/chat/completions")
    assert fake.request[2]["model"] == "x-ai/grok-4.6"


@pytest.mark.asyncio
async def test_external_client_routes_go_messages_model(monkeypatch):
    import opencode_client

    monkeypatch.setenv("AI_LAB_LLM_PROVIDER", "opencode")
    monkeypatch.setenv("AI_LAB_LLM_API_KEY", "test-key")
    fake = _FakeAsyncClient(
        {"content": [{"type": "tool_use", "input": {"ok": True}}]}
    )
    monkeypatch.setattr(opencode_client.httpx, "AsyncClient", lambda **_: fake)
    client = OpenCodeClient()

    payload, _telemetry = await client._structured_json(
        model="minimax-m3",
        instructions="return JSON",
        context={},
        schema_name="model_probe",
        schema={"type": "object"},
        protocol="messages",
    )

    assert payload == {"ok": True}
    assert fake.request[0] == "https://opencode.ai/zen/go/v1/messages"
    assert fake.request[1]["anthropic-version"] == "2023-06-01"
    assert fake.request[2]["model"] == "minimax-m3"

