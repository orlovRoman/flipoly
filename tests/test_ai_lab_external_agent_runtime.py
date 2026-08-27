import os
import sys

import pytest

SERVICES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "services", "ai_research_agent"
)
sys.path.insert(0, os.path.abspath(SERVICES_DIR))

import api_client
import opencode_client


@pytest.mark.asyncio
async def test_mock_provider_never_constructs_http_client(monkeypatch):
    monkeypatch.setenv("AI_LAB_LLM_PROVIDER", "mock")

    class ForbiddenAsyncClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("mock provider must not perform network I/O")

    monkeypatch.setattr(opencode_client.httpx, "AsyncClient", ForbiddenAsyncClient)
    client = opencode_client.OpenCodeClient()
    context = {"scope": {"asset": "BTC"}}

    proposal_bundle = await client.propose_hypothesis(context)
    assert proposal_bundle["proposal"]["asset"] == "BTC"
    assert proposal_bundle["proposal"]["model_family"] == "LOGREG"
    decision_bundle = await client.decide(
        context=context,
        proposal=proposal_bundle["proposal"],
        result=None,
    )
    assert decision_bundle["decision"]["action"] == "FINALIZE_NO_WINNER"
    assert decision_bundle["telemetry"]["total_tokens"] == 0


@pytest.mark.asyncio
async def test_wait_for_result_does_not_send_duplicate_heartbeat(monkeypatch):
    client = api_client.AILabApiClient("http://api.test", "token", poll_seconds=0)
    client._lease_token = "lease"
    calls = []

    async def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {
            "state": "READY",
            "result": {
                "config_id": 7,
                "evaluation_kind": "POLYMARKET_OOT",
                "status": "SUCCEEDED",
            },
        }

    monkeypatch.setattr(client, "_request", request)
    result = await client.wait_for_experiment_result(7, timeout_seconds=1)

    assert result is not None
    assert calls == [
        (
            "GET",
            "/api/ai-lab/agent/runs/7/result",
            {"params": {"lease_token": "lease"}},
        )
    ]
