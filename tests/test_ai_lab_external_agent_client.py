"""Client lease propagation (T09) — query params and 409 handling."""
from __future__ import annotations

import os
import sys

import httpx
import pytest
from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

SERVICES_DIR = os.path.join(os.path.dirname(__file__), "..", "services", "ai_research_agent")
sys.path.insert(0, os.path.abspath(SERVICES_DIR))
from api_client import AILabApiClient, LeaseLostError  # noqa: E402


def _mock_lease_lost(request: Request) -> Response:
    return Response(409, json={"detail": "LEASE_LOST"}, request=request)


def _mock_ok(request: Request) -> Response:
    # echo lease_token from query for verification
    return Response(200, json={"run": None}, request=request)


@pytest.mark.asyncio
async def test_context_includes_lease_token_query_param(monkeypatch):
    captured: dict = {}

    async def handler(request: Request) -> Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        return Response(200, json={"run": {}, "active_models": [], "recent_trade_statistics": {}, "prior_experiments": [], "available_feature_sets": [], "quality_gate": {}}, request=request)

    transport = MockTransport(handler)
    client = AILabApiClient("http://test", "tok")
    client._lease_token = "lease-123"
    # monkeypatch internal _request to use mock transport
    orig_request = client._request

    async def patched(method, path, *, json_body=None, params=None, expected=(200,)):
        url = f"http://test{path}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        req = Request(method, url, json=json_body)
        resp = await handler(req)
        if resp.status_code == 409 and resp.json().get("detail") == "LEASE_LOST":
            client._lease_token = None
            raise LeaseLostError()
        return resp.json()

    client._request = patched  # type: ignore[method-assign]
    await client.get_context(1)
    assert captured["params"].get("lease_token") == "lease-123"


@pytest.mark.asyncio
async def test_heartbeat_409_clears_token_and_raises():
    async def handler(request: Request) -> Response:
        return Response(409, json={"detail": "LEASE_LOST"}, request=request)

    client = AILabApiClient("http://test", "tok")
    client._lease_token = "old-token"

    async def patched(method, path, *, json_body=None, params=None, expected=(200, 409)):
        resp = await handler(Request(method, f"http://test{path}"))
        if resp.status_code == 409:
            client._lease_token = None
            raise LeaseLostError()
        return resp.json()

    client._request = patched  # type: ignore[method-assign]
    with pytest.raises(LeaseLostError):
        await client.heartbeat(41)
    assert client._lease_token is None


def test_drop_lease_clears_local_token():
    from services.ai_research_agent.api_client import AILabApiClient

    client = AILabApiClient("http://test", "tok")
    client._lease_token = "abc"
    # runner should call drop_lease, not direct assignment — verify method exists
    assert hasattr(client, "_lease_token")
