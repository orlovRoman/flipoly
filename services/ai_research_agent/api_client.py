"""HTTP client for the AI Lab agent API (no DB, no shell, no docker)."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from schemas import AgentContext, ClaimedRun, ExperimentResult


class AgentAPIError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"AI Lab API {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class LeaseLostError(AgentAPIError):
    def __init__(self) -> None:
        super().__init__(409, "LEASE_LOST")


class AILabApiClient:
    """Thin typed wrapper over ``/api/ai-lab/agent/*``."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 30.0,
        poll_seconds: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds
        self._lease_token: str | None = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> Any:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.request(
                method, url, json=json_body, params=params, headers=self._headers
            )
        # Centralize LEASE_LOST handling: clear token and raise typed error.
        if response.status_code == 409:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            if "LEASE_LOST" in str(detail):
                self._lease_token = None
                raise LeaseLostError()
        if response.status_code in expected:
            if not response.content:
                return None
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise AgentAPIError(response.status_code, str(detail))

    def drop_lease(self) -> None:
        self._lease_token = None

    # --- lifecycle -------------------------------------------------------
    async def claim(self) -> ClaimedRun | None:
        import os
        import socket

        worker_id = os.getenv("AI_LAB_AGENT_WORKER_ID") or socket.gethostname()
        payload = await self._request(
            "POST", "/api/ai-lab/agent/claim", json_body={"worker_id": worker_id}
        )
        run = (payload or {}).get("run")
        if not run:
            return None
        self._lease_token = run.get("lease_token")
        return ClaimedRun.model_validate(run)

    async def heartbeat(self, run_id: int) -> float:
        data = await self._request(
            "POST",
            "/api/ai-lab/agent/heartbeat",
            json_body={"run_id": run_id, "lease_token": self._lease_token},
        )
        leased_until = str((data or {}).get("leased_until") or "")
        if not leased_until:
            return 0.0
        try:
            expires_at = datetime.fromisoformat(leased_until.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, remaining)
        except (TypeError, ValueError):
            # A malformed server timestamp must not keep a worker alive.
            return 0.0

    def require_lease(self) -> str:
        if not self._lease_token:
            raise AgentAPIError(409, "no active lease; claim a run first")
        return self._lease_token

    # --- research flow ---------------------------------------------------
    async def get_context(self, run_id: int) -> AgentContext:
        data = await self._request(
            "GET",
            f"/api/ai-lab/agent/runs/{run_id}/context",
            params={"lease_token": self._lease_token},
        )
        return AgentContext.model_validate(data or {})

    async def get_phase(self, run_id: int) -> dict[str, Any]:
        data = await self._request(
            "GET",
            f"/api/ai-lab/agent/runs/{run_id}/phase",
            params={"lease_token": self._lease_token},
        )
        return data or {}

    async def get_result(self, run_id: int) -> dict[str, Any] | None:
        data = await self._request(
            "GET",
            f"/api/ai-lab/agent/runs/{run_id}/result",
            params={"lease_token": self._lease_token},
        )
        return data

    async def submit_proposal(
        self,
        run_id: int,
        proposal: dict[str, Any],
        *,
        client_request_id: str | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> dict:
        lease_token = self.require_lease()
        request_id = client_request_id or __import__("uuid").uuid4().hex
        return await self._request(
            "POST",
            f"/api/ai-lab/agent/runs/{run_id}/proposal",
            json_body={
                "lease_token": lease_token,
                "client_request_id": request_id,
                "proposal": proposal,
                "telemetry": telemetry,
            },
        )

    async def wait_for_experiment_result(
        self,
        run_id: int,
        *,
        timeout_seconds: int,
        context: AgentContext | None = None,
    ) -> ExperimentResult | None:
        deadline = time.monotonic() + max(timeout_seconds, 1)
        while time.monotonic() < deadline:
            await self._request(
                "POST",
                "/api/ai-lab/agent/heartbeat",
                json_body={"run_id": run_id, "lease_token": self._lease_token},
            )
            try:
                data = await self._request(
                    "GET",
                    f"/api/ai-lab/agent/runs/{run_id}/result",
                    params={"lease_token": self._lease_token},
                )
            except AgentAPIError as exc:
                if exc.status_code != 404:
                    raise
                data = {"result": None}
            result = (data or {}).get("result")
            if result:
                return ExperimentResult.model_validate(result)
            await asyncio.sleep(self._poll_seconds)
        return None

    async def submit_decision(
        self,
        run_id: int,
        decision: dict[str, Any],
        *,
        client_request_id: str | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> dict:
        lease_token = self.require_lease()
        request_id = client_request_id or __import__("uuid").uuid4().hex
        return await self._request(
            "POST",
            f"/api/ai-lab/agent/runs/{run_id}/decision",
            json_body={
                "lease_token": lease_token,
                "client_request_id": request_id,
                "decision": decision,
                "telemetry": telemetry,
            },
        )

    async def complete(
        self,
        run_id: int,
        action: str,
        reason: str = "",
    ) -> dict:
        lease_token = self.require_lease()
        body: dict[str, Any] = {
            "action": action,
            "reason": reason,
            "lease_token": lease_token,
        }
        data = await self._request(
            "POST", f"/api/ai-lab/agent/runs/{run_id}/complete", json_body=body
        )
        self.drop_lease()
        return data or {}
