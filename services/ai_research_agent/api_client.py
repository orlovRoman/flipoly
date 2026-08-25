"""HTTP client for the AI Lab agent API (no DB, no shell, no docker)."""
from __future__ import annotations

import asyncio
import time
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
        expected: tuple[int, ...] = (200,),
    ) -> Any:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.request(
                method, url, json=json_body, headers=self._headers
            )
        if response.status_code in expected:
            if not response.content:
                return None
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise AgentAPIError(response.status_code, str(detail))

    # --- lifecycle -------------------------------------------------------
    async def claim(self) -> ClaimedRun | None:
        payload = await self._request("POST", "/api/ai-lab/agent/claim", json_body={})
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
            expected=(200, 409),
        )
        if data and data.get("detail") == "LEASE_LOST":
            raise LeaseLostError()
        leased_until = str((data or {}).get("leased_until") or "")
        return 0.0 if not leased_until else 0.0  # server TTL drives renewal

    def require_lease(self) -> str:
        if not self._lease_token:
            raise AgentAPIError(409, "no active lease; claim a run first")
        return self._lease_token

    # --- research flow ---------------------------------------------------
    async def get_context(self, run_id: int) -> AgentContext:
        data = await self._request(
            "GET", f"/api/ai-lab/agent/runs/{run_id}/context"
        )
        return AgentContext.model_validate(data or {})

    async def submit_proposal(self, run_id: int, proposal: dict[str, Any]) -> dict:
        self.require_lease()
        return await self._request(
            "POST",
            f"/api/ai-lab/agent/runs/{run_id}/proposal",
            json_body={"lease_token": self._lease_token, "proposal": proposal},
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
                expected=(200, 409),
            )
            try:
                data = await self._request(
                    "GET", f"/api/ai-lab/agent/runs/{run_id}/result"
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

    async def submit_decision(self, run_id: int, decision: dict[str, Any]) -> dict:
        self.require_lease()
        return await self._request(
            "POST",
            f"/api/ai-lab/agent/runs/{run_id}/decision",
            json_body={"lease_token": self._lease_token, "decision": decision},
        )

    async def complete(
        self,
        run_id: int,
        action: str,
        reason: str = "",
    ) -> dict:
        body: dict[str, Any] = {"action": action, "reason": reason}
        if self._lease_token:
            body["lease_token"] = self._lease_token
        data = await self._request(
            "POST", f"/api/ai-lab/agent/runs/{run_id}/complete", json_body=body
        )
        self._lease_token = None
        return data or {}
