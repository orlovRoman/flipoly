"""Autonomous research-loop behavior of the external agent runner (T08)."""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

SERVICES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "services", "ai_research_agent"
)
sys.path.insert(0, os.path.abspath(SERVICES_DIR))

import runner as agent_runner  # noqa: E402
from schemas import ExperimentResult  # noqa: E402


class FakeClient:
    def __init__(self, *, result=None, wait_timeout=False):
        self.calls: list[str] = []
        self._result = result
        self._wait_timeout = wait_timeout
        self.completed_with = None
        self.submitted_proposal = None
        self.submitted_decision = None
        self._lease_token: str | None = "lease-token"

    def drop_lease(self) -> None:
        self._lease_token = None

    async def claim(self):
        self.calls.append("claim")
        return SimpleNamespace(
            id=41,
            status="QUEUED",
            objective="improve outsider pnl",
            scope={"asset": "BTC"},
            autonomy_level="EXPERIMENT",
            budget_experiments=1,
            experiments_completed=0,
            budget_seconds=600,
            lease_token="lease-token",
            llm_provider="opencode",
            llm_research_model="research-model",
            llm_summary_model="summary-model",
        )

    async def get_context(self, run_id: int):
        self.calls.append("context")
        return SimpleNamespace(
            active_models=[], recent_trade_statistics={},
            prior_experiments=[], available_feature_sets=["FS_D0"],
            quality_gate={"min_trades": 30},
        )

    async def submit_proposal(self, run_id: int, proposal):
        self.calls.append("proposal")
        self.submitted_proposal = proposal
        return {"config_id": 91}

    async def wait_for_experiment_result(self, *, run_id: int, timeout_seconds: int, context=None):
        self.calls.append("wait")
        if self._wait_timeout:
            return None
        return ExperimentResult(
            config_id=91,
            evaluation_kind="POLYMARKET_OOT",
            status="SUCCEEDED",
            metrics={"median_pnl": 1.2},
            net_pnl=3.4,
            trade_count=120,
            max_drawdown=-2.0,
            summary="positive",
        )

    async def submit_decision(self, run_id: int, decision):
        self.calls.append("decision")
        self.submitted_decision = decision
        return {"accepted": True}

    async def complete(self, run_id: int, action: str, reason: str = ""):
        self.calls.append(f"complete:{action}")
        self.completed_with = action


class FakeLLM:
    def __init__(self, action="FINALIZE_NO_WINNER"):
        self._action = action

    async def propose_hypothesis(self, context: dict):
        return {
            "proposal": {
                "hypothesis": f"test hypothesis for {context['run_id']}",
                "asset": "BTC",
                "market_role": "OUTSIDER",
                "model_family": "LogisticRegression",
                "feature_set": "FS_D0",
                "parameter_changes": {},
                "strategy_parameter_changes": {},
                "expected_effect": {
                    "metric": "median_oot_pnl", "direction": "increase",
                    "target_gain": None,
                },
                "reasoning": [], "risks": [],
                "test_plan": {"oot_windows": 3, "min_markets": 50,
                               "execution_mode": "PAPER_REALISTIC"},
            },
            "latency_ms": 12,
        }

    async def decide(self, *, context, proposal, result):
        return {
            "decision": {
                "action": self._action,
                "rationale": "deterministic test rationale",
                "key_findings": ["pnl positive"],
                "recommended_config_id": 91,
                "proposed_overlay": None,
                "next_step_focus": None,
            },
            "latency_ms": 9,
        }


def _run(coro):
    return asyncio.run(coro)


def test_full_cycle_happy_path_completes_run():
    client = FakeClient(result=SimpleNamespace())
    progressed = _run(agent_runner.process_one_run(client, FakeLLM()))
    assert progressed is True
    assert client.calls[0] == "claim"
    assert "proposal" in client.calls and "decision" in client.calls
    assert client.calls[-1] == "complete:COMPLETED"
    assert client.submitted_proposal["hypothesis"].startswith("test hypothesis")
    assert client.submitted_decision["action"] == "FINALIZE_NO_WINNER"


def test_wait_timeout_fails_run_without_llm_decision():
    client = FakeClient(wait_timeout=True)
    progressed = _run(agent_runner.process_one_run(client, FakeLLM()))
    assert progressed is True
    assert client.calls[-1] == "complete:FAILED"
    assert "decision" not in client.calls


def test_continue_research_within_budget_requeues_iteration():
    client = FakeClient()
    progressed = _run(agent_runner.process_one_run(
        client, FakeLLM(action="CONTINUE_RESEARCH")
    ))
    assert progressed is True
    # budget_experiments == 1 already consumed -> terminal COMPLETED, no REQUEUE.
    assert client.calls[-1] == "complete:COMPLETED"


def test_idle_queue_returns_false_without_side_effects():
    class EmptyClient(FakeClient):
        async def claim(self):
            self.calls.append("claim")
            return None

    client = EmptyClient()
    progressed = _run(agent_runner.process_one_run(client, FakeLLM()))
    assert progressed is False
    assert client.calls == ["claim"]


def test_result_serializes_to_json():
    result = ExperimentResult(
        config_id=91,
        evaluation_kind="POLYMARKET_OOT",
        status="SUCCEEDED",
        metrics={"median_pnl": 1.2},
        net_pnl=3.4,
        trade_count=120,
        max_drawdown=-2.0,
        summary="positive",
    )
    payload = result.model_dump(mode="json")
    assert payload["config_id"] == 91
    assert payload["evaluation_kind"] == "POLYMARKET_OOT"
    assert payload["metrics"]["median_pnl"] == 1.2


def test_api_client_params_and_lease_lost_clears_token():
    import httpx
    from api_client import AILabApiClient, LeaseLostError

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        # Simulate LEASE_LOST for heartbeat
        if request.url.path.endswith("/heartbeat"):
            return httpx.Response(409, json={"detail": "LEASE_LOST"})
        if request.url.path.endswith("/context"):
            # Verify lease_token in query params
            assert captured["params"].get("lease_token") == "test-lease"
            return httpx.Response(200, json={
                "run": {"id": 1, "status": "RUNNING", "objective": "x", "scope": {}, "autonomy_level": "EXPERIMENT", "iteration": 0, "budget_remaining_steps": 1},
                "active_models": [], "recent_trade_statistics": {}, "prior_experiments": [], "available_feature_sets": [], "quality_gate": {}
            })
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    client = AILabApiClient("http://test", "token")
    client._lease_token = "test-lease"
    # Patch httpx.AsyncClient to use mock transport
    original_async_client = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    import api_client as ac_module
    ac_module.httpx.AsyncClient = PatchedAsyncClient
    try:
        # GET context should include lease_token in params
        _run(client.get_context(1))
        assert captured["params"].get("lease_token") == "test-lease"
        # Heartbeat LEASE_LOST should clear token and raise
        try:
            _run(client.heartbeat(1))
            assert False, "should have raised LeaseLostError"
        except LeaseLostError:
            pass
        assert client._lease_token is None
        # drop_lease should work
        client._lease_token = "again"
        client.drop_lease()
        assert client._lease_token is None
    finally:
        ac_module.httpx.AsyncClient = original_async_client


def test_runner_uses_drop_lease_not_direct_assignment():
    # Ensure runner no longer does client._lease_token = None directly
    import pathlib
    import ast

    p = pathlib.Path(SERVICES_DIR) / "runner.py"
    src = p.read_text()
    assert "client._lease_token = None" not in src
    assert "drop_lease" in src
