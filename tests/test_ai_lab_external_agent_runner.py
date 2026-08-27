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
        self.proposal_request_id = None
        self.decision_request_id = None

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
            active_models=[],
            recent_trade_statistics={},
            prior_experiments=[],
            available_feature_sets=["FS_D0"],
            quality_gate={"min_trades": 30},
        )

    async def submit_proposal(self, run_id: int, proposal, *, client_request_id=None, telemetry=None):
        self.calls.append("proposal")
        self.submitted_proposal = proposal
        self.proposal_request_id = client_request_id
        return {"config_id": 91}

    async def wait_for_experiment_result(
        self, *, run_id: int, timeout_seconds: int, context=None
    ):
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

    async def submit_decision(self, run_id: int, decision, *, client_request_id=None, telemetry=None):
        self.calls.append("decision")
        self.submitted_decision = decision
        self.decision_request_id = client_request_id
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
                    "metric": "median_oot_pnl",
                    "direction": "increase",
                    "target_gain": None,
                },
                "reasoning": [],
                "risks": [],
                "test_plan": {
                    "oot_windows": 3,
                    "min_markets": 50,
                    "execution_mode": "PAPER_REALISTIC",
                },
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


class FailingLLM(FakeLLM):
    async def propose_hypothesis(self, context: dict):
        import httpx

        raise httpx.TimeoutException("OpenCode request timed out")


def test_runner_requeues_unexpected_llm_error_and_drops_lease():
    client = FakeClient()

    progressed = _run(agent_runner.process_one_run(client, FailingLLM()))

    assert progressed is False
    assert client.calls[-1] == "complete:REQUEUE"
    assert client._lease_token is None


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
    progressed = _run(
        agent_runner.process_one_run(client, FakeLLM(action="CONTINUE_RESEARCH"))
    )
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
            return httpx.Response(
                200,
                json={
                    "run": {
                        "id": 1,
                        "status": "RUNNING",
                        "objective": "x",
                        "scope": {},
                        "autonomy_level": "EXPERIMENT",
                        "iteration": 0,
                        "budget_remaining_steps": 1,
                    },
                    "active_models": [],
                    "recent_trade_statistics": {},
                    "prior_experiments": [],
                    "available_feature_sets": [],
                    "quality_gate": {},
                },
            )
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


def test_opencode_client_uses_explicit_protocol():
    import httpx
    from opencode_client import OpenCodeClient

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        # Return minimal structured output
        if "chat/completions" in captured["url"]:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"hypothesis": "h", "asset": "BTC", "market_role": "ALL", "model_family": "LOGREG", "feature_set": "FS_D0", "parameter_changes": [], "strategy_parameter_changes": [], "expected_effect": {"metric": "median_oot_pnl", "direction": "increase", "target_gain": null}, "reasoning": [], "risks": [], "test_plan": {"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"}}'
                            }
                        }
                    ]
                },
            )
        else:
            return httpx.Response(
                200,
                json={
                    "output_text": '{"hypothesis": "h", "asset": "BTC", "market_role": "ALL", "model_family": "LOGREG", "feature_set": "FS_D0", "parameter_changes": [], "strategy_parameter_changes": [], "expected_effect": {"metric": "median_oot_pnl", "direction": "increase", "target_gain": null}, "reasoning": [], "risks": [], "test_plan": {"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"}}'
                },
            )

    # Patch httpx.AsyncClient to use mock transport
    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    import opencode_client as oc_mod

    oc_mod.httpx.AsyncClient = PatchedAsyncClient
    try:
        client = OpenCodeClient()
        # Case 1: explicit chat_completions protocol should hit chat endpoint
        ctx_chat = {
            "research": {"model_id": "any-model", "protocol": "chat_completions"},
            "research_model": "any-model",
        }
        captured.clear()
        _run(client.propose_hypothesis(ctx_chat))
        assert "chat/completions" in captured["url"]
        # Case 2: explicit responses protocol should hit responses endpoint
        ctx_resp = {
            "research": {"model_id": "any-model", "protocol": "responses"},
            "research_model": "any-model",
        }
        captured.clear()
        _run(client.propose_hypothesis(ctx_resp))
        assert "responses" in captured["url"]
        # Case 3: context via runner's snapshot shape
        snap_ctx = {
            "research": {"model_id": "m1", "protocol": "chat_completions"},
            "summary": {"model_id": "m2", "protocol": "responses"},
            "research_model": "m1",
            "summary_model": "m2",
        }
        captured.clear()
        _run(client.propose_hypothesis(snap_ctx))
        assert "chat/completions" in captured["url"]
    finally:
        oc_mod.httpx.AsyncClient = original


def test_resumable_runner_phases_and_heartbeat():
    # Simulate a resumable run that goes through all phases
    class PhaseClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.phase_calls = 0
            self.heartbeats = 0
            self.phases = [
                "NEEDS_PROPOSAL",
                "WAITING_RESULT",
                "NEEDS_DECISION",
                "NEEDS_COMPLETION",
            ]
            self._lease_token = "lease-token"

        async def get_phase(self, run_id: int):
            # Return next phase each time
            idx = min(self.phase_calls, len(self.phases) - 1)
            self.phase_calls += 1
            # Track calls
            self.calls.append(f"phase:{self.phases[idx]}")
            return {
                "phase": self.phases[idx],
                "latest_config_id": 91,
                "latest_result_id": 1,
            }

        async def get_result(self, run_id: int):
            self.calls.append("get_result")
            return {
                "state": "READY",
                "result": {
                    "result_id": 1,
                    "config_id": 91,
                    "evaluation_kind": "POLYMARKET_OOT",
                    "status": "SUCCEEDED",
                },
            }

        async def heartbeat(self, run_id: int):
            self.heartbeats += 1
            self.calls.append("heartbeat")
            return 0.0

        async def get_context(self, run_id: int):
            self.calls.append("context")
            return SimpleNamespace(
                active_models=[],
                recent_trade_statistics={},
                prior_experiments=[],
                available_feature_sets=["FS_D0"],
                quality_gate={"min_trades": 30},
            )

        async def wait_for_experiment_result(
            self, *, run_id: int, timeout_seconds: int, context=None
        ):
            self.calls.append("wait")
            await asyncio.sleep(0.12)
            return ExperimentResult(
                config_id=91,
                evaluation_kind="POLYMARKET_OOT",
                status="SUCCEEDED",
                metrics={"median_pnl": 1.2},
                net_pnl=1.2,
                trade_count=100,
                max_drawdown=-0.5,
                summary="ok",
            )

    # Patch POLL_SECONDS to make heartbeat fast
    import runner as rmod

    orig_poll = rmod.POLL_SECONDS
    rmod.POLL_SECONDS = 0.05
    try:
        client = PhaseClient()

        # Need to also have claim return a run with budget 2 so it can go through phases
        async def fast_claim():
            return SimpleNamespace(
                id=99,
                status="RUNNING",
                objective="test",
                scope={"asset": "BTC"},
                autonomy_level="EXPERIMENT",
                budget_experiments=2,
                experiments_completed=0,
                budget_seconds=10,
                lease_token="lease-token",
                llm_provider="opencode",
                llm_research_model="m",
                llm_summary_model="m",
                llm_snapshot={
                    "provider": "opencode",
                    "research": {"model_id": "m", "protocol": "responses"},
                    "summary": {"model_id": "m", "protocol": "responses"},
                },
            )

        client.claim = fast_claim
        progressed = _run(agent_runner.process_one_run(client, FakeLLM()))
        assert progressed is True
        # Should have gone through all phases and completed
        assert any("phase:NEEDS_PROPOSAL" in c for c in client.calls)
        assert any("phase:WAITING_RESULT" in c for c in client.calls)
        assert any("phase:NEEDS_DECISION" in c for c in client.calls)
        assert client.proposal_request_id == "proposal-99-0"
        assert client.decision_request_id == "decision-99-1"
        assert any("phase:NEEDS_COMPLETION" in c for c in client.calls)
        assert client.completed_with in ("COMPLETED", "REQUEUE", "FAILED")
        # Heartbeat over whole run should have been called at least once
        assert client.heartbeats >= 1
    finally:
        rmod.POLL_SECONDS = orig_poll


def test_runner_handles_lease_loss_and_transient_errors():
    # Lease loss should drop lease and return False
    class LeaseLostClient(FakeClient):
        async def claim(self):
            self.calls.append("claim")
            return SimpleNamespace(
                id=1,
                status="RUNNING",
                objective="x",
                scope={},
                autonomy_level="EXPERIMENT",
                budget_experiments=1,
                experiments_completed=0,
                budget_seconds=10,
                lease_token="lease-token",
                llm_provider="opencode",
                llm_research_model="m",
                llm_summary_model="m",
            )

        async def get_phase(self, run_id: int):
            from api_client import LeaseLostError

            raise LeaseLostError()

    client = LeaseLostClient()
    progressed = _run(agent_runner.process_one_run(client, FakeLLM()))
    assert progressed is False
    assert client._lease_token is None

    # Transient 5xx should be retried and result in REQUEUE or COMPLETED, not crash
    class TransientClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0
            self._lease_token = "lease-token"

        async def claim(self):
            return SimpleNamespace(
                id=2,
                status="RUNNING",
                objective="x",
                scope={},
                autonomy_level="EXPERIMENT",
                budget_experiments=1,
                experiments_completed=0,
                budget_seconds=10,
                lease_token="lease-token",
                llm_provider="opencode",
                llm_research_model="m",
                llm_summary_model="m",
            )

        async def get_phase(self, run_id: int):
            self.attempts += 1
            if self.attempts == 1:
                from api_client import AgentAPIError

                raise AgentAPIError(500, "transient")
            return {"phase": "NEEDS_COMPLETION"}

        async def complete(self, run_id: int, action: str, reason: str = ""):
            self.calls.append(f"complete:{action}")
            self.completed_with = action

    client2 = TransientClient()
    progressed2 = _run(agent_runner.process_one_run(client2, FakeLLM()))
    assert progressed2 is True
    assert client2.completed_with is not None


def test_runner_budget_exhaustion():
    class BudgetClient(FakeClient):
        async def claim(self):
            self.calls.append("claim")
            return SimpleNamespace(
                id=3,
                status="RUNNING",
                objective="x",
                scope={},
                autonomy_level="EXPERIMENT",
                budget_experiments=1,
                experiments_completed=1,
                budget_seconds=10,
                lease_token="lease-token",
                llm_provider="opencode",
                llm_research_model="m",
                llm_summary_model="m",
            )

        async def complete(self, run_id: int, action: str, reason: str = ""):
            self.calls.append(f"complete:{action}")
            self.completed_with = action

    client = BudgetClient()
    progressed = _run(agent_runner.process_one_run(client, FakeLLM()))
    assert progressed is True

    assert client.completed_with == "COMPLETED"
    assert client.calls[-1] == "complete:COMPLETED"


def test_decision_schema_describes_overlay_items_as_objects():
    from opencode_client import _decision_schema

    overlay = _decision_schema()["properties"]["proposed_overlay"]
    assert overlay["type"] == ["array", "null"]
    assert overlay["items"]["type"] == "object"
    assert overlay["items"]["required"] == ["key", "value"]

def test_opencode_decision_uses_snapshot_summary_model():
    from opencode_client import OpenCodeClient

    client = OpenCodeClient()
    captured = {}

    async def fake_structured_json(**kwargs):
        captured.update(kwargs)
        return (
            {"action": "FINALIZE_NO_WINNER", "rationale": "ok", "key_findings": [],
             "recommended_config_id": None, "proposed_overlay": None, "next_step_focus": None},
            {"model": kwargs["model"], "latency_ms": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0},
        )

    client._structured_json = fake_structured_json
    result = _run(client.decide(
        context={"research": {"model_id": "m1", "protocol": "chat_completions"},
                 "summary": {"model_id": "m2", "protocol": "responses"}},
        proposal={},
        result=None,
    ))
    assert captured["model"] == "m2"
    assert captured["protocol"] == "responses"
    assert result["telemetry"]["model"] == "m2"
