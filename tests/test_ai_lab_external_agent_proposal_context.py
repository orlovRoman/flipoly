import asyncio
import os
import sys
from types import SimpleNamespace

SERVICES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "services", "ai_research_agent"
)
sys.path.insert(0, os.path.abspath(SERVICES_DIR))

import runner as agent_runner


class _PhaseClient:
    def __init__(self):
        self.phase_calls = 0
        self.proposal_seen = None
        self.completed = None
        self._lease_token = "lease"

    async def claim(self):
        return SimpleNamespace(
            id=41,
            status="RUNNING",
            objective="test",
            scope={"asset": "BTC"},
            autonomy_level="EXPERIMENT",
            budget_experiments=1,
            experiments_completed=0,
            budget_seconds=10,
            lease_token="lease",
            llm_provider="mock",
            llm_research_model="research",
            llm_summary_model="summary",
        )

    async def get_phase(self, _run_id):
        self.phase_calls += 1
        if self.phase_calls == 1:
            return {
                "phase": "NEEDS_DECISION",
                "latest_config_id": 9,
                "latest_result_id": 12,
                "latest_proposal": {
                    "hypothesis": "persisted hypothesis",
                    "asset": "BTC",
                    "model_family": "LOGREG",
                },
            }
        return {"phase": "NEEDS_COMPLETION"}

    async def get_context(self, _run_id):
        return SimpleNamespace(
            active_models=[],
            recent_trade_statistics={},
            prior_experiments=[],
            available_feature_sets=["FS_D0"],
            quality_gate={},
        )

    async def get_result(self, _run_id):
        return {
            "state": "READY",
            "result": {"result_id": 12, "config_id": 9, "status": "SUCCEEDED"},
        }

    async def submit_decision(
        self, _run_id, decision, *, client_request_id=None, telemetry=None
    ):
        return {"accepted": True, "decision": decision, "request_id": client_request_id}

    async def heartbeat(self, _run_id):
        return 0.0

    async def complete(self, _run_id, action, reason=""):
        self.completed = action

    def drop_lease(self):
        self._lease_token = None


class _RecordingLLM:
    def __init__(self):
        self.proposal = None

    async def decide(self, *, context, proposal, result):
        self.proposal = proposal
        return {
            "decision": {
                "action": "FINALIZE_NO_WINNER",
                "rationale": "done",
                "key_findings": [],
                "recommended_config_id": None,
                "proposed_overlay": None,
                "next_step_focus": None,
            }
        }


def test_runner_uses_persisted_proposal_for_decision():
    client = _PhaseClient()
    llm = _RecordingLLM()

    assert asyncio.run(agent_runner.process_one_run(client, llm)) is True
    assert llm.proposal["hypothesis"] == "persisted hypothesis"
    assert llm.proposal["model_family"] == "LOGREG"
    assert client.completed == "COMPLETED"
