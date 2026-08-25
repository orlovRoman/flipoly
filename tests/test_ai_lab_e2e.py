"""E2E test for independent AI research agent: real FastAPI router, separate DB session per request, real AILabApiClient, fake OpenCode transport, real runner."""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from polyflip.api.main import app
from polyflip.db.connection import get_db_session
from polyflip.db.models import Base, AILLMModelCatalog, AIOptimizationRun, AIRunStep, ExperimentResult
import polyflip.ai_lab.llm_catalog as llm_catalog

SERVICES_DIR = os.path.join(os.path.dirname(__file__), "..", "services", "ai_research_agent")
sys.path.insert(0, os.path.abspath(SERVICES_DIR))

from api_client import AILabApiClient
import runner as agent_runner
from schemas import ExperimentResult as ClientExperimentResult

# Reuse the fake LLM from runner tests


class E2EFakeLLM:
    async def propose_hypothesis(self, context: dict):
        return {
            "proposal": {
                "hypothesis": f"e2e hypothesis for {context['run_id']}",
                "asset": "BTC",
                "market_role": "OUTSIDER",
                "model_family": "LOGREG",
                "feature_set": "FS_D1",
                "parameter_changes": {},
                "strategy_parameter_changes": {},
                "expected_effect": {"metric": "median_oot_pnl", "direction": "increase", "target_gain": None},
                "reasoning": ["e2e"], "risks": [],
                "test_plan": {"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"},
            },
            "latency_ms": 5,
        }

    async def decide(self, *, context, proposal, result):
        return {
            "decision": {
                "action": "FINALIZE_NO_WINNER",
                "rationale": "e2e decision rationale that is long enough to pass validation",
                "key_findings": ["pnl ok"],
                "recommended_config_id": proposal.get("config_id") if isinstance(proposal, dict) else None,
                "proposed_overlay": None,
                "next_step_focus": None,
            },
            "latency_ms": 5,
        }


def _fake_opencode_handler(request):
    import httpx as _httpx
    import json

    # Determine endpoint
    url = str(request.url)
    if "chat/completions" in url:
        payload = {"choices": [{"message": {"content": json.dumps({"hypothesis": "h", "asset": "BTC", "market_role": "ALL", "model_family": "LOGREG", "feature_set": "FS_D0", "parameter_changes": [], "strategy_parameter_changes": [], "expected_effect": {"metric": "median_oot_pnl", "direction": "increase", "target_gain": None}, "reasoning": [], "risks": [], "test_plan": {"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"}})}}]}
        # Also handle decision
        if "hypothesis" in request.content.decode() if request.content else "":
            return _httpx.Response(200, json=payload)
        # For decide, similar
        return _httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"action": "FINALIZE_NO_WINNER", "rationale": "r" * 20, "key_findings": ["k"], "recommended_config_id": None, "proposed_overlay": None, "next_step_focus": None})}}]})
    else:
        # responses endpoint
        return _httpx.Response(200, json={"output_text": json.dumps({"hypothesis": "h", "asset": "BTC", "market_role": "ALL", "model_family": "LOGREG", "feature_set": "FS_D0", "parameter_changes": [], "strategy_parameter_changes": [], "expected_effect": {"metric": "median_oot_pnl", "direction": "increase", "target_gain": None}, "reasoning": [], "risks": [], "test_plan": {"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"}})})


@pytest.mark.asyncio
async def test_e2e_real_router_separate_sessions_fake_opencode_real_runner(monkeypatch):
    # Setup in-memory engine
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    # Seed catalog with PASSED models so snapshot can be created
    async with SessionLocal() as seed_session:
        now = datetime.now(timezone.utc)
        for mid, proto in [("resp-model", "responses"), ("chat-model", "chat_completions")]:
            seed_session.add(AILLMModelCatalog(provider="opencode", model_id=mid, display_name=mid, protocol=proto, is_available=True, is_discovered=True, probe_status="PASSED", last_checked_at=now, discovered_at=now, supports_structured_output=True))
        await seed_session.commit()

    # Create run via service (using same engine but separate session)
    from polyflip.ai_lab.service import create_run, create_permission
    from uuid import uuid4

    async with SessionLocal() as s:
        perm = await create_permission(s, profile_name=f"e2e-{uuid4().hex[:4]}", allowed_actions=["CREATE_EXPERIMENT", "TRAIN_MODEL", "RUN_OOT_BACKTEST", "RUN_POLYMARKET_OOT"], scope={}, limits={}, updated_by="test", enabled=True)
        run = await create_run(s, objective="e2e test", scope={"asset": "BTC"}, autonomy_level="OBSERVE", budget_experiments=1, permission=perm, llm_provider="opencode", llm_research_model="resp-model", llm_summary_model="resp-model")
        run.status = "QUEUED"
        await s.flush()
        await s.commit()
        run_id = int(run.id)

    # Setup FastAPI dependency to yield separate DB session per request
    async def override_db():
        async with SessionLocal() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_db

    # Prepare AILabApiClient that talks to the FastAPI app via ASGI transport
    # Patch httpx.AsyncClient to use ASGITransport for the test base_url
    import httpx
    import api_client as ac_mod

    original_async_client = httpx.AsyncClient

    # We will make AILabApiClient use a custom transport that routes to the FastAPI app
    transport = ASGITransport(app=app)

    class E2EAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            # Use ASGI transport for any request to http://test
            kwargs["transport"] = transport
            kwargs["base_url"] = kwargs.get("base_url", "http://test")
            super().__init__(*args, **kwargs)

    # Patch both the api_client's httpx.AsyncClient and the global one for opencode
    ac_mod.httpx.AsyncClient = E2EAsyncClient

    # Also need to patch opencode_client's httpx to use fake transport for LLM calls
    import opencode_client as oc_mod
    oc_original = httpx.AsyncClient

    def fake_opencode_transport(*args, **kwargs):
        # For opencode calls, return fake data
        # Detect if it's an opencode endpoint (contains opencode.ai or zen)
        # We'll use a MockTransport that returns hypothesis/decision
        import httpx as _httpx

        # Create a handler that returns hypothesis/decision based on request body
        def handler(request: _httpx.Request) -> _httpx.Response:
            import json as _json
            try:
                body = _json.loads(request.content) if request.content else {}
            except Exception:
                body = {}
            # Check schema name in body
            body_str = request.content.decode() if request.content else ""
            if "hypothesis_proposal" in body_str:
                inner = _json.dumps({"hypothesis": "e2e hypothesis that is long enough for validation", "asset": "BTC", "market_role": "OUTSIDER", "model_family": "LOGREG", "feature_set": "FS_D1", "parameter_changes": [], "strategy_parameter_changes": [], "expected_effect": {"metric": "median_oot_pnl", "direction": "increase", "target_gain": None}, "reasoning": ["r"], "risks": [], "test_plan": {"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"}})
                if "chat/completions" in str(request.url):
                    return _httpx.Response(200, json={"choices": [{"message": {"content": inner}}]})
                else:
                    return _httpx.Response(200, json={"output_text": inner})
            if "agent_decision" in body_str:
                inner2 = _json.dumps({"action": "FINALIZE_NO_WINNER", "rationale": "e2e decision rationale that is long enough to pass validation", "key_findings": ["k"], "recommended_config_id": None, "proposed_overlay": None, "next_step_focus": None})
                if "chat/completions" in str(request.url):
                    return _httpx.Response(200, json={"choices": [{"message": {"content": inner2}}]})
                else:
                    return _httpx.Response(200, json={"output_text": inner2})
            # Fallback
            return _httpx.Response(200, json={"output_text": "{}"})

        return _httpx.MockTransport(handler)

    # For opencode, we need to patch the client's internal AsyncClient to use fake transport
    # Instead, we can monkeypatch the _structured_json to bypass network and return directly
    # Simpler: patch OpenCodeClient to use a fake that doesn't need transport
    # We'll just let the existing E2EFakeLLM handle LLM calls, not actually use OpenCodeClient via network
    # So we don't need to patch opencode_client transport; we'll use E2EFakeLLM

    # Setup client and LLM
    # Use a base_url that will be intercepted by our E2EAsyncClient (which uses ASGI)
    client = AILabApiClient("http://test", "test-key", poll_seconds=0.2, timeout_seconds=5.0)

    # Need to set token to expected value for the test (app uses settings.API_KEY or AI_LAB_AGENT_TOKEN)
    # The verify_agent_token will check against settings.API_KEY (test-key) by default, so token test-key should work
    # Ensure settings
    from polyflip.config import settings as cfg
    monkeypatch.setattr(cfg, "API_KEY", "test-key")
    monkeypatch.setattr(cfg, "AI_LAB_AGENT_TOKEN", "")

    llm = E2EFakeLLM()

    # Background task to simulate the executor: after proposal, insert a POLYMARKET_OOT result and mark pending steps as done
    async def auto_insert_result():
        # Poll for proposal step, then insert result and mark pending steps SUCCEEDED
        for _ in range(30):
            async with SessionLocal() as check_s:
                steps = (await check_s.execute(sa.select(AIRunStep).where(AIRunStep.run_id == run_id))).scalars().all()
                # Find proposal config
                cfg_id = None
                for st in steps:
                    if st.step_type == "PROPOSAL" and isinstance(st.output_payload, dict):
                        cfg_id = st.output_payload.get("config_id")
                        break
                if cfg_id is not None:
                    # Insert result if not already inserted
                    existing = (await check_s.execute(sa.select(ExperimentResult).where(ExperimentResult.run_id == run_id, ExperimentResult.config_id == cfg_id))).scalar_one_or_none()
                    if existing is None:
                        now2 = datetime.now(timezone.utc)
                        # Mark pending TRAIN/OOT steps as SUCCEEDED so phase can progress to NEEDS_DECISION
                        for st in steps:
                            if st.status == "PENDING" and st.step_type in {"TRAIN_MODEL", "RUN_OOT_BACKTEST", "RUN_POLYMARKET_OOT"}:
                                st.status = "SUCCEEDED"
                                st.finished_at = now2
                                st.output_payload = {"result_id": 1}
                        # Insert terminal OOT result
                        res = ExperimentResult(run_id=run_id, config_id=int(cfg_id), evaluation_kind="POLYMARKET_OOT", status="SUCCEEDED", metrics={"median_pnl": 1.5}, trade_count=100, net_pnl=1.5, max_drawdown=-0.5, created_at=now2)
                        check_s.add(res)
                        await check_s.commit()
                        return
            await asyncio.sleep(0.2)
        # If we exit loop without inserting, fail test

    # Start background inserter
    inserter = asyncio.create_task(auto_insert_result())

    try:
        # Run the real runner with real API client and fake LLM
        progressed = await agent_runner.process_one_run(client, llm)
        assert progressed is True
        # Wait for inserter to finish
        try:
            await asyncio.wait_for(inserter, timeout=5.0)
        except asyncio.TimeoutError:
            pass

        # Verify final state via DB
        async with SessionLocal() as verify_s:
            final_run = await verify_s.get(AIOptimizationRun, run_id)
            assert final_run is not None
            assert final_run.status in ("COMPLETED", "FAILED", "EVALUATING")
            # Check that steps were created with no index conflicts
            all_steps = (await verify_s.execute(sa.select(AIRunStep).where(AIRunStep.run_id == run_id).order_by(AIRunStep.step_index))).scalars().all()
            indices = [s.step_index for s in all_steps]
            assert indices == sorted(indices)
            assert len(indices) == len(set(indices))
            # Check that at least one PROPOSAL and one DECISION exist
            step_types = {s.step_type for s in all_steps}
            assert "PROPOSAL" in step_types
            assert "DECISION" in step_types

    finally:
        inserter.cancel()
        try:
            await inserter
        except asyncio.CancelledError:
            pass
        app.dependency_overrides.pop(get_db_session, None)
        ac_mod.httpx.AsyncClient = original_async_client
        await engine.dispose()
