from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

from polyflip.ai_lab.service import (
    AILabError,
    approve_and_activate_deployment,
    create_deployment_revision,
    generate_deployment_diff,
    propose_live_deployment,
    record_deployment_event,
    reject_deployment_approval,
    rollback_deployment,
    transition_run,
)
from polyflip.db.models import (
    AIApprovalRequest,
    AIExperimentConfig,
    AIModelArtifact,
    AIOptimizationRun,
    DeploymentEvent,
    DeploymentRevision,
    ModelRegistry,
)


class _MockSession:
    def __init__(self, entities=None):
        self.entities = entities or {}
        self.added = []
        self.flush_count = 0

    async def get(self, model, object_id):
        name = getattr(model, "__name__", "")
        return self.entities.get((name, int(object_id)))

    def add(self, item):
        self.added.append(item)
        name = getattr(item.__class__, "__name__", "")
        if hasattr(item, "id") and item.id is not None:
            self.entities[(name, item.id)] = item

    async def execute(self, stmt):
        class _Result:
            def __init__(self, items):
                self._items = items

            def scalars(self):
                return self

            def all(self):
                return self._items

            def scalar_one_or_none(self):
                return self._items[0] if self._items else None

        stmt_str = str(stmt)
        if "FROM deployment_events" in stmt_str:
            events = [e for e in self.added if isinstance(e, DeploymentEvent)]
            return _Result(events[-1:] if events else [])
        if "FROM deployment_revisions" in stmt_str:
            revs = [
                v
                for (k, v) in self.entities.items()
                if k[0] == "DeploymentRevision"
            ] + [r for r in self.added if isinstance(r, DeploymentRevision)]
            return _Result(revs)
        if "FROM model_registry" in stmt_str:
            models = [
                v for (k, v) in self.entities.items() if k[0] == "ModelRegistry"
            ]
            return _Result(models)
        if "FROM ai_approval_requests" in stmt_str:
            approvals = [
                v
                for (k, v) in self.entities.items()
                if k[0] == "AIApprovalRequest"
            ]
            return _Result(approvals)
        if "FROM ai_optimization_runs" in stmt_str:
            runs = [
                v
                for (k, v) in self.entities.items()
                if k[0] == "AIOptimizationRun"
            ]
            return _Result(runs)
        return _Result([])

    async def flush(self):
        self.flush_count += 1
        for item in self.added:
            if not getattr(item, "id", None):
                item.id = len(self.entities) + 1
                name = getattr(item.__class__, "__name__", "")
                self.entities[(name, item.id)] = item


@pytest.mark.asyncio
async def test_hash_chain_no_bifurcation():
    """P0-1: Test per-revision event hash chain."""
    session = _MockSession()
    ev1 = await record_deployment_event(
        session,
        revision_id=1,
        event_type="CREATED",
        actor="operator",
        reason="initial revision",
    )
    assert ev1.previous_hash == "0" * 64
    assert len(ev1.event_hash) == 64

    ev2 = await record_deployment_event(
        session,
        revision_id=1,
        event_type="APPROVED",
        actor="admin",
        reason="approved",
    )
    assert ev2.previous_hash == ev1.event_hash
    assert len(ev2.event_hash) == 64
    assert ev2.event_hash != ev1.event_hash


@pytest.mark.asyncio
async def test_propose_live_deployment_generates_diff_and_revision():
    run = SimpleNamespace(
        id=7,
        status="SHADOW",
        summary='{"report":{"recommended_config_id":11,"rows":[{"config_id":11,"artifact_ids":[101],"median_oot_pnl":2.5,"total_trades":60,"median_oot_drawdown":-1.0,"window_count":3}]}}',
        scope={"max_drawdown": -3.0, "min_trades": 50},
    )
    config = SimpleNamespace(
        id=11,
        name="candidate_v2",
        asset="BTCUSDT",
        regime="mid_vol",
        model_family="lightgbm",
        feature_set="FS_D1",
        feature_pipeline_version="v2",
        strategy_params={"decision_threshold": 0.58, "decision_threshold_down": 0.42},
        model_params={"num_leaves": 31},
        backtest_params={},
    )
    active_baseline = SimpleNamespace(
        id=1,
        asset="BTCUSDT",
        version=1,
        model_type="logreg",
        features="FS_D0",
        decision_threshold=0.55,
        decision_threshold_down=0.45,
        accuracy=0.60,
        backtest_pnl=1.0,
        backtest_trades=50,
        is_active=True,
    )

    session = _MockSession(
        {
            ("AIOptimizationRun", 7): run,
            ("AIExperimentConfig", 11): config,
            ("ModelRegistry", 1): active_baseline,
        }
    )

    approval, revision = await propose_live_deployment(
        session,
        run_id=7,
        actor="operator",
        reason="Promoting OOT winner to live review",
    )

    assert approval.status == "PENDING"
    assert approval.target_type == "DEPLOYMENT_REVISION"
    assert approval.requested_action == "ACTIVATE"
    assert approval.diff["asset"] == "BTCUSDT"
    assert approval.diff["candidate"]["decision_threshold"] == 0.58
    assert approval.diff["baseline"]["decision_threshold"] == 0.55
    assert approval.diff["metrics"]["median_pnl"] == 2.5

    assert revision.status == "PENDING_APPROVAL"
    assert revision.manifest["strategy"]["decision_threshold"] == 0.58
    assert revision.manifest_hash is not None
    assert revision.revision_key.startswith("rev_7_11_")
    assert run.status == "PENDING_APPROVAL"


@pytest.mark.asyncio
async def test_propose_live_deployment_rejects_non_shadow_run():
    run = SimpleNamespace(id=8, status="RUNNING", summary=None)
    session = _MockSession({("AIOptimizationRun", 8): run})
    with pytest.raises(AILabError, match="requires run in SHADOW or PENDING_APPROVAL"):
        await propose_live_deployment(session, run_id=8)


@pytest.mark.asyncio
async def test_approve_and_activate_switches_model_registry_pointers():
    """P0-2, P0-3: Test activation with strict artifact check & transition_run."""
    run = SimpleNamespace(id=7, status="PENDING_APPROVAL", summary="Initial")
    old_active = SimpleNamespace(id=1, asset="BTCUSDT", is_active=True)
    new_candidate = SimpleNamespace(id=2, asset="BTCUSDT", is_active=False)
    artifact = SimpleNamespace(id=101, model_registry_id=2)

    revision = SimpleNamespace(
        id=10,
        status="PENDING_APPROVAL",
        manifest={
            "models": [{"asset": "BTCUSDT", "artifact_id": 101}],
            "strategy": {"decision_threshold": 0.58},
            "risk_policy": {},
            "execution_policy": {},
        },
        activated_at=None,
    )
    approval = SimpleNamespace(
        id=5,
        run_id=7,
        target_id="10",
        status="PENDING",
        decided_at=None,
        decided_by=None,
    )

    session = _MockSession(
        {
            ("AIApprovalRequest", 5): approval,
            ("DeploymentRevision", 10): revision,
            ("AIModelArtifact", 101): artifact,
            ("ModelRegistry", 1): old_active,
            ("ModelRegistry", 2): new_candidate,
            ("AIOptimizationRun", 7): run,
        }
    )

    activated_rev = await approve_and_activate_deployment(
        session,
        approval_id=5,
        actor="admin",
        reason="Approved by risk committee",
    )

    assert activated_rev.status == "ACTIVE"
    assert activated_rev.activated_at is not None
    assert approval.status == "APPROVED"
    assert approval.decided_by == "admin"
    assert old_active.is_active is False
    assert new_candidate.is_active is True
    assert run.status == "ACTIVE"
    assert "Activated by admin" in run.summary


@pytest.mark.asyncio
async def test_activate_no_artifact_raises():
    """P0-3: Test that missing artifact raises AILabError and does not deactivate old model."""
    old_active = SimpleNamespace(id=1, asset="BTCUSDT", is_active=True)
    revision = SimpleNamespace(
        id=10,
        status="PENDING_APPROVAL",
        manifest={
            "models": [{"asset": "BTCUSDT", "artifact_id": 999}],
            "strategy": {},
            "risk_policy": {},
            "execution_policy": {},
        },
    )
    approval = SimpleNamespace(id=5, target_id="10", status="PENDING", run_id=None)
    session = _MockSession(
        {
            ("AIApprovalRequest", 5): approval,
            ("DeploymentRevision", 10): revision,
            ("ModelRegistry", 1): old_active,
        }
    )

    with pytest.raises(AILabError, match="has no linked ModelRegistry entry"):
        await approve_and_activate_deployment(session, approval_id=5, actor="admin")
    assert old_active.is_active is True


@pytest.mark.asyncio
async def test_reject_deployment_approval_marks_revision_and_run_rejected():
    """P0-2: Test reject uses transition_run and locks."""
    run = SimpleNamespace(id=7, status="PENDING_APPROVAL", summary="Initial")
    revision = SimpleNamespace(id=10, status="PENDING_APPROVAL")
    approval = SimpleNamespace(
        id=5,
        run_id=7,
        target_id="10",
        status="PENDING",
        decided_at=None,
        decided_by=None,
        decision_reason=None,
    )

    session = _MockSession(
        {
            ("AIApprovalRequest", 5): approval,
            ("DeploymentRevision", 10): revision,
            ("AIOptimizationRun", 7): run,
        }
    )

    rejected_appr = await reject_deployment_approval(
        session,
        approval_id=5,
        actor="risk_officer",
        reason="Drawdown too high",
    )

    assert rejected_appr.status == "REJECTED"
    assert rejected_appr.decided_by == "risk_officer"
    assert revision.status == "REJECTED"
    assert run.status == "REJECTED"
    assert "Rejected by risk_officer" in run.summary


@pytest.mark.asyncio
async def test_rollback_deployment_restores_parent_revision():
    model_v1 = SimpleNamespace(id=1, asset="BTCUSDT", is_active=False)
    model_v2 = SimpleNamespace(id=2, asset="BTCUSDT", is_active=True)
    art1 = SimpleNamespace(id=101, model_registry_id=1)

    parent_rev = SimpleNamespace(
        id=1,
        status="SUPERSEDED",
        manifest={"models": [{"asset": "BTCUSDT", "artifact_id": 101}]},
        activated_at=None,
    )
    current_rev = SimpleNamespace(
        id=2,
        parent_id=1,
        status="ACTIVE",
        manifest={"models": [{"asset": "BTCUSDT", "artifact_id": 102}]},
        rolled_back_at=None,
    )

    session = _MockSession(
        {
            ("DeploymentRevision", 1): parent_rev,
            ("DeploymentRevision", 2): current_rev,
            ("AIModelArtifact", 101): art1,
            ("ModelRegistry", 1): model_v1,
            ("ModelRegistry", 2): model_v2,
        }
    )

    restored_rev = await rollback_deployment(
        session,
        target_revision_id=1,
        actor="admin",
        reason="Emergency rollback due to market anomaly",
    )

    assert restored_rev.id == 1
    assert restored_rev.status == "ACTIVE"
    assert current_rev.status == "ROLLED_BACK"
    assert model_v1.is_active is True
    assert model_v2.is_active is False


@pytest.mark.asyncio
async def test_rollback_no_artifact_raises():
    """P0-3: Test rollback with missing artifact raises AILabError."""
    current_rev = SimpleNamespace(
        id=2,
        parent_id=1,
        status="ACTIVE",
        manifest={"models": [{"asset": "BTCUSDT", "artifact_id": 102}]},
    )
    parent_rev = SimpleNamespace(
        id=1,
        status="SUPERSEDED",
        manifest={"models": [{"asset": "BTCUSDT", "artifact_id": 999}]},
    )
    active_model = SimpleNamespace(id=2, asset="BTCUSDT", is_active=True)

    session = _MockSession(
        {
            ("DeploymentRevision", 1): parent_rev,
            ("DeploymentRevision", 2): current_rev,
            ("ModelRegistry", 2): active_model,
        }
    )

    with pytest.raises(AILabError, match="has no linked ModelRegistry entry"):
        await rollback_deployment(session, target_revision_id=1)
    assert active_model.is_active is True


@pytest.mark.asyncio
async def test_idempotency_excludes_rolled_back():
    """P0-4: Test that create_deployment_revision does not reuse ROLLED_BACK revisions."""
    manifest = {
        "models": [{"asset": "BTCUSDT", "artifact_id": 101}],
        "strategy": {"decision_threshold": 0.58},
        "risk_policy": {},
        "execution_policy": {},
    }
    dead_rev = SimpleNamespace(
        id=1,
        revision_key="rev_old",
        manifest=manifest,
        status="ROLLED_BACK",
        manifest_hash="abc123hash",
    )
    session = _MockSession({("DeploymentRevision", 1): dead_rev})

    new_rev = await create_deployment_revision(
        session,
        revision_key="rev_new",
        manifest=manifest,
        status="PENDING_APPROVAL",
    )
    assert new_rev is not dead_rev
    assert new_rev.status == "PENDING_APPROVAL"


@pytest.mark.asyncio
async def test_propose_idempotent():
    """P1-1: Test that propose_live_deployment returns existing pending approval."""
    run = SimpleNamespace(
        id=7,
        status="PENDING_APPROVAL",
        summary='{"report":{"recommended_config_id":11,"rows":[{"config_id":11,"artifact_ids":[101],"median_oot_pnl":2.5,"total_trades":60,"median_oot_drawdown":-1.0,"window_count":3}]}}',
        scope={},
    )
    config = SimpleNamespace(
        id=11,
        name="cand",
        asset="BTCUSDT",
        regime="mid_vol",
        model_family="lightgbm",
        feature_set="FS_D1",
        feature_pipeline_version="v2",
        strategy_params={},
        model_params={},
        backtest_params={},
    )
    existing_appr = SimpleNamespace(
        id=42,
        run_id=7,
        requested_action="ACTIVATE",
        status="PENDING",
    )
    session = _MockSession(
        {
            ("AIOptimizationRun", 7): run,
            ("AIExperimentConfig", 11): config,
            ("AIApprovalRequest", 42): existing_appr,
        }
    )

    approval, revision = await propose_live_deployment(session, run_id=7)
    assert approval.id == 42


@pytest.mark.asyncio
async def test_diff_broken_summary_logs_warning():
    """P1-2: Test that broken summary is logged without unhandled crash."""
    run = SimpleNamespace(id=7, summary="INVALID_JSON{", scope={})
    shadow_assign = SimpleNamespace(
        id=1,
        run_id=7,
        candidate_artifact_id=101,
        asset="BTCUSDT",
        regime="mid_vol",
    )
    art = SimpleNamespace(
        id=101,
        artifact_metadata={"config_id": 11},
    )
    config = SimpleNamespace(
        id=11,
        name="cand",
        asset="BTCUSDT",
        regime="mid_vol",
        model_family="lightgbm",
        feature_set="FS_D1",
        feature_pipeline_version="v2",
        strategy_params={},
        model_params={},
        backtest_params={},
    )
    session = _MockSession(
        {
            ("AIOptimizationRun", 7): run,
            ("AIShadowAssignment", 1): shadow_assign,
            ("AIModelArtifact", 101): art,
            ("AIExperimentConfig", 11): config,
        }
    )
    with patch("polyflip.ai_lab.service.logger.warning") as mock_warn:
        diff = await generate_deployment_diff(session, run_id=7)
        mock_warn.assert_called_once()
        assert diff["asset"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_transition_preserves_reason():
    """P1-3: Test transition_run appends reason."""
    run = SimpleNamespace(id=7, status="SHADOW", summary="Summary from finalization")
    session = _MockSession()
    await transition_run(session, run, "PENDING_APPROVAL", reason="Proposed for live activation")
    assert "Summary from finalization" in run.summary
    assert "Proposed for live activation" in run.summary
