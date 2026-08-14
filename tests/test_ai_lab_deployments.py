from types import SimpleNamespace
from unittest.mock import AsyncMock
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

        # Simple string-matching mock for queries
        stmt_str = str(stmt)
        if "FROM deployment_events" in stmt_str:
            events = [e for e in self.added if isinstance(e, DeploymentEvent)]
            return _Result(events[-1:] if events else [])
        if "FROM deployment_revisions" in stmt_str:
            revs = [r for r in self.added if isinstance(r, DeploymentRevision)]
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
        return _Result([])

    async def flush(self):
        self.flush_count += 1
        for item in self.added:
            if not getattr(item, "id", None):
                item.id = len(self.entities) + 1
                name = getattr(item.__class__, "__name__", "")
                self.entities[(name, item.id)] = item


@pytest.mark.asyncio
async def test_deployment_event_hash_chain_is_deterministic():
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
async def test_propose_live_deployment_generates_diff_and_revision(monkeypatch):
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
    assert run.status == "PENDING_APPROVAL"


@pytest.mark.asyncio
async def test_propose_live_deployment_rejects_non_shadow_run():
    run = SimpleNamespace(id=8, status="RUNNING", summary=None)
    session = _MockSession({("AIOptimizationRun", 8): run})
    with pytest.raises(AILabError, match="requires run in SHADOW or PENDING_APPROVAL"):
        await propose_live_deployment(session, run_id=8)


@pytest.mark.asyncio
async def test_approve_and_activate_switches_model_registry_pointers():
    run = SimpleNamespace(id=7, status="PENDING_APPROVAL")
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


@pytest.mark.asyncio
async def test_reject_deployment_approval_marks_revision_and_run_rejected():
    run = SimpleNamespace(id=7, status="PENDING_APPROVAL")
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
