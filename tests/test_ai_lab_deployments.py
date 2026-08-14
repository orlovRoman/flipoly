import asyncio
import contextlib
import os
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

try:
    import pytest
except ImportError:
    class _PytestMock:
        class mark:
            @staticmethod
            def asyncio(fn):
                return fn

        @staticmethod
        @contextlib.contextmanager
        def raises(expected_exc, match=None):
            try:
                yield
            except expected_exc as exc:
                if match and match not in str(exc):
                    raise AssertionError(
                        f"Pattern {match!r} not found in {str(exc)!r}"
                    ) from exc
            else:
                raise AssertionError(
                    f"Expected {expected_exc.__name__} was not raised"
                )

    pytest = _PytestMock()

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
    AIShadowAssignment,
    DeploymentEvent,
    DeploymentRevision,
    ModelRegistry,
)


class FakeSession:
    """Robust in-memory async session for unit testing without brittle SQL string parsing."""

    def __init__(self, entities: dict[tuple[type, int], Any] | None = None):
        self._store: dict[type, dict[int, Any]] = {}
        self.added: list[Any] = []
        self.flush_count = 0

        if entities:
            for (model_cls, obj_id), obj in entities.items():
                self._store.setdefault(model_cls, {})[int(obj_id)] = obj
                if hasattr(obj, "id") and obj.id is None:
                    obj.id = int(obj_id)

    async def get(self, model: type, object_id: int | str | None) -> Any | None:
        if object_id is None:
            return None
        return self._store.get(model, {}).get(int(object_id))

    def add(self, item: Any) -> None:
        self.added.append(item)
        model_cls = item.__class__
        if getattr(item, "id", None) is not None:
            self._store.setdefault(model_cls, {})[int(item.id)] = item

    async def flush(self) -> None:
        self.flush_count += 1
        for item in self.added:
            model_cls = item.__class__
            if getattr(item, "id", None) is None:
                existing_ids = self._store.get(model_cls, {}).keys()
                next_id = max(existing_ids, default=0) + 1
                item.id = next_id
                self._store.setdefault(model_cls, {})[next_id] = item

    async def execute(self, stmt: Any) -> Any:
        class _Result:
            def __init__(self, items: list[Any]):
                self._items = list(items)

            def scalars(self):
                return self

            def all(self) -> list[Any]:
                return list(self._items)

            def scalar_one_or_none(self) -> Any | None:
                return self._items[0] if self._items else None

        descriptions = getattr(stmt, "column_descriptions", None)
        if not descriptions:
            return _Result([])
        target_entity = descriptions[0].get("entity")
        if not target_entity:
            return _Result([])

        items_map = dict(self._store.get(target_entity, {}))
        for item in self.added:
            if isinstance(item, target_entity) and getattr(item, "id", None) is not None:
                items_map[int(item.id)] = item
        items = list(items_map.values())

        params: dict[str, Any] = {}
        try:
            params = stmt.compile().params
        except Exception:
            pass

        filtered = []
        for item in items:
            matches = True
            for param_key, param_val in params.items():
                clean_key = param_key.rsplit("_", 1)[0]
                if hasattr(item, clean_key):
                    attr_val = getattr(item, clean_key)
                    if isinstance(param_val, (set, list, tuple)):
                        if attr_val not in param_val:
                            matches = False
                            break
                    elif attr_val != param_val:
                        matches = False
                        break
            if matches:
                filtered.append(item)

        order_clauses = getattr(stmt, "_order_by_clauses", ())
        if any("DESC" in str(c).upper() for c in order_clauses):
            filtered.sort(key=lambda x: getattr(x, "id", 0), reverse=True)
        elif order_clauses:
            filtered.sort(key=lambda x: getattr(x, "id", 0))

        limit_val = getattr(stmt, "_limit", None)
        if limit_val is not None and isinstance(limit_val, int):
            filtered = filtered[:limit_val]

        return _Result(filtered)


@pytest.mark.asyncio
async def test_hash_chain_no_bifurcation():
    """P0-1: Test per-revision event hash chain."""
    session = FakeSession({(DeploymentRevision, 1): SimpleNamespace(id=1)})
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

    session = FakeSession(
        {
            (AIOptimizationRun, 7): run,
            (AIExperimentConfig, 11): config,
            (ModelRegistry, 1): active_baseline,
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
    session = FakeSession({(AIOptimizationRun, 8): run})
    with pytest.raises(AILabError, match="requires run in SHADOW or PENDING_APPROVAL"):
        await propose_live_deployment(session, run_id=8)


@pytest.mark.asyncio
async def test_approve_and_activate_switches_model_registry_pointers():
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
        target_type="DEPLOYMENT_REVISION",
        requested_action="ACTIVATE",
        target_id="10",
        status="PENDING",
        decided_at=None,
        decided_by=None,
    )

    session = FakeSession(
        {
            (AIApprovalRequest, 5): approval,
            (DeploymentRevision, 10): revision,
            (AIModelArtifact, 101): artifact,
            (ModelRegistry, 1): old_active,
            (ModelRegistry, 2): new_candidate,
            (AIOptimizationRun, 7): run,
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
    assert "Activated by admin: Approved by risk committee" in run.summary


@pytest.mark.asyncio
async def test_activate_no_artifact_raises():
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
    approval = SimpleNamespace(
        id=5,
        target_type="DEPLOYMENT_REVISION",
        requested_action="ACTIVATE",
        target_id="10",
        status="PENDING",
        run_id=None,
    )
    session = FakeSession(
        {
            (AIApprovalRequest, 5): approval,
            (DeploymentRevision, 10): revision,
            (ModelRegistry, 1): old_active,
        }
    )

    with pytest.raises(AILabError, match="has no linked ModelRegistry entry"):
        await approve_and_activate_deployment(session, approval_id=5, actor="admin")
    assert old_active.is_active is True


@pytest.mark.asyncio
async def test_reject_deployment_approval_marks_revision_and_run_rejected():
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

    session = FakeSession(
        {
            (AIApprovalRequest, 5): approval,
            (DeploymentRevision, 10): revision,
            (AIOptimizationRun, 7): run,
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
    assert "Rejected by risk_officer: Drawdown too high" in run.summary


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

    session = FakeSession(
        {
            (DeploymentRevision, 1): parent_rev,
            (DeploymentRevision, 2): current_rev,
            (AIModelArtifact, 101): art1,
            (ModelRegistry, 1): model_v1,
            (ModelRegistry, 2): model_v2,
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

    session = FakeSession(
        {
            (DeploymentRevision, 1): parent_rev,
            (DeploymentRevision, 2): current_rev,
            (ModelRegistry, 2): active_model,
        }
    )

    with pytest.raises(AILabError, match="has no linked ModelRegistry entry"):
        await rollback_deployment(session, target_revision_id=1)
    assert active_model.is_active is True


@pytest.mark.asyncio
async def test_idempotency_excludes_rolled_back():
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
    session = FakeSession({(DeploymentRevision, 1): dead_rev})

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
    existing_rev = SimpleNamespace(
        id=99,
        status="PENDING_APPROVAL",
        manifest={},
    )
    existing_appr = SimpleNamespace(
        id=42,
        run_id=7,
        target_id="99",
        requested_action="ACTIVATE",
        status="PENDING",
    )
    session = FakeSession(
        {
            (AIOptimizationRun, 7): run,
            (AIExperimentConfig, 11): config,
            (DeploymentRevision, 99): existing_rev,
            (AIApprovalRequest, 42): existing_appr,
        }
    )

    approval, revision = await propose_live_deployment(session, run_id=7)
    assert approval.id == 42
    assert revision.id == 99


@pytest.mark.asyncio
async def test_diff_broken_summary_logs_warning():
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
    session = FakeSession(
        {
            (AIOptimizationRun, 7): run,
            (AIShadowAssignment, 1): shadow_assign,
            (AIModelArtifact, 101): art,
            (AIExperimentConfig, 11): config,
        }
    )
    with patch("polyflip.ai_lab.service.logger.warning") as mock_warn:
        diff = await generate_deployment_diff(session, run_id=7)
        mock_warn.assert_called_once()
        assert diff["asset"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_transition_preserves_reason():
    run = SimpleNamespace(id=7, status="SHADOW", summary="Summary from finalization")
    session = FakeSession()
    await transition_run(session, run, "PENDING_APPROVAL", reason="Proposed for live activation")
    assert "Summary from finalization" in run.summary
    assert "Proposed for live activation" in run.summary


async def main():
    test_funcs = [
        test_hash_chain_no_bifurcation,
        test_propose_live_deployment_generates_diff_and_revision,
        test_propose_live_deployment_rejects_non_shadow_run,
        test_approve_and_activate_switches_model_registry_pointers,
        test_activate_no_artifact_raises,
        test_reject_deployment_approval_marks_revision_and_run_rejected,
        test_rollback_deployment_restores_parent_revision,
        test_rollback_no_artifact_raises,
        test_idempotency_excludes_rolled_back,
        test_propose_idempotent,
        test_diff_broken_summary_logs_warning,
        test_transition_preserves_reason,
    ]
    passed = 0
    for fn in test_funcs:
        print(f"Running {fn.__name__}...", end=" ")
        res = fn()
        if asyncio.iscoroutine(res):
            await res
        print("PASSED")
        passed += 1
    print(f"\nALL {passed} HARDENED TESTS PASSED VIA FAKESESSION!")


if __name__ == "__main__":
    asyncio.run(main())


@pytest.mark.asyncio
async def test_activation_rejects_non_activation_approval():
    approval = SimpleNamespace(
        id=5,
        target_type="DEPLOYMENT_REVISION",
        requested_action="ROLLBACK",
        target_id="10",
        status="PENDING",
    )
    session = FakeSession({(AIApprovalRequest, 5): approval})
    with pytest.raises(AILabError, match="not a deployment activation"):
        await approve_and_activate_deployment(session, approval_id=5, actor="admin")


@pytest.mark.asyncio
async def test_activation_rejects_artifact_for_wrong_asset():
    approval = SimpleNamespace(
        id=5,
        target_type="DEPLOYMENT_REVISION",
        requested_action="ACTIVATE",
        target_id="10",
        status="PENDING",
    )
    revision = SimpleNamespace(
        id=10,
        status="PENDING_APPROVAL",
        manifest={
            "models": [{"asset": "BTCUSDT", "artifact_id": 101}],
        },
    )
    artifact = SimpleNamespace(id=101, model_registry_id=2)
    wrong_model = SimpleNamespace(id=2, asset="ETHUSDT", is_active=False)
    session = FakeSession({
        (AIApprovalRequest, 5): approval,
        (DeploymentRevision, 10): revision,
        (AIModelArtifact, 101): artifact,
        (ModelRegistry, 2): wrong_model,
    })
    with pytest.raises(AILabError, match="does not match manifest asset"):
        await approve_and_activate_deployment(session, approval_id=5, actor="admin")
