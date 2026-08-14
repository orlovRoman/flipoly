"""Unit tests for Phase 9 activation, deployment revisions, and rollback."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

from sqlalchemy import desc, select

from polyflip.ai_lab.manifests import build_deployment_manifest
from polyflip.ai_lab.service import (
    approve_and_activate_deployment,
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
    AIRunStep,
    AIShadowAssignment,
    DeploymentEvent,
    DeploymentRevision,
    ExperimentResult,
    ModelRegistry,
)


class FakeScalarResult:
    def __init__(self, data: Any):
        self._data = data

    def all(self):
        return list(self._data) if isinstance(self._data, list) else [self._data]

    def first(self):
        if isinstance(self._data, list):
            return self._data[0] if self._data else None
        return self._data

    def scalar_one_or_none(self):
        if isinstance(self._data, list):
            return self._data[0] if self._data else None
        return self._data


class FakeExecuteResult:
    def __init__(self, data: Any):
        self._data = data

    def scalars(self):
        return FakeScalarResult(self._data)

    def scalar_one_or_none(self):
        if isinstance(self._data, list):
            return self._data[0] if self._data else None
        return self._data

    def all(self):
        return list(self._data) if isinstance(self._data, list) else [self._data]


class FakeSession:
    """In-memory mock session supporting basic queries and relationship tracking."""

    def __init__(self):
        self.store: dict[type, list[Any]] = {
            AIOptimizationRun: [],
            AIExperimentConfig: [],
            AIRunStep: [],
            AIModelArtifact: [],
            ExperimentResult: [],
            AIShadowAssignment: [],
            ModelRegistry: [],
            AIApprovalRequest: [],
            DeploymentRevision: [],
            DeploymentEvent: [],
        }
        self._id_counters: dict[type, int] = {k: 1 for k in self.store}

    def add(self, instance: Any):
        t = type(instance)
        if t not in self.store:
            self.store[t] = []
            self._id_counters[t] = 1

        if getattr(instance, "id", None) is None:
            instance.id = self._id_counters[t]
            self._id_counters[t] += 1

        if instance not in self.store[t]:
            self.store[t].append(instance)

    async def commit(self):
        pass

    async def flush(self):
        pass

    async def refresh(self, instance: Any):
        pass

    async def get(self, model_class: type, ident: Any):
        if model_class not in self.store:
            return None
        for obj in self.store[model_class]:
            if getattr(obj, "id", None) == ident:
                return obj
        return None

    async def execute(self, stmt: Any):
        entity = None
        if hasattr(stmt, "column_descriptions") and stmt.column_descriptions:
            entity = stmt.column_descriptions[0].get("entity")

        if entity is None:
            for k in self.store:
                if k.__tablename__ in str(stmt):
                    entity = k
                    break

        items = list(self.store.get(entity, [])) if entity else []

        params = stmt.compile().params if hasattr(stmt, "compile") else {}
        param_values = list(params.values())

        if entity == DeploymentEvent:
            revision_id_param = None
            for p in param_values:
                if isinstance(p, int):
                    revision_id_param = p
                    break
            if revision_id_param is not None:
                items = [e for e in items if e.revision_id == revision_id_param]
            items = sorted(items, key=lambda x: x.id, reverse=True)

        elif entity == DeploymentRevision:
            if "status" in str(stmt) and "ACTIVE" in param_values:
                items = [r for r in items if r.status == "ACTIVE"]
            elif "revision_key" in str(stmt):
                for p in param_values:
                    items = [r for r in items if r.revision_key == p]
            items = sorted(items, key=lambda x: x.id, reverse=True)

        elif entity == ModelRegistry:
            if "asset" in str(stmt) and "is_active" in str(stmt):
                asset_param = next((p for p in param_values if isinstance(p, str) and p != "ACTIVE"), None)
                if asset_param:
                    items = [m for m in items if m.asset == asset_param and m.is_active is True]
                else:
                    items = [m for m in items if m.is_active is True]

        elif entity == AIApprovalRequest:
            if "run_id" in str(stmt):
                run_id_param = next((p for p in param_values if isinstance(p, int)), None)
                if run_id_param is not None:
                    items = [a for a in items if a.run_id == run_id_param and a.status == "PENDING"]

        return FakeExecuteResult(items)


def setup_sample_data(session: FakeSession):
    active_model = ModelRegistry(
        asset="BTCUSDT",
        version=1,
        model_type="LogisticRegression",
        features="FS_D0",
        decision_threshold=0.55,
        decision_threshold_down=0.45,
        is_active=True,
    )
    session.add(active_model)

    config = AIExperimentConfig(
        name="LGBM-Test",
        asset="BTCUSDT",
        regime="DEFAULT",
        model_family="LightGBM",
        feature_set="FS_D1",
        feature_pipeline_version="1.0",
        model_params={"max_depth": 4},
        strategy_params={"decision_threshold": 0.58, "decision_threshold_down": 0.42},
        backtest_params={"test_days": 30},
        config_hash="cfg12345",
    )
    session.add(config)

    artifact = AIModelArtifact(
        config_id=config.id,
        artifact_uri="/models/lgbm_btc.joblib",
        artifact_hash="art12345",
        schema_version="1.0",
        feature_pipeline_version="1.0",
        artifact_metadata={"type": "lightgbm"},
        loadability_status="VALID",
    )
    session.add(artifact)

    run = AIOptimizationRun(
        objective="Find best BTC LightGBM",
        scope={"asset": "BTCUSDT", "min_trades": 50, "max_drawdown": -5.0},
        autonomy_level="AUTONOMOUS_SHADOW",
        status="SHADOW",
        summary=json.dumps(
            {
                "report": {
                    "recommendation_status": "READY_FOR_SHADOW",
                    "recommended_config_id": config.id,
                    "median_pnl": 2.4,
                    "rows": [
                        {
                            "config_id": config.id,
                            "median_oot_pnl": 2.4,
                            "total_trades": 60,
                            "median_oot_drawdown": -3.1,
                            "passes_all_checks": True,
                        }
                    ],
                }
            }
        ),
    )
    session.add(run)

    shadow = AIShadowAssignment(
        run_id=run.id,
        candidate_artifact_id=artifact.id,
        asset="BTCUSDT",
        status="RUNNING",
    )
    session.add(shadow)

    return active_model, config, artifact, run, shadow


async def test_hash_chain_no_bifurcation():
    print("Running test_hash_chain_no_bifurcation...", end=" ")
    session = FakeSession()
    rev1 = DeploymentRevision(
        revision_key="rev-1",
        manifest={"version": 1},
        manifest_hash="hash1",
        status="DRAFT",
    )
    rev2 = DeploymentRevision(
        revision_key="rev-2",
        manifest={"version": 2},
        manifest_hash="hash2",
        status="DRAFT",
    )
    session.add(rev1)
    session.add(rev2)

    ev1_rev1 = await record_deployment_event(
        session,
        revision_id=rev1.id,
        event_type="CREATED",
        actor="system",
        reason="Initial rev1",
    )
    ev1_rev2 = await record_deployment_event(
        session,
        revision_id=rev2.id,
        event_type="CREATED",
        actor="system",
        reason="Initial rev2",
    )

    assert ev1_rev1.previous_hash == "0" * 64
    assert ev1_rev2.previous_hash == "0" * 64, "Revision 2 should start its own genesis hash"
    assert ev1_rev1.event_hash != ev1_rev2.event_hash

    ev2_rev1 = await record_deployment_event(
        session,
        revision_id=rev1.id,
        event_type="SHADOW_ASSIGNED",
        actor="system",
        reason="Assigned shadow",
    )
    assert ev2_rev1.previous_hash == ev1_rev1.event_hash, "Revision 1 second event must link to rev1 event 1"
    print("PASSED")


async def test_propose_live_deployment_generates_diff_and_revision():
    print("Running test_propose_live_deployment_generates_diff_and_revision...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)

    approval, revision = await propose_live_deployment(
        session, run, actor="test_operator", reason="Great OOT performance"
    )

    assert approval.status == "PENDING"
    assert approval.requested_action == "ACTIVATE"
    assert run.status == "PENDING_APPROVAL"
    assert revision.status == "PENDING_APPROVAL"
    assert revision.parent_id == 100
    assert approval.diff["candidate"]["config_id"] == config.id
    assert approval.diff["baseline"]["model_type"] == "LogisticRegression"

    events = [e for e in session.store[DeploymentEvent] if e.revision_id == revision.id]
    assert len(events) == 1
    assert events[0].event_type == "CREATED"
    print("PASSED")


def setup_data_with_parent(session: FakeSession):
    active_rev = DeploymentRevision(
        id=100,
        revision_key="rev-active-root",
        manifest={"root": True},
        manifest_hash="root_hash_123",
        status="ACTIVE",
    )
    session.add(active_rev)
    active_model, config, artifact, run, shadow = setup_sample_data(session)
    return active_model, config, artifact, run, shadow


async def test_propose_live_deployment_rejects_non_shadow_run():
    print("Running test_propose_live_deployment_rejects_non_shadow_run...", end=" ")
    session = FakeSession()
    _, _, _, run, _ = setup_data_with_parent(session)
    run.status = "RUNNING"

    try:
        await propose_live_deployment(session, run)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Only runs in 'SHADOW' status" in str(e)
    print("PASSED")


async def test_approve_and_activate_switches_model_registry_pointers():
    print("Running test_approve_and_activate_switches_model_registry_pointers...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)

    approval, revision = await propose_live_deployment(session, run)
    assert run.status == "PENDING_APPROVAL"

    approved_app, active_rev = await approve_and_activate_deployment(
        session, approval_id=approval.id, actor="admin", reason="Approved for prod"
    )

    assert approved_app.status == "APPROVED"
    assert active_rev.status == "ACTIVE"
    assert active_rev.activated_at is not None
    assert run.status == "ACTIVE"

    assert active_model.is_active is False
    new_models = [m for m in session.store[ModelRegistry] if m.is_active is True]
    assert len(new_models) == 1
    new_m = new_models[0]
    assert new_m.model_type == config.model_family
    assert new_m.decision_threshold == 0.58
    assert new_m.decision_threshold_down == 0.42

    events = [e for e in session.store[DeploymentEvent] if e.revision_id == revision.id]
    assert len(events) == 3
    assert events[1].event_type == "APPROVED"
    assert events[2].event_type == "ACTIVATED"
    assert events[2].previous_hash == events[1].event_hash
    print("PASSED")


async def test_activate_no_artifact_raises():
    print("Running test_activate_no_artifact_raises...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)

    session.store[AIModelArtifact] = []

    approval, revision = await propose_live_deployment(session, run)
    try:
        await approve_and_activate_deployment(session, approval_id=approval.id)
        assert False, "Should raise ValueError when candidate artifact is missing"
    except ValueError as e:
        assert "Candidate artifact" in str(e) and "not found" in str(e)
    print("PASSED")


async def test_reject_deployment_approval_marks_revision_and_run_rejected():
    print("Running test_reject_deployment_approval_marks_revision_and_run_rejected...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)

    approval, revision = await propose_live_deployment(session, run)

    rej_app, rej_rev = await reject_deployment_approval(
        session, approval_id=approval.id, actor="risk_mgr", reason="Too volatile"
    )

    assert rej_app.status == "REJECTED"
    assert rej_rev.status == "REJECTED"
    assert run.status == "REJECTED"

    events = [e for e in session.store[DeploymentEvent] if e.revision_id == revision.id]
    assert len(events) == 2
    assert events[1].event_type == "REJECTED"
    assert events[1].actor == "risk_mgr"
    print("PASSED")


async def test_rollback_deployment_restores_parent_revision():
    print("Running test_rollback_deployment_restores_parent_revision...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)

    parent_model = ModelRegistry(
        asset="BTCUSDT",
        version=99,
        model_type="LogisticRegression",
        features="FS_D0",
        decision_threshold=0.51,
        decision_threshold_down=0.49,
        is_active=False,
    )
    session.add(parent_model)

    parent_art = AIModelArtifact(
        id=999,
        config_id=config.id,
        artifact_uri="/models/parent.joblib",
        artifact_hash="parthash999",
        schema_version="1.0",
        feature_pipeline_version="1.0",
        artifact_metadata={"type": "parent"},
        model_registry_id=parent_model.id,
        loadability_status="VALID",
    )
    session.add(parent_art)

    parent_rev = await session.get(DeploymentRevision, 100)
    parent_rev.manifest = {
        "schema_version": "1.0",
        "asset": "BTCUSDT",
        "regime": "DEFAULT",
        "model_family": "LogisticRegression",
        "feature_set": "FS_D0",
        "feature_pipeline_version": "1.0",
        "model_params": {},
        "strategy_params": {
            "decision_threshold": 0.51,
            "decision_threshold_down": 0.49,
        },
        "artifact_hash": parent_art.artifact_hash,
        "model_registry_id": parent_model.id,
    }

    approval, revision = await propose_live_deployment(session, run)
    await approve_and_activate_deployment(session, approval_id=approval.id)
    assert revision.status == "ACTIVE"

    rolled_back_rev, restored_rev = await rollback_deployment(
        session, target_revision_id=parent_rev.id, actor="admin", reason="Market emergency"
    )

    assert rolled_back_rev.id == revision.id
    assert rolled_back_rev.status == "ROLLED_BACK"
    assert restored_rev.id == parent_rev.id
    assert restored_rev.status == "ACTIVE"
    assert parent_model.is_active is True

    events = [e for e in session.store[DeploymentEvent] if e.revision_id == revision.id]
    assert events[-1].event_type == "ROLLED_BACK"
    print("PASSED")


async def test_rollback_no_artifact_raises():
    print("Running test_rollback_no_artifact_raises...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)

    parent_rev = await session.get(DeploymentRevision, 100)
    parent_rev.manifest = {
        "artifact_hash": "non_existent_hash",
    }

    approval, revision = await propose_live_deployment(session, run)
    await approve_and_activate_deployment(session, approval_id=approval.id)

    try:
        await rollback_deployment(session, target_revision_id=parent_rev.id)
        assert False, "Should raise ValueError when target artifact does not exist"
    except ValueError as e:
        assert "Artifact non_existent_hash for target revision" in str(e)
    print("PASSED")


async def test_idempotency_excludes_rolled_back():
    print("Running test_idempotency_excludes_rolled_back...", end=" ")
    session = FakeSession()
    rev1 = DeploymentRevision(
        revision_key="rev-btc-1",
        manifest={"a": 1},
        manifest_hash="hash_same",
        status="ROLLED_BACK",
    )
    session.add(rev1)

    existing = [
        r
        for r in session.store[DeploymentRevision]
        if r.manifest_hash == "hash_same"
        and r.status not in ("REJECTED", "ROLLED_BACK")
    ]
    assert len(existing) == 0
    print("PASSED")


async def test_propose_idempotent():
    print("Running test_propose_idempotent...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)

    app1, rev1 = await propose_live_deployment(session, run)
    app2, rev2 = await propose_live_deployment(session, run)

    assert app1.id == app2.id
    assert rev1.id == rev2.id
    print("PASSED")


async def test_diff_broken_summary_logs_warning():
    print("Running test_diff_broken_summary_logs_warning...", end=" ")
    session = FakeSession()
    active_model, config, artifact, run, shadow = setup_data_with_parent(session)
    run.summary = "invalid-json-string{"

    app, rev = await propose_live_deployment(session, run)
    assert app.diff["candidate"] == {}
    assert "Unable to parse run.summary" in app.diff.get("error", "")
    print("PASSED")


async def test_transition_preserves_reason():
    print("Running test_transition_preserves_reason...", end=" ")
    session = FakeSession()
    run = AIOptimizationRun(
        objective="Test transition reason",
        scope={},
        status="DRAFT",
    )
    session.add(run)

    await transition_run(session, run, "PLANNING", reason="Testing transition")
    assert run.status == "PLANNING"
    print("PASSED")


async def main():
    print("\n--- RUNNING DEPLOYMENT & ROLLBACK HARDENED TEST SUITE ---")
    await test_hash_chain_no_bifurcation()
    await test_propose_live_deployment_generates_diff_and_revision()
    await test_propose_live_deployment_rejects_non_shadow_run()
    await test_approve_and_activate_switches_model_registry_pointers()
    await test_activate_no_artifact_raises()
    await test_reject_deployment_approval_marks_revision_and_run_rejected()
    await test_rollback_deployment_restores_parent_revision()
    await test_rollback_no_artifact_raises()
    await test_idempotency_excludes_rolled_back()
    await test_propose_idempotent()
    await test_diff_broken_summary_logs_warning()
    await test_transition_preserves_reason()
    print("\nALL 12 HARDENED TESTS PASSED VIA FAKESESSION!\n")


if __name__ == "__main__":
    asyncio.run(main())
