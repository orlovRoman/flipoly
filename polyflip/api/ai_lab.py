"""API endpoints for autonomous AI Lab optimization lifecycle, artifacts, and guardrails."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.manifests import compute_manifest_hash
from polyflip.ai_lab.orchestrator import AILabOrchestrator
from polyflip.ai_lab.service import (
    AILabError,
    AIPermissionError,
    approve_and_activate_deployment,
    create_deployment_revision,
    create_optimization_run,
    create_run_step,
    get_run_detail,
    list_optimization_runs,
    propose_live_deployment,
    record_deployment_event,
    record_experiment_config,
    record_experiment_result,
    record_model_artifact,
    record_step_audit,
    reject_deployment_approval,
    rollback_deployment,
    transition_run,
)
from polyflip.api.auth import get_current_user
from polyflip.db.database import get_db_session
from polyflip.db.models import (
    AIApprovalRequest,
    AIExperimentConfig,
    AIModelArtifact,
    AIOptimizationRun,
    AIRunStep,
    AIShadowAssignment,
    AIStepAuditLog,
    DeploymentEvent,
    DeploymentRevision,
    ExperimentResult,
)

router = APIRouter(
    prefix="/api/ai-lab",
    tags=["ai-lab"],
    dependencies=[Depends(get_current_user)],
)


# ----------------------------------------------------------------------
# Request / Response Schemas
# ----------------------------------------------------------------------
class CreateRunRequest(BaseModel):
    objective: str = Field(..., min_length=3, max_length=500)
    scope: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: str = Field(default="AUTONOMOUS_SHADOW")


class TransitionRunRequest(BaseModel):
    target_status: str
    reason: str | None = None


class RunStepRequest(BaseModel):
    step_type: str
    sequence: int
    inputs: dict[str, Any] = Field(default_factory=dict)


class StepAuditRequest(BaseModel):
    step_id: int | None = None
    action: str
    decision_reason: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    passed_checks: bool = True
    guardrail_failures: list[str] = Field(default_factory=list)


class ExperimentConfigRequest(BaseModel):
    name: str
    asset: str
    regime: str = "DEFAULT"
    model_family: str
    feature_set: str
    feature_pipeline_version: str = "1.0"
    model_params: dict[str, Any] = Field(default_factory=dict)
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    backtest_params: dict[str, Any] = Field(default_factory=dict)
    config_hash: str
    parent_config_id: int | None = None


class ModelArtifactRequest(BaseModel):
    config_id: int
    artifact_uri: str
    artifact_hash: str
    schema_version: str = "1.0"
    feature_pipeline_version: str = "1.0"
    artifact_metadata: dict[str, Any] = Field(default_factory=dict)
    model_registry_id: int | None = None
    loadability_status: str = "PENDING"


class ExperimentResultRequest(BaseModel):
    config_id: int
    metrics: dict[str, Any]
    validation_status: str
    validation_failures: list[str] = Field(default_factory=list)


class CreateApprovalRequest(BaseModel):
    requested_action: str = Field(default="ACTIVATE")
    actor: str = Field(default="operator")
    reason: str | None = None


class ApprovalDecisionRequest(BaseModel):
    actor: str = Field(default="operator")
    reason: str | None = None


class RollbackRequest(BaseModel):
    target_revision_id: int | None = None
    actor: str = Field(default="admin")
    reason: str | None = None


class OrchestrateRunRequest(BaseModel):
    objective: str = Field(..., min_length=3, max_length=500)
    asset: str = Field(default="BTCUSDT")
    model_families: list[str] = Field(default_factory=lambda: ["LogisticRegression", "LightGBM"])
    feature_sets: list[str] = Field(default_factory=lambda: ["FS_D0", "FS_D1"])
    timeframe: str = Field(default="1d")
    days: int = Field(default=90, ge=10, le=365)
    autonomy_level: str = Field(default="AUTONOMOUS_SHADOW")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _run_payload(row: AIOptimizationRun) -> dict[str, Any]:
    return {
        "id": row.id,
        "objective": row.objective,
        "scope": row.scope,
        "autonomy_level": row.autonomy_level,
        "status": row.status,
        "summary": row.summary,
        "created_by": row.created_by,
        "permission_id": row.permission_id,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
    }


def _step_payload(row: AIRunStep) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "step_type": row.step_type,
        "sequence": row.sequence,
        "status": row.status,
        "inputs": row.inputs,
        "outputs": row.outputs,
        "error_message": row.error_message,
        "started_at": row.started_at,
        "completed_at": row.completed_at,
    }


def _audit_payload(row: AIStepAuditLog) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "step_id": row.step_id,
        "action": row.action,
        "decision_reason": row.decision_reason,
        "inputs": row.inputs,
        "outputs": row.outputs,
        "passed_checks": row.passed_checks,
        "guardrail_failures": row.guardrail_failures,
        "created_at": row.created_at,
    }


def _config_payload(row: AIExperimentConfig) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "asset": row.asset,
        "regime": row.regime,
        "model_family": row.model_family,
        "feature_set": row.feature_set,
        "feature_pipeline_version": row.feature_pipeline_version,
        "model_params": row.model_params,
        "strategy_params": row.strategy_params,
        "backtest_params": row.backtest_params,
        "config_hash": row.config_hash,
        "parent_config_id": row.parent_config_id,
        "created_at": row.created_at,
    }


def _artifact_payload(row: AIModelArtifact) -> dict[str, Any]:
    return {
        "id": row.id,
        "config_id": row.config_id,
        "artifact_uri": row.artifact_uri,
        "artifact_hash": row.artifact_hash,
        "schema_version": row.schema_version,
        "feature_pipeline_version": row.feature_pipeline_version,
        "artifact_metadata": row.artifact_metadata,
        "model_registry_id": row.model_registry_id,
        "loadability_status": row.loadability_status,
        "created_at": row.created_at,
    }


def _result_payload(row: ExperimentResult) -> dict[str, Any]:
    return {
        "id": row.id,
        "config_id": row.config_id,
        "run_id": row.run_id,
        "metrics": row.metrics,
        "validation_status": row.validation_status,
        "validation_failures": row.validation_failures,
        "created_at": row.created_at,
    }


def _revision_payload(row: DeploymentRevision) -> dict[str, Any]:
    return {
        "id": row.id,
        "revision_key": row.revision_key,
        "manifest": row.manifest,
        "manifest_hash": row.manifest_hash,
        "status": row.status,
        "parent_id": row.parent_id,
        "description": row.description,
        "created_at": row.created_at,
        "activated_at": row.activated_at,
        "superseded_at": row.superseded_at,
        "rolled_back_at": row.rolled_back_at,
    }


def _event_payload(row: DeploymentEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "revision_id": row.revision_id,
        "event_type": row.event_type,
        "actor": row.actor,
        "event_hash": row.event_hash,
        "previous_hash": row.previous_hash,
        "reason": row.reason,
        "payload": row.payload,
        "created_at": row.created_at,
    }


def _approval_payload(row: AIApprovalRequest) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "target_type": row.target_type,
        "target_id": row.target_id,
        "requested_action": row.requested_action,
        "status": row.status,
        "diff": row.diff,
        "requested_at": row.requested_at,
        "decided_at": row.decided_at,
        "decided_by": row.decided_by,
        "decision_reason": row.decision_reason,
    }


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@router.post("/runs", status_code=status.HTTP_201_CREATED)
async def create_run_endpoint(
    payload: CreateRunRequest,
    db: AsyncSession = Depends(get_db_session),
    user: Any = Depends(get_current_user),
):
    try:
        actor = getattr(user, "username", "system")
        run = await create_optimization_run(
            db,
            objective=payload.objective,
            scope=payload.scope,
            autonomy_level=payload.autonomy_level,
            created_by=actor,
        )
        await db.commit()
        await db.refresh(run)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _run_payload(run)


@router.get("/runs")
async def list_runs_endpoint(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db_session),
):
    rows = await list_optimization_runs(db, status=status_filter, limit=limit)
    return {"runs": [_run_payload(r) for r in rows]}


@router.get("/runs/{run_id}")
async def get_run_endpoint(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    detail = await get_run_detail(db, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {
        "run": _run_payload(detail["run"]),
        "steps": [_step_payload(s) for s in detail["steps"]],
        "results": [_result_payload(r) for r in detail["results"]],
        "audits": [_audit_payload(audit) for audit in detail.get("audits", [])],
        "approvals": [_approval_payload(app) for app in detail.get("approvals", [])],
    }


@router.post("/runs/{run_id}/transition")
async def transition_run_endpoint(
    run_id: int,
    payload: TransitionRunRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = await db.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    try:
        updated = await transition_run(
            db, run, payload.target_status, reason=payload.reason
        )
        await db.commit()
        await db.refresh(updated)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _run_payload(updated)


@router.post("/runs/{run_id}/steps", status_code=status.HTTP_201_CREATED)
async def create_step_endpoint(
    run_id: int,
    payload: RunStepRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = await db.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    try:
        step = await create_run_step(
            db,
            run_id=run_id,
            step_type=payload.step_type,
            sequence=payload.sequence,
            inputs=payload.inputs,
        )
        await db.commit()
        await db.refresh(step)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _step_payload(step)


@router.post("/runs/{run_id}/audits", status_code=status.HTTP_201_CREATED)
async def record_audit_endpoint(
    run_id: int,
    payload: StepAuditRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = await db.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    try:
        audit = await record_step_audit(
            db,
            run_id=run_id,
            step_id=payload.step_id,
            action=payload.action,
            decision_reason=payload.decision_reason,
            inputs=payload.inputs,
            outputs=payload.outputs,
            passed_checks=payload.passed_checks,
            guardrail_failures=payload.guardrail_failures,
        )
        await db.commit()
        await db.refresh(audit)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _audit_payload(audit)


@router.post("/configs", status_code=status.HTTP_201_CREATED)
async def create_config_endpoint(
    payload: ExperimentConfigRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        config = await record_experiment_config(
            db,
            name=payload.name,
            asset=payload.asset,
            regime=payload.regime,
            model_family=payload.model_family,
            feature_set=payload.feature_set,
            feature_pipeline_version=payload.feature_pipeline_version,
            model_params=payload.model_params,
            strategy_params=payload.strategy_params,
            backtest_params=payload.backtest_params,
            config_hash=payload.config_hash,
            parent_config_id=payload.parent_config_id,
        )
        await db.commit()
        await db.refresh(config)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _config_payload(config)


@router.post("/artifacts", status_code=status.HTTP_201_CREATED)
async def create_artifact_endpoint(
    payload: ModelArtifactRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        artifact = await record_model_artifact(
            db,
            config_id=payload.config_id,
            artifact_uri=payload.artifact_uri,
            artifact_hash=payload.artifact_hash,
            schema_version=payload.schema_version,
            feature_pipeline_version=payload.feature_pipeline_version,
            artifact_metadata=payload.artifact_metadata,
            model_registry_id=payload.model_registry_id,
            loadability_status=payload.loadability_status,
        )
        await db.commit()
        await db.refresh(artifact)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _artifact_payload(artifact)


@router.post("/results", status_code=status.HTTP_201_CREATED)
async def create_result_endpoint(
    payload: ExperimentResultRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        result = await record_experiment_result(
            db,
            config_id=payload.config_id,
            run_id=payload.run_id,
            metrics=payload.metrics,
            validation_status=payload.validation_status,
            validation_failures=payload.validation_failures,
        )
        await db.commit()
        await db.refresh(result)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _result_payload(result)


# ----------------------------------------------------------------------
# Deployment Revisions, Events, Approvals, and Safe Rollback Endpoints
# ----------------------------------------------------------------------
@router.post("/runs/{run_id}/approval", status_code=status.HTTP_201_CREATED)
async def propose_run_approval(
    run_id: int,
    payload: CreateApprovalRequest,
    db: AsyncSession = Depends(get_db_session),
    user: Any = Depends(get_current_user),
):
    actor = payload.actor or getattr(user, "username", "operator")
    try:
        approval, revision = await propose_live_deployment(
            db,
            run_id=run_id,
            actor=actor,
            reason=payload.reason,
        )
        await db.commit()
        await db.refresh(approval)
        await db.refresh(revision)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "id": approval.id,
        "run_id": approval.run_id,
        "target_type": approval.target_type,
        "target_id": approval.target_id,
        "requested_action": approval.requested_action,
        "status": approval.status,
        "diff": approval.diff,
        "requested_at": approval.requested_at,
    }


@router.get("/approvals/{approval_id}")
async def get_ai_approval(
    approval_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    app = await db.get(AIApprovalRequest, approval_id)
    if not app:
        raise HTTPException(status_code=404, detail=f"Approval {approval_id} not found")
    return _approval_payload(app)


@router.post("/approvals/{approval_id}/approve")
async def approve_ai_deployment(
    approval_id: int,
    payload: ApprovalDecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        approval, revision = await approve_and_activate_deployment(
            db,
            approval_id=approval_id,
            actor=payload.actor,
            reason=payload.reason,
        )
        await db.commit()
        await db.refresh(revision)
        await db.refresh(approval)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "APPROVED",
        "approval_id": approval.id,
        "revision_id": revision.id,
        "revision_key": revision.revision_key,
        "revision_status": revision.status,
        "activated_at": revision.activated_at,
        "decided_by": approval.decided_by,
    }


@router.post("/approvals/{approval_id}/reject")
async def reject_ai_deployment(
    approval_id: int,
    payload: ApprovalDecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        approval, revision = await reject_deployment_approval(
            db,
            approval_id=approval_id,
            actor=payload.actor,
            reason=payload.reason,
        )
        await db.commit()
        await db.refresh(approval)
        if revision:
            await db.refresh(revision)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "REJECTED",
        "approval_id": approval.id,
        "revision_id": revision.id if revision else None,
        "decided_by": approval.decided_by,
        "decision_reason": approval.decision_reason,
        "decided_at": approval.decided_at,
    }


@router.post("/deployments/rollback")
async def rollback_ai_deployment(
    payload: RollbackRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        rolled_back, restored = await rollback_deployment(
            db,
            target_revision_id=payload.target_revision_id,
            actor=payload.actor,
            reason=payload.reason,
        )
        await db.commit()
        if rolled_back:
            await db.refresh(rolled_back)
        await db.refresh(restored)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "ROLLED_BACK",
        "rolled_back_revision_id": rolled_back.id if rolled_back else None,
        "restored_revision_id": restored.id,
        "restored_revision_key": restored.revision_key,
        "activated_at": restored.activated_at,
    }


@router.get("/deployments/revisions")
async def list_deployment_revisions(
    db: AsyncSession = Depends(get_db_session),
):
    rows = (
        await db.execute(
            select(DeploymentRevision).order_by(DeploymentRevision.id.desc())
        )
    ).scalars().all()
    return {"revisions": [_revision_payload(r) for r in rows]}


@router.get("/deployments/revisions/{revision_id}/events")
async def list_revision_events(
    revision_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    rev = await db.get(DeploymentRevision, revision_id)
    if not rev:
        raise HTTPException(status_code=404, detail=f"Revision {revision_id} not found")

    rows = (
        await db.execute(
            select(DeploymentEvent)
            .where(DeploymentEvent.revision_id == revision_id)
            .order_by(DeploymentEvent.id.asc())
        )
    ).scalars().all()
    return {"events": [_event_payload(e) for e in rows]}


@router.post("/orchestrate", status_code=status.HTTP_201_CREATED)
async def orchestrate_optimization_pipeline(
    payload: OrchestrateRunRequest,
    db: AsyncSession = Depends(get_db_session),
    user: Any = Depends(get_current_user),
):
    actor = getattr(user, "username", "system")
    orchestrator = AILabOrchestrator(db)
    try:
        run, winner_config, winner_artifact, report = await orchestrator.execute_full_optimization(
            objective=payload.objective,
            asset=payload.asset,
            model_families=payload.model_families,
            feature_sets=payload.feature_sets,
            timeframe=payload.timeframe,
            days=payload.days,
            autonomy_level=payload.autonomy_level,
            actor=actor,
        )
        await db.commit()
        await db.refresh(run)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}") from exc

    return {
        "status": "COMPLETED",
        "run": _run_payload(run),
        "winner_config": _config_payload(winner_config) if winner_config else None,
        "winner_artifact": _artifact_payload(winner_artifact) if winner_artifact else None,
        "recommendation_status": report.get("recommendation_status"),
        "median_oot_pnl": report.get("median_oot_pnl"),
    }
