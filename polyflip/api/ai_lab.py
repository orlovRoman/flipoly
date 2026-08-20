"""HTTP API for the safe AI Lab experiment contour.

The router exposes only experiment, audit and approval operations. It does not
activate models or mutate live execution settings.
"""

from __future__ import annotations

from polyflip.ai_lab.agent import AILabAgent
from polyflip.ai_lab.agent_tools import expire_overlays, rollback_overlay
from polyflip.db.models import AIConfigOverlay

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
import structlog
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.executor import ExecutionBatchError
from polyflip.ai_lab.lgbm_worker import MAX_LGBM_WORKER_STEPS, execute_lgbm_steps
from polyflip.ai_lab.scheduler import (
    MAX_LEASE_TTL_SECONDS,
    MAX_SCHEDULER_INTERVAL_SECONDS,
    MAX_SCHEDULER_ITERATIONS,
    MIN_LEASE_TTL_SECONDS,
    run_lgbm_scheduler,
)
from polyflip.ai_lab.service import (
    AILabError,
    AIPermissionError,
    AIRunTransitionError,
    append_step,
    approve_and_activate_deployment,
    authorize_run_action,
    create_deployment_revision,
    create_experiment_config,
    create_permission,
    create_run,
    get_run_detail,
    propose_live_deployment,
    reject_deployment_approval,
    request_approval,
    rollback_deployment,
    transition_action_for_target,
    transition_run,
    utc_now,
)
from polyflip.ai_lab.orchestrator import (
    claim_next_step,
    evaluate_run,
    finalize_run,
    plan_run,
    promote_to_shadow,
    record_result,
)
from polyflip.api.auth import verify_api_key
from polyflip.db.connection import get_db_session
from polyflip.db.models import (
    AIApprovalRequest,
    AIExperimentConfig,
    AIOptimizationRun,
    AIPermission,
    AIRunStep,
    DeploymentEvent,
    DeploymentRevision,
)

router = APIRouter(
    prefix="/api/ai-lab",
    tags=["ai-lab"],
    dependencies=[Depends(verify_api_key)],
)

logger = structlog.get_logger(__name__)


class WorkerRunRequest(BaseModel):
    max_steps: int = Field(default=1, ge=1, le=MAX_LGBM_WORKER_STEPS)


class SchedulerRunRequest(BaseModel):
    max_iterations: int = Field(default=1, ge=1, le=MAX_SCHEDULER_ITERATIONS)
    max_steps: int = Field(default=1, ge=1, le=MAX_LGBM_WORKER_STEPS)
    interval_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=MAX_SCHEDULER_INTERVAL_SECONDS,
    )
    lease_ttl_seconds: float = Field(
        default=120.0,
        ge=MIN_LEASE_TTL_SECONDS,
        le=MAX_LEASE_TTL_SECONDS,
    )


def _execution_payload(outcome: Any) -> dict[str, Any]:
    return {
        "run_id": outcome.run_id,
        "step_id": outcome.step_id,
        "action": outcome.action,
        "evaluation_kind": outcome.evaluation_kind,
        "status": outcome.status,
        "result_id": outcome.result_id,
        "error_code": outcome.error_code,
        "config_id": outcome.config_id,
    }


class RunCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=4000)
    scope: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: str = "EXPERIMENT"
    budget_experiments: int = Field(default=1, ge=1, le=10000)
    budget_seconds: int = Field(default=0, ge=0, le=7 * 24 * 3600)
    created_by: str = Field(default="api", max_length=128)
    permission_id: int | None = Field(
        default=None,
        description="Concrete AIPermission.id version captured as the run snapshot.",
    )


class TransitionRequest(BaseModel):
    target: str
    reason: str | None = Field(default=None, max_length=4000)


class StepCreateRequest(BaseModel):
    step_index: int = Field(ge=0)
    step_type: str = Field(min_length=1, max_length=32)
    status: str = "SUCCEEDED"
    hypothesis: str | None = None
    action: str | None = Field(default=None, max_length=64)
    input_payload: dict[str, Any] | None = None
    output_payload: dict[str, Any] | None = None
    summary: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class ConfigCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    model_family: str = Field(min_length=1, max_length=32)
    feature_set: str = Field(min_length=1, max_length=32)
    feature_pipeline_version: str = Field(min_length=1, max_length=64)
    model_params: dict[str, Any] = Field(default_factory=dict)
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    backtest_params: dict[str, Any] = Field(default_factory=dict)
    asset: str | None = Field(default=None, max_length=32)
    regime: str | None = Field(default=None, max_length=32)
    description: str | None = None
    created_by: str = Field(default="api", max_length=128)
    parent_id: int | None = None




class PermissionCreateRequest(BaseModel):
    profile_name: str = Field(min_length=1, max_length=64)
    allowed_actions: list[str] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)
    updated_by: str = Field(default="api", max_length=128)
    enabled: bool = True


class ActionCheckRequest(BaseModel):
    action: str = Field(min_length=1, max_length=64)


class ApprovalRequest(BaseModel):
    target_type: str = Field(default="DEPLOYMENT_REVISION", min_length=1, max_length=32)
    target_id: str | None = Field(default=None, max_length=64)
    requested_action: str = Field(default="ACTIVATE", min_length=1, max_length=32)
    diff: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecisionRequest(BaseModel):
    actor: str = Field(default="admin", min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=4000)


class RollbackRequest(BaseModel):
    target_revision_id: int | None = Field(default=None, gt=0)
    actor: str = Field(default="admin", min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=4000)


class PlanRequest(BaseModel):
    config_ids: list[int] = Field(min_length=1, max_length=1000)


class ResultCreateRequest(BaseModel):
    config_id: int = Field(gt=0)
    evaluation_kind: Literal["TRAIN", "OOT", "POLYMARKET_OOT", "SHADOW"]
    status: str = Field(default="SUCCEEDED", min_length=1, max_length=24)
    metrics: dict[str, Any] | None = None
    slices: dict[str, Any] | None = None
    trade_count: int | None = Field(default=None, ge=0)
    net_pnl: float | None = None
    max_drawdown: float | None = None
    artifact_id: int | None = Field(default=None, gt=0)
    step_id: int | None = Field(default=None, gt=0)
    code_sha: str | None = Field(default=None, max_length=64)
    dataset_fingerprint: str | None = Field(default=None, max_length=128)
    train_window_start: datetime | None = None
    train_window_end: datetime | None = None
    oot_window_start: datetime | None = None
    oot_window_end: datetime | None = None
    summary: str | None = Field(default=None, max_length=4000)
    error_code: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=4000)


class ShadowPromoteRequest(BaseModel):
    candidate_artifact_id: int = Field(gt=0)
    baseline_artifact_id: int | None = Field(default=None, gt=0)
    asset: str = Field(min_length=1, max_length=32)
    regime: str | None = Field(default=None, max_length=32)


class FinalizeRunRequest(BaseModel):
    """Post-worker finalization; SHADOW is optional, ACTIVE is impossible."""

    auto_shadow: bool = True
    asset: str | None = Field(default=None, max_length=32)
    regime: str | None = Field(default=None, max_length=32)
    candidate_artifact_id: int | None = Field(default=None, gt=0)
    baseline_artifact_id: int | None = Field(default=None, gt=0)


def _run_payload(run: AIOptimizationRun) -> dict[str, Any]:
    from polyflip.config import settings
    return {
        "id": run.id,
        "objective": run.objective,
        "scope": run.scope,
        "autonomy_level": run.autonomy_level,
        "status": run.status,
        "ai_lab_mode": getattr(settings, "AI_LAB_MODE", "STANDARD"),
        "agent_type": run.agent_type,
        "budget_experiments": run.budget_experiments,
        "budget_seconds": run.budget_seconds,
        "created_by": run.created_by,
        "permission_id": run.permission_id,
        "summary": run.summary,
        "error": run.error,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
    }


def _step_payload(step: AIRunStep) -> dict[str, Any]:
    return {
        "id": step.id,
        "run_id": step.run_id,
        "step_index": step.step_index,
        "step_type": step.step_type,
        "status": step.status,
        "hypothesis": step.hypothesis,
        "action": step.action,
        "input_payload": step.input_payload,
        "output_payload": step.output_payload,
        "summary": step.summary,
        "error_code": step.error_code,
        "error_message": step.error_message,
        "created_at": step.created_at,
        "started_at": step.started_at,
        "finished_at": step.finished_at,
    }


def _audit_payload(audit: Any) -> dict[str, Any]:
    return {
        "id": audit.id,
        "run_id": audit.run_id,
        "step_id": audit.step_id,
        "config_id": audit.config_id,
        "action": audit.action,
        "error_code": audit.error_code,
        "error_message": audit.error_message,
        "payload": audit.payload,
        "created_at": audit.created_at,
    }


def _assignment_payload(assignment: Any) -> dict[str, Any]:
    return {
        "id": assignment.id,
        "run_id": assignment.run_id,
        "candidate_artifact_id": assignment.candidate_artifact_id,
        "baseline_artifact_id": assignment.baseline_artifact_id,
        "asset": assignment.asset,
        "regime": assignment.regime,
        "status": assignment.status,
        "metrics": assignment.metrics,
        "started_at": assignment.started_at,
        "ended_at": assignment.ended_at,
        "created_at": assignment.created_at,
    }


def _approval_payload(app: Any) -> dict[str, Any]:
    return {
        "id": app.id,
        "run_id": app.run_id,
        "target_type": app.target_type,
        "target_id": app.target_id,
        "requested_action": app.requested_action,
        "diff": app.diff,
        "status": app.status,
        "requested_at": app.requested_at,
        "decided_at": app.decided_at,
        "decided_by": app.decided_by,
        "decision_reason": app.decision_reason,
    }


@router.post("/permissions", status_code=201)
async def create_ai_permission(
    payload: PermissionCreateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        row = await create_permission(
            db,
            profile_name=payload.profile_name,
            allowed_actions=payload.allowed_actions,
            scope=payload.scope,
            limits=payload.limits,
            updated_by=payload.updated_by,
            enabled=payload.enabled,
        )
        await db.commit()
        await db.refresh(row)
    except (AILabError, AIPermissionError) as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "id": row.id,
        "profile_name": row.profile_name,
        "version": row.version,
        "is_current": row.is_current,
        "allowed_actions": row.allowed_actions,
        "scope": row.scope,
        "limits": row.limits,
        "enabled": row.enabled,
    }


@router.get("/permissions")
async def list_ai_permissions(db: AsyncSession = Depends(get_db_session)):
    rows = (
        await db.execute(
            select(AIPermission)
            .order_by(AIPermission.profile_name, AIPermission.version.desc())
        )
    ).scalars().all()
    return {
        "permissions": [
            {
                "id": row.id,
                "profile_name": row.profile_name,
                "version": row.version,
                "is_current": row.is_current,
                "allowed_actions": row.allowed_actions,
                "scope": row.scope,
                "limits": row.limits,
                "enabled": row.enabled,
            }
            for row in rows
        ]
    }


@router.post("/runs/{run_id}/actions/check")
async def check_ai_action(
    run_id: int,
    payload: ActionCheckRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        run = await authorize_run_action(db, run_id, payload.action)
    except AILabError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"run_id": run.id, "action": payload.action.upper(), "allowed": True}


@router.post("/runs", status_code=201)
async def create_ai_run(payload: RunCreateRequest, db: AsyncSession = Depends(get_db_session)):
    permission = None
    if payload.permission_id is not None:
        # Lock the concrete permission version while the run captures its
        # immutable snapshot; profile updates cannot race this read.
        permission = (
            await db.execute(
                select(AIPermission)
                .where(AIPermission.id == payload.permission_id)
                .with_for_update(read=True)
            )
        ).scalar_one_or_none()
    if payload.permission_id is not None and permission is None:
        raise HTTPException(status_code=404, detail="permission profile not found")
    if payload.permission_id is None and payload.autonomy_level.upper() != "OBSERVE":
        raise HTTPException(
            status_code=422,
            detail="permission_id is required for non-OBSERVE AI Lab runs",
        )
    try:
        run = await create_run(
            db,
            objective=payload.objective,
            scope=payload.scope,
            autonomy_level=payload.autonomy_level.upper(),
            budget_experiments=payload.budget_experiments,
            budget_seconds=payload.budget_seconds,
            created_by=payload.created_by,
            permission=permission,
        )
        await db.commit()
        await db.refresh(run)
    except (AILabError, AIPermissionError) as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _run_payload(run)


@router.get("/runs")
async def list_ai_runs(
    status: str | None = None,
    created_by: str | None = None,
    before_id: int | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
):
    # Cursor pagination is stable even when multiple runs share a timestamp.
    limit = min(max(limit, 1), 100)
    query = select(AIOptimizationRun).order_by(
        AIOptimizationRun.id.desc()
    ).limit(limit)
    if status:
        query = query.where(AIOptimizationRun.status == status.strip().upper())
    if created_by:
        query = query.where(AIOptimizationRun.created_by == created_by.strip())
    if before_id is not None:
        query = query.where(AIOptimizationRun.id < before_id)
    rows = (await db.execute(query)).scalars().all()
    next_before_id = rows[-1].id if len(rows) == limit else None
    return {
        "runs": [_run_payload(row) for row in rows],
        "next_before_id": next_before_id,
    }


@router.get("/runs/{run_id}")
async def get_ai_run(run_id: int, db: AsyncSession = Depends(get_db_session)):
    detail = await get_run_detail(db, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="AI Lab run not found")
    return {
        "run": _run_payload(detail["run"]),
        "steps": [_step_payload(step) for step in detail["steps"]],
        "results": [
            {
                "id": result.id,
                "config_id": result.config_id,
                "artifact_id": result.artifact_id,
                "evaluation_kind": result.evaluation_kind,
                "status": result.status,
                "metrics": result.metrics,
                "slices": result.slices,
                "trade_count": result.trade_count,
                "net_pnl": result.net_pnl,
                "max_drawdown": result.max_drawdown,
                "code_sha": result.code_sha,
                "dataset_fingerprint": result.dataset_fingerprint,
                "train_window_start": result.train_window_start,
                "train_window_end": result.train_window_end,
                "oot_window_start": result.oot_window_start,
                "oot_window_end": result.oot_window_end,
                "created_at": result.created_at,
            }
            for result in detail["results"]
        ],
        "audits": [_audit_payload(audit) for audit in detail.get("audits", [])],
        "approvals": [_approval_payload(app) for app in detail.get("approvals", [])],
    }


@router.post("/runs/{run_id}/plan")
async def plan_ai_run(
    run_id: int,
    payload: PlanRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        steps = await plan_run(db, run_id, payload.config_ids)
        await db.commit()
        for step in steps:
            await db.refresh(step)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run_id, "steps": [_step_payload(step) for step in steps]}


@router.post("/runs/{run_id}/steps/claim")
async def claim_ai_step(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        step = await claim_next_step(db, run_id)
        if step is not None:
            await db.commit()
            await db.refresh(step)
        else:
            await db.rollback()
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run_id, "step": _step_payload(step) if step else None}


@router.post("/runs/{run_id}/worker/lgbm/schedule")
async def schedule_ai_lgbm_worker(
    run_id: int,
    payload: SchedulerRunRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Run a finite leased sequence of offline LightGBM worker batches."""
    run = await db.get(AIOptimizationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="AI Lab run not found")
    if run.status not in {"PLANNING", "RUNNING"}:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} cannot execute from {run.status}",
        )
    try:
        result = await run_lgbm_scheduler(
            db,
            run_id,
            max_iterations=payload.max_iterations,
            max_steps=payload.max_steps,
            interval_seconds=payload.interval_seconds,
            lease_ttl_seconds=payload.lease_ttl_seconds,
        )
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        await db.rollback()
        logger.exception(
            "ai_lab_lgbm_scheduler_failed",
            run_id=run_id,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail="AI Lab scheduler failed") from exc

    logger.info(
        "ai_lab_lgbm_scheduler_completed",
        run_id=run_id,
        status=result.status,
        iterations=result.iterations,
        stop_reason=result.stop_reason,
    )
    return {
        "status": result.status,
        "run_id": result.run_id,
        "owner_token": result.owner_token,
        "iterations": result.iterations,
        "stop_reason": result.stop_reason,
        "outcomes": [_execution_payload(item) for item in result.outcomes],
    }


@router.post("/runs/{run_id}/worker/lgbm")
async def execute_ai_lgbm_worker(
    run_id: int,
    payload: WorkerRunRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Execute a bounded offline LightGBM batch and return its audit outcomes."""
    run = await db.get(AIOptimizationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="AI Lab run not found")
    if run.status not in {"PLANNING", "RUNNING"}:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} cannot execute from {run.status}",
        )
    try:
        outcomes = await execute_lgbm_steps(
            db,
            run_id,
            max_steps=payload.max_steps,
        )
    except ExecutionBatchError as exc:
        await db.rollback()
        logger.exception(
            "ai_lab_lgbm_worker_partial_failure",
            run_id=run_id,
            completed=len(exc.completed),
            error=str(exc.cause),
        )
        return {
            "status": "partial_failure",
            "run_id": run_id,
            "completed": [_execution_payload(item) for item in exc.completed],
            "error": str(exc.cause)[:4000],
        }
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "ai_lab_lgbm_worker_batch_completed",
        run_id=run_id,
        requested_steps=payload.max_steps,
        completed_steps=len(outcomes),
        statuses=[item.status for item in outcomes],
    )
    return {
        "status": "completed",
        "run_id": run_id,
        "requested_steps": payload.max_steps,
        "completed_steps": len(outcomes),
        "outcomes": [_execution_payload(item) for item in outcomes],
    }


@router.post("/runs/{run_id}/results", status_code=201)
async def record_ai_result(
    run_id: int,
    payload: ResultCreateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        result = await record_result(
            db,
            run_id=run_id,
            config_id=payload.config_id,
            evaluation_kind=payload.evaluation_kind,
            status=payload.status,
            metrics=payload.metrics,
            slices=payload.slices,
            trade_count=payload.trade_count,
            net_pnl=payload.net_pnl,
            max_drawdown=payload.max_drawdown,
            artifact_id=payload.artifact_id,
            step_id=payload.step_id,
            code_sha=payload.code_sha,
            dataset_fingerprint=payload.dataset_fingerprint,
            train_window_start=payload.train_window_start,
            train_window_end=payload.train_window_end,
            oot_window_start=payload.oot_window_start,
            oot_window_end=payload.oot_window_end,
            summary=payload.summary,
            error_code=payload.error_code,
            error_message=payload.error_message,
        )
        await db.commit()
        await db.refresh(result)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "id": result.id,
        "run_id": result.run_id,
        "config_id": result.config_id,
        "artifact_id": result.artifact_id,
        "evaluation_kind": result.evaluation_kind,
        "status": result.status,
        "summary": payload.summary,
        "error_code": payload.error_code,
    }


@router.post("/runs/{run_id}/evaluate")
async def evaluate_ai_run(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        report = await evaluate_run(db, run_id)
        await db.commit()
        run = await db.get(AIOptimizationRun, run_id)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run": _run_payload(run), "report": report}


@router.post("/runs/{run_id}/finalize")
async def finalize_ai_run(
    run_id: int,
    payload: FinalizeRunRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Evaluate a run and optionally place its winner into passive SHADOW."""
    try:
        result = await finalize_run(
            db,
            run_id,
            auto_shadow=payload.auto_shadow,
            asset=payload.asset,
            regime=payload.regime,
            candidate_artifact_id=payload.candidate_artifact_id,
            baseline_artifact_id=payload.baseline_artifact_id,
        )
        await db.commit()
        run = await db.get(AIOptimizationRun, run_id)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    assignment = result.get("assignment")
    return {
        "run": _run_payload(run),
        "report": result["report"],
        "assignment": _assignment_payload(assignment) if assignment else None,
    }


@router.post("/runs/{run_id}/shadow")
async def promote_ai_shadow(
    run_id: int,
    payload: ShadowPromoteRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        assignment = await promote_to_shadow(
            db,
            run_id=run_id,
            candidate_artifact_id=payload.candidate_artifact_id,
            baseline_artifact_id=payload.baseline_artifact_id,
            asset=payload.asset,
            regime=payload.regime,
        )
        await db.commit()
        await db.refresh(assignment)
        run = await db.get(AIOptimizationRun, run_id)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run": _run_payload(run), "assignment": _assignment_payload(assignment)}


@router.post("/runs/{run_id}/transition")
async def transition_ai_run(
    run_id: int,
    payload: TransitionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = await db.get(AIOptimizationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="AI Lab run not found")
    target = payload.target.strip().upper()
    try:
        action = transition_action_for_target(target)
        if action:
            await authorize_run_action(db, run_id, action)
        await transition_run(db, run, target, reason=payload.reason)
        await db.commit()
        await db.refresh(run)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AIRunTransitionError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _run_payload(run)


@router.post("/runs/{run_id}/steps", status_code=201)
async def add_ai_step(
    run_id: int,
    payload: StepCreateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        step = await append_step(
            db,
            run_id,
            step_index=payload.step_index,
            step_type=payload.step_type,
            status=payload.status,
            hypothesis=payload.hypothesis,
            action=payload.action,
            input_payload=payload.input_payload,
            output_payload=payload.output_payload,
            summary=payload.summary,
            error_code=payload.error_code,
            error_message=payload.error_message,
        )
        await db.commit()
        await db.refresh(step)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _step_payload(step)


@router.post("/configs", status_code=201)
async def create_ai_config(
    payload: ConfigCreateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        row = await create_experiment_config(
            db,
            name=payload.name,
            model_family=payload.model_family,
            feature_set=payload.feature_set,
            feature_pipeline_version=payload.feature_pipeline_version,
            model_params=payload.model_params,
            strategy_params=payload.strategy_params,
            backtest_params=payload.backtest_params,
            asset=payload.asset,
            regime=payload.regime,
            description=payload.description,
            created_by=payload.created_by,
            parent_id=payload.parent_id,
        )
        await db.commit()
        await db.refresh(row)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "id": row.id,
        "name": row.name,
        "config_hash": row.config_hash,
        "feature_set": row.feature_set,
        "model_family": row.model_family,
        "created_at": row.created_at,
    }


@router.post("/runs/{run_id}/approval", status_code=201)
async def request_ai_approval(
    run_id: int,
    payload: ApprovalRequest,
    db: AsyncSession = Depends(get_db_session),
):
    if await db.get(AIOptimizationRun, run_id) is None:
        raise HTTPException(status_code=404, detail="AI Lab run not found")
    try:
        if not payload.diff and payload.requested_action.upper() == "ACTIVATE":
            row, _ = await propose_live_deployment(
                db,
                run_id=run_id,
                actor="operator",
            )
            await db.commit()
            await db.refresh(row)
        else:
            row = await request_approval(
                db,
                run_id=run_id,
                target_type=payload.target_type,
                target_id=payload.target_id or str(run_id),
                requested_action=payload.requested_action,
                diff=payload.diff,
            )
            await db.commit()
            await db.refresh(row)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "id": row.id,
        "run_id": row.run_id,
        "target_type": row.target_type,
        "target_id": row.target_id,
        "requested_action": row.requested_action,
        "status": row.status,
        "diff": row.diff,
        "requested_at": row.requested_at,
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
        restored = await rollback_deployment(
            db,
            target_revision_id=payload.target_revision_id,
            actor=payload.actor,
            reason=payload.reason,
        )
        await db.commit()
        await db.refresh(restored)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "ROLLED_BACK",
        "active_revision_id": restored.id,
        "revision_key": restored.revision_key,
        "revision_status": restored.status,
        "activated_at": restored.activated_at,
    }


@router.get("/deployments/revisions")
async def list_ai_deployment_revisions(
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
):
    revisions = (
        await db.execute(
            select(DeploymentRevision)
            .order_by(DeploymentRevision.id.desc())
            .limit(min(max(1, limit), 200))
        )
    ).scalars().all()
    return [
        {
            "id": rev.id,
            "revision_key": rev.revision_key,
            "parent_id": rev.parent_id,
            "manifest_hash": rev.manifest_hash,
            "status": rev.status,
            "created_by": rev.created_by,
            "created_at": rev.created_at,
            "activated_at": rev.activated_at,
            "rolled_back_at": rev.rolled_back_at,
        }
        for rev in revisions
    ]


@router.get("/deployments/revisions/{revision_id}")
async def get_ai_deployment_revision(
    revision_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    rev = await db.get(DeploymentRevision, revision_id)
    if rev is None:
        raise HTTPException(status_code=404, detail="Deployment revision not found")
    events = (
        await db.execute(
            select(DeploymentEvent)
            .where(DeploymentEvent.revision_id == revision_id)
            .order_by(DeploymentEvent.created_at, DeploymentEvent.id)
        )
    ).scalars().all()
    return {
        "id": rev.id,
        "revision_key": rev.revision_key,
        "parent_id": rev.parent_id,
        "manifest": rev.manifest,
        "manifest_hash": rev.manifest_hash,
        "status": rev.status,
        "created_by": rev.created_by,
        "created_at": rev.created_at,
        "activated_at": rev.activated_at,
        "rolled_back_at": rev.rolled_back_at,
        "events": [
            {
                "id": ev.id,
                "event_type": ev.event_type,
                "actor": ev.actor,
                "reason": ev.reason,
                "payload": ev.payload,
                "previous_hash": ev.previous_hash,
                "event_hash": ev.event_hash,
                "created_at": ev.created_at,
            }
            for ev in events
        ],
    }


# ---------------------------------------------------------------------------
# Phase 10: Autonomous Agent, Overlays, and Runtime Controls
# ---------------------------------------------------------------------------
@router.post("/runs/{run_id}/iterate", status_code=202)
async def trigger_agent_iteration(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Queue one autonomous iteration for the dedicated agent worker.

    LLM calls and training never run in the FastAPI request process.  The
    worker claims the queued run and persists progress in AIRunStep.
    """
    run = (
        await db.execute(
            select(AIOptimizationRun)
            .where(AIOptimizationRun.id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status in {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "REJECTED",
        "ROLLED_BACK",
        "ACTIVE",
        "SHADOW",
        "PENDING_APPROVAL",
    }:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot queue iteration from status {run.status}",
        )
    if run.status != "QUEUED":
        try:
            await transition_run(db, run, "QUEUED", reason="agent iteration queued")
            await db.commit()
        except (AILabError, AIRunTransitionError) as exc:
            await db.rollback()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    else:
        await db.rollback()
    return {"run_id": run_id, "status": "QUEUED"}


@router.post("/runs/{run_id}/pause")
async def pause_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Pause an active optimization run."""
    run = await db.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status in {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "REJECTED",
        "ROLLED_BACK",
        "ACTIVE",
        "SHADOW",
        "PENDING_APPROVAL",
        "PAUSED",
    }:
        raise HTTPException(status_code=409, detail=f"Cannot pause run in status {run.status}")
    try:
        await transition_run(db, run, "PAUSED", reason="paused by operator")
        await db.commit()
        await db.refresh(run)
    except (AILabError, AIRunTransitionError) as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _run_payload(run)


@router.post("/runs/{run_id}/resume")
async def resume_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Resume a paused optimization run."""
    run = await db.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "PAUSED":
        raise HTTPException(status_code=409, detail=f"Run is not paused (current: {run.status})")
    try:
        await transition_run(db, run, "RUNNING", reason="resumed by operator")
        await db.commit()
        await db.refresh(run)
    except (AILabError, AIRunTransitionError) as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _run_payload(run)


@router.post("/runs/{run_id}/cancel")
async def cancel_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Cancel an active or paused optimization run."""
    run = await db.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "REJECTED",
        "ROLLED_BACK",
        "ACTIVE",
    }:
        raise HTTPException(status_code=409, detail=f"Cannot cancel run in status {run.status}")
    try:
        await transition_run(db, run, "CANCELLED", reason="cancelled by operator")
        await db.commit()
        await db.refresh(run)
    except (AILabError, AIRunTransitionError) as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _run_payload(run)


@router.get("/runs/{run_id}/overlays")
async def list_run_overlays(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """List all runtime setting overlays associated with an optimization run."""
    await expire_overlays(db)
    await db.commit()
    stmt = (
        select(AIConfigOverlay)
        .where(AIConfigOverlay.run_id == run_id)
        .order_by(AIConfigOverlay.id.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "parent_overlay_id": r.parent_overlay_id,
            "scope": r.scope,
            "changes": r.changes,
            "status": r.status,
            "created_by": r.created_by,
            "expires_at": r.expires_at,
            "created_at": r.created_at,
        }
        for r in rows
    ]


@router.post("/overlays/{overlay_id}/rollback")
async def rollback_config_overlay_endpoint(
    overlay_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Roll back a specific active configuration overlay."""
    overlay = await db.get(AIConfigOverlay, overlay_id)
    if not overlay:
        raise HTTPException(status_code=404, detail="Overlay not found")

    try:
        await rollback_overlay(db, overlay_id)
        await db.commit()
        await db.refresh(overlay)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "ROLLED_BACK",
        "overlay_id": overlay.id,
        "run_id": overlay.run_id,
    }
