"""HTTP API for the safe AI Lab experiment contour.

The router exposes only experiment, audit and approval operations. It does not
activate models or mutate live execution settings.
"""

from __future__ import annotations

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


class RunCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=500)
    scope: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: str = Field(
        default="EXPERIMENT",
        pattern=r"^(EXPERIMENT|AUTONOMOUS_SHADOW|DIRECTED)$",
    )
    budget_experiments: int = Field(default=10, ge=1, le=1000)
    created_by: str = Field(default="api", max_length=128)
    permission_profile: str = Field(
        default="experiment-only", min_length=1, max_length=64
    )
    agent_thread_id: str | None = Field(default=None, max_length=128)


class StepCreateRequest(BaseModel):
    step_index: int = Field(ge=0)
    step_type: str = Field(min_length=1, max_length=64)
    status: str = Field(default="SUCCEEDED", min_length=1, max_length=24)
    hypothesis: str | None = Field(default=None, max_length=4000)
    action: str | None = Field(default=None, max_length=64)
    input_payload: dict[str, Any] | None = None
    output_payload: dict[str, Any] | None = None
    summary: str | None = Field(default=None, max_length=4000)
    error_code: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=4000)


class ConfigCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    model_family: str = Field(min_length=1, max_length=32)
    feature_set: str = Field(min_length=1, max_length=8)
    feature_pipeline_version: str = Field(min_length=1, max_length=64)
    model_params: dict[str, Any]
    strategy_params: dict[str, Any]
    backtest_params: dict[str, Any]
    asset: str | None = Field(default=None, max_length=32)
    regime: str | None = Field(default=None, max_length=32)
    description: str | None = Field(default=None, max_length=4000)
    created_by: str = Field(default="api", max_length=128)
    parent_id: int | None = Field(default=None, gt=0)


class PermissionCreateRequest(BaseModel):
    profile_name: str = Field(min_length=1, max_length=64)
    allowed_actions: list[str] = Field(min_length=1)
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
    return {
        "id": run.id,
        "objective": run.objective,
        "scope": run.scope,
        "autonomy_level": run.autonomy_level,
        "status": run.status,
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
        "created_at": assignment.created_at,
    }


class TransitionRequest(BaseModel):
    target: str = Field(min_length=1, max_length=24)
    reason: str | None = Field(default=None, max_length=4000)


@router.post("/runs", status_code=201)
async def create_ai_run(
    payload: RunCreateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    permission = (
        await db.execute(
            select(AIPermission)
            .where(
                AIPermission.profile_name == payload.permission_profile,
                AIPermission.is_current.is_(True),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if permission is None:
        raise HTTPException(
            status_code=404,
            detail=f"permission profile {payload.permission_profile!r} not found",
        )
    try:
        run = await create_run(
            db,
            objective=payload.objective,
            scope=payload.scope,
            autonomy_level=payload.autonomy_level,
            budget_experiments=payload.budget_experiments,
            permission=permission,
            created_by=payload.created_by,
            agent_thread_id=payload.agent_thread_id,
        )
        await db.commit()
        await db.refresh(run)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _run_payload(run)


@router.get("/runs")
async def list_ai_runs(
    status: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
):
    stmt = select(AIOptimizationRun).order_by(AIOptimizationRun.id.desc())
    if status:
        stmt = stmt.where(AIOptimizationRun.status == status.upper())
    stmt = stmt.limit(min(max(1, limit), 200))
    runs = (await db.execute(stmt)).scalars().all()
    return [_run_payload(run) for run in runs]


@router.get("/runs/{run_id}")
async def get_ai_run(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
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
                "evaluation_kind": result.evaluation_kind,
                "status": result.status,
                "net_pnl": result.net_pnl,
                "trade_count": result.trade_count,
                "max_drawdown": result.max_drawdown,
                "metrics": result.metrics,
                "artifact_id": result.artifact_id,
                "created_at": result.created_at,
            }
            for result in detail["results"]
        ],
        "audits": [_audit_payload(audit) for audit in detail["audits"]],
    }


@router.post("/runs/{run_id}/plan", status_code=201)
async def plan_ai_run(
    run_id: int,
    payload: PlanRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        steps = await plan_run(db, run_id=run_id, config_ids=payload.config_ids)
        await db.commit()
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_step_payload(step) for step in steps]


@router.post("/runs/{run_id}/claim-step")
async def claim_ai_step(
    run_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        step = await claim_next_step(db, run_id)
        await db.commit()
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if step is None:
        return {"step": None}
    await db.refresh(step)
    return {"step": _step_payload(step)}


@router.post("/runs/{run_id}/worker-run")
async def run_ai_worker(
    run_id: int,
    payload: WorkerRunRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Execute up to max_steps planned LightGBM steps for one run."""
    run = await db.get(AIOptimizationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="AI Lab run not found")
    try:
        results = await execute_lgbm_steps(db, run_id=run_id, max_steps=payload.max_steps)
        await db.commit()
    except ExecutionBatchError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"error": "BATCH_EXECUTION_FAILED", "message": str(exc)},
        ) from exc
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "run_id": run_id,
        "processed_steps": len(results),
        "steps": results,
    }


@router.post("/scheduler/run")
async def run_ai_scheduler(
    payload: SchedulerRunRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Run the bounded scheduling loop across all active AI Lab runs."""
    try:
        summary = await run_lgbm_scheduler(
            db,
            max_iterations=payload.max_iterations,
            max_steps_per_iteration=payload.max_steps,
            interval_seconds=payload.interval_seconds,
            lease_ttl_seconds=payload.lease_ttl_seconds,
        )
    except ExecutionBatchError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"error": "SCHEDULER_EXECUTION_FAILED", "message": str(exc)},
        ) from exc
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return summary


@router.post("/runs/{run_id}/results", status_code=201)
async def add_ai_result(
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


@router.post("/approvals/{approval_id}/approve")
async def approve_ai_deployment(
    approval_id: int,
    payload: ApprovalDecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        revision = await approve_and_activate_deployment(
            db,
            approval_id=approval_id,
            actor=payload.actor,
            reason=payload.reason,
        )
        await db.commit()
        await db.refresh(revision)
        approval = await db.get(AIApprovalRequest, approval_id)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "APPROVED",
        "approval_id": approval_id,
        "revision_id": revision.id,
        "revision_key": revision.revision_key,
        "revision_status": revision.status,
        "activated_at": revision.activated_at,
        "decided_by": approval.decided_by if approval else payload.actor,
    }


@router.post("/approvals/{approval_id}/reject")
async def reject_ai_deployment(
    approval_id: int,
    payload: ApprovalDecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    try:
        approval = await reject_deployment_approval(
            db,
            approval_id=approval_id,
            actor=payload.actor,
            reason=payload.reason,
        )
        await db.commit()
        await db.refresh(approval)
    except AIPermissionError as exc:
        await db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "REJECTED",
        "approval_id": approval.id,
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
