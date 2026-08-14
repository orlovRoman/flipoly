"""HTTP API for the safe AI Lab experiment contour.

The router exposes only experiment, audit and approval operations. It does not
activate models or mutate live execution settings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.api.auth import verify_api_key
from polyflip.db.connection import get_db_session
from polyflip.db.models import (
    AIApprovalRequest,
    AIExperimentConfig,
    AIModelArtifact,
    AIOptimizationRun,
    AIPermission,
    AIRunStep,
    AIStepAuditLog,
    DeploymentEvent,
    DeploymentRevision,
    ExperimentResult,
)
from polyflip.ai_lab.service import (
    approve_and_activate_deployment,
    propose_live_deployment,
    record_deployment_event,
    reject_deployment_approval,
    rollback_deployment,
    transition_run,
)

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/ai-lab",
    tags=["AI Lab"],
    dependencies=[Depends(verify_api_key)],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    objective: str = Field(..., min_length=3, max_length=4000)
    scope: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: Literal[
        "OBSERVE",
        "EXPERIMENT",
        "SHADOW",
        "LIVE_PROPOSE",
        "AUTONOMOUS_SHADOW",
        "DIRECTED",
    ] = "EXPERIMENT"
    budget_experiments: int = Field(default=10, ge=1, le=1000)
    created_by: str = Field(default="system", max_length=128)
    permission_id: int | None = None
    agent_thread_id: str | None = Field(default=None, max_length=128)
    agent_type: str | None = Field(default=None, max_length=64)


class RunResponse(BaseModel):
    id: int
    objective: str
    scope: dict[str, Any]
    autonomy_level: str
    status: str
    agent_thread_id: str | None
    agent_type: str | None
    budget_experiments: int
    experiments_completed: int
    created_by: str
    summary: str | None
    error: str | None
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class StepResponse(BaseModel):
    id: int
    run_id: int
    step_index: int
    step_type: str
    status: str
    hypothesis: str | None
    action: str | None
    summary: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class ResultResponse(BaseModel):
    id: int
    run_id: int
    config_id: int
    evaluation_kind: str
    status: str
    trade_count: int | None
    net_pnl: float | None
    max_drawdown: float | None
    metrics: dict[str, Any] | None
    created_at: datetime | None

    model_config = {"from_attributes": True}


class RunDetailResponse(BaseModel):
    run: RunResponse
    steps: list[StepResponse]
    results: list[ResultResponse]


class ApprovalRequest(BaseModel):
    requested_action: Literal["ACTIVATE", "EXPAND_BUDGET", "SHUTDOWN_CIRCUIT"]
    target_type: str = "run"
    target_id: str | None = None
    actor: str = Field(default="system", max_length=128)
    reason: str | None = None


class ApprovalDecision(BaseModel):
    actor: str = Field(default="admin", max_length=128)
    reason: str | None = None


class RollbackRequest(BaseModel):
    target_revision_id: int | None = None
    actor: str = Field(default="admin", max_length=128)
    reason: str | None = None


class PermissionResponse(BaseModel):
    id: int
    profile_name: str
    version: int
    is_current: bool
    allowed_actions: list[str]
    scope: dict[str, Any]
    limits: dict[str, Any]
    enabled: bool

    model_config = {"from_attributes": True}


class RevisionResponse(BaseModel):
    id: int
    revision_key: str
    parent_id: int | None
    manifest: dict[str, Any]
    manifest_hash: str
    status: str
    created_by: str
    created_at: datetime
    activated_at: datetime | None
    rolled_back_at: datetime | None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Route Handlers
# ---------------------------------------------------------------------------


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an AI optimization run",
)
async def create_run(
    payload: CreateRunRequest,
    session: AsyncSession = Depends(get_db_session),
) -> AIOptimizationRun:
    """Register a new offline experiment run in the safe contour."""
    if payload.permission_id is not None:
        perm = await session.get(AIPermission, payload.permission_id)
        if not perm or not perm.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Permission profile {payload.permission_id} is invalid or disabled.",
            )

    run = AIOptimizationRun(
        objective=payload.objective,
        scope=payload.scope,
        autonomy_level=payload.autonomy_level,
        status="DRAFT",
        agent_thread_id=payload.agent_thread_id,
        agent_type=payload.agent_type,
        permission_id=payload.permission_id,
        experiment_budget=payload.budget_experiments,
        created_by=payload.created_by,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    logger.info(
        "ai_lab.run_created",
        run_id=run.id,
        objective=run.objective[:80],
        autonomy_level=run.autonomy_level,
        budget=run.experiment_budget,
    )
    return run


@router.get(
    "/runs",
    summary="List AI optimization runs",
)
async def list_runs(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Retrieve runs ordered by creation time descending."""
    stmt = select(AIOptimizationRun).order_by(desc(AIOptimizationRun.id))
    if status_filter:
        stmt = stmt.where(AIOptimizationRun.status == status_filter.upper())
    stmt = stmt.limit(limit).offset(offset)

    result = await session.execute(stmt)
    runs = result.scalars().all()

    return {
        "total": len(runs),
        "limit": limit,
        "offset": offset,
        "runs": [
            {
                "id": r.id,
                "objective": r.objective,
                "scope": r.scope,
                "autonomy_level": r.autonomy_level,
                "status": r.status,
                "agent_thread_id": r.agent_thread_id,
                "agent_type": r.agent_type,
                "budget_experiments": r.experiment_budget,
                "experiments_completed": r.experiments_completed,
                "created_by": r.created_by,
                "summary": r.summary,
                "error": r.error,
                "created_at": r.created_at,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
            }
            for r in runs
        ],
    }


@router.get(
    "/runs/{run_id}",
    summary="Get run details including steps and results",
)
async def get_run_detail(
    run_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Retrieve a run with all child steps and experiment results."""
    run = await session.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found.",
        )

    steps_res = await session.execute(
        select(AIRunStep)
        .where(AIRunStep.run_id == run_id)
        .order_by(AIRunStep.step_index)
    )
    steps = steps_res.scalars().all()

    results_res = await session.execute(
        select(ExperimentResult)
        .where(ExperimentResult.run_id == run_id)
        .order_by(desc(ExperimentResult.id))
    )
    results = results_res.scalars().all()

    return {
        "run": {
            "id": run.id,
            "objective": run.objective,
            "scope": run.scope,
            "autonomy_level": run.autonomy_level,
            "status": run.status,
            "agent_thread_id": run.agent_thread_id,
            "agent_type": run.agent_type,
            "budget_experiments": run.experiment_budget,
            "experiments_completed": run.experiments_completed,
            "created_by": run.created_by,
            "summary": run.summary,
            "error": run.error,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        },
        "steps": [
            {
                "id": s.id,
                "run_id": s.run_id,
                "step_index": s.step_index,
                "step_type": s.step_type,
                "status": s.status,
                "hypothesis": s.hypothesis,
                "action": s.action,
                "summary": s.summary,
                "error_code": s.error_code,
                "error_message": s.error_message,
                "created_at": s.created_at,
                "started_at": s.started_at,
                "finished_at": s.finished_at,
            }
            for s in steps
        ],
        "results": [
            {
                "id": r.id,
                "run_id": r.run_id,
                "config_id": r.config_id,
                "evaluation_kind": r.evaluation_kind,
                "status": r.status,
                "trade_count": r.trade_count,
                "net_pnl": r.net_pnl,
                "max_drawdown": r.max_drawdown,
                "metrics": r.metrics,
                "created_at": r.created_at,
            }
            for r in results
        ],
    }


@router.post(
    "/runs/{run_id}/approval",
    status_code=status.HTTP_201_CREATED,
    summary="Request human approval for a run action",
)
async def request_approval(
    run_id: int,
    payload: ApprovalRequest,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Create a human approval gate request for a sensitive action."""
    run = await session.get(AIOptimizationRun, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found.",
        )

    if payload.requested_action == "ACTIVATE":
        if run.status not in ("SHADOW", "PENDING_APPROVAL"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Run {run_id} is in status '{run.status}'. "
                "Only runs in 'SHADOW' or 'PENDING_APPROVAL' status may be proposed for LIVE deployment.",
            )
        try:
            row, _ = await propose_live_deployment(
                session,
                run,
                actor=payload.actor,
                reason=payload.reason,
            )
            return {
                "id": row.id,
                "run_id": row.run_id,
                "target_type": row.target_type,
                "target_id": row.target_id,
                "requested_action": row.requested_action,
                "diff": row.diff,
                "status": row.status,
                "requested_at": row.requested_at,
            }
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    approval = AIApprovalRequest(
        run_id=run_id,
        target_type=payload.target_type,
        target_id=payload.target_id or str(run_id),
        requested_action=payload.requested_action,
        diff={},
        status="PENDING",
    )
    session.add(approval)
    await session.commit()
    await session.refresh(approval)

    logger.info(
        "ai_lab.approval_requested",
        approval_id=approval.id,
        run_id=run_id,
        action=approval.requested_action,
    )

    return {
        "id": approval.id,
        "run_id": approval.run_id,
        "target_type": approval.target_type,
        "target_id": approval.target_id,
        "requested_action": approval.requested_action,
        "diff": approval.diff,
        "status": approval.status,
        "requested_at": approval.requested_at,
    }


@router.post(
    "/approvals/{approval_id}/approve",
    summary="Approve and activate a proposed deployment",
)
async def approve_approval(
    approval_id: int,
    payload: ApprovalDecision,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Approve a pending request and transactionally activate the deployment revision."""
    try:
        approval, revision = await approve_and_activate_deployment(
            session,
            approval_id=approval_id,
            actor=payload.actor,
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "status": "APPROVED",
        "approval_id": approval.id,
        "revision_id": revision.id,
        "revision_key": revision.revision_key,
        "manifest_hash": revision.manifest_hash,
        "activated_at": revision.activated_at,
    }


@router.post(
    "/approvals/{approval_id}/reject",
    summary="Reject a proposed deployment",
)
async def reject_approval(
    approval_id: int,
    payload: ApprovalDecision,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Reject a pending request and mark the associated revision as REJECTED."""
    try:
        approval, revision = await reject_deployment_approval(
            session,
            approval_id=approval_id,
            actor=payload.actor,
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "status": "REJECTED",
        "approval_id": approval.id,
        "revision_id": revision.id if revision else None,
        "decided_by": approval.decided_by,
        "decision_reason": approval.decision_reason,
    }


@router.post(
    "/deployments/rollback",
    summary="Rollback active deployment revision to parent",
)
async def rollback_deployment_route(
    payload: RollbackRequest,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Emergency rollback: switches active ModelRegistry pointers without touching positions."""
    try:
        rolled_back_rev, active_rev = await rollback_deployment(
            session,
            target_revision_id=payload.target_revision_id,
            actor=payload.actor,
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "status": "ROLLED_BACK",
        "rolled_back_revision_id": rolled_back_rev.id,
        "active_revision_id": active_rev.id,
        "active_revision_key": active_rev.revision_key,
        "active_manifest_hash": active_rev.manifest_hash,
    }


@router.get(
    "/permissions",
    summary="List active AI permission profiles",
)
async def list_permissions(
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """List permission profiles for autonomous AI execution."""
    result = await session.execute(
        select(AIPermission).where(AIPermission.is_current.is_(True))
    )
    perms = result.scalars().all()
    return {
        "permissions": [
            {
                "id": p.id,
                "profile_name": p.profile_name,
                "version": p.version,
                "is_current": p.is_current,
                "allowed_actions": p.allowed_actions,
                "scope": p.scope,
                "limits": p.limits,
                "enabled": p.enabled,
            }
            for p in perms
        ]
    }


@router.get(
    "/deployments/revisions",
    summary="List deployment revisions",
)
async def list_revisions(
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """List recorded deployment revisions ordered by creation time descending."""
    result = await session.execute(
        select(DeploymentRevision)
        .order_by(desc(DeploymentRevision.id))
        .limit(limit)
    )
    revisions = result.scalars().all()
    return [
        {
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
        }
        for rev in revisions
    ]


@router.get(
    "/deployments/revisions/{revision_id}",
    summary="Get a deployment revision and its audit events",
)
async def get_revision_detail(
    revision_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Retrieve revision manifest and its hash chain events."""
    rev = await session.get(DeploymentRevision, revision_id)
    if not rev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Revision {revision_id} not found.",
        )

    events_res = await session.execute(
        select(DeploymentEvent)
        .where(DeploymentEvent.revision_id == revision_id)
        .order_by(DeploymentEvent.id.asc())
    )
    events = events_res.scalars().all()

    return {
        "revision": {
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
        },
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
