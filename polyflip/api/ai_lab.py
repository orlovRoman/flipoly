"""HTTP API for the safe AI Lab experiment contour.

The router exposes only experiment, audit and approval operations. It does not
activate models or mutate live execution settings.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.service import (
    AILabError,
    AIPermissionError,
    AIRunTransitionError,
    append_step,
    create_experiment_config,
    create_permission,
    create_run,
    authorize_run_action,
    get_run_detail,
    request_approval,
    transition_run,
)
from polyflip.api.auth import verify_api_key
from polyflip.db.connection import get_db_session
from polyflip.db.models import (
    AIOptimizationRun,
    AIPermission,
    AIRunStep,
    AIExperimentConfig,
)

router = APIRouter(
    prefix="/api/ai-lab",
    tags=["ai-lab"],
    dependencies=[Depends(verify_api_key)],
)


class RunCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=4000)
    scope: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: str = "EXPERIMENT"
    budget_experiments: int = Field(default=0, ge=0, le=10000)
    budget_seconds: int = Field(default=0, ge=0, le=7 * 24 * 3600)
    created_by: str = Field(default="api", max_length=128)
    permission_id: int | None = None


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
    target_type: str = Field(min_length=1, max_length=32)
    target_id: str = Field(min_length=1, max_length=64)
    requested_action: str = Field(min_length=1, max_length=32)
    diff: dict[str, Any] = Field(default_factory=dict)


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
    permission = (
        await db.get(AIPermission, payload.permission_id)
        if payload.permission_id is not None
        else None
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
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
):
    limit = min(max(limit, 1), 200)
    query = select(AIOptimizationRun).order_by(
        AIOptimizationRun.created_at.desc(), AIOptimizationRun.id.desc()
    ).limit(limit)
    if status:
        query = query.where(AIOptimizationRun.status == status.upper())
    rows = (await db.execute(query)).scalars().all()
    return {"runs": [_run_payload(row) for row in rows]}


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
                "created_at": result.created_at,
            }
            for result in detail["results"]
        ],
    }


@router.post("/runs/{run_id}/transition")
async def transition_ai_run(
    run_id: int,
    payload: TransitionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = await db.get(AIOptimizationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="AI Lab run not found")
    try:
        await transition_run(db, run, payload.target, reason=payload.reason)
        await db.commit()
        await db.refresh(run)
    except (AILabError, AIRunTransitionError) as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        row = await request_approval(
            db,
            run_id=run_id,
            target_type=payload.target_type,
            target_id=payload.target_id,
            requested_action=payload.requested_action,
            diff=payload.diff,
        )
        await db.commit()
        await db.refresh(row)
    except AILabError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "id": row.id,
        "run_id": row.run_id,
        "requested_action": row.requested_action,
        "status": row.status,
        "requested_at": row.requested_at,
    }
