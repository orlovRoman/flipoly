"""Safe lifecycle services for the autonomous AI laboratory.

Phase 2 deliberately stops at experiment orchestration and audit persistence.
It never activates a model, changes RuntimeSettings, or submits an order.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.manifests import compute_manifest_hash
from polyflip.db.models import (
    AIApprovalRequest,
    AIExperimentConfig,
    AIOptimizationRun,
    AIPermission,
    AIRunStep,
    ExperimentResult,
)


class AILabError(ValueError):
    """Base error for rejected laboratory operations."""


class AIRunTransitionError(AILabError):
    """Raised when a run state transition is not allowed."""


class AIPermissionError(AILabError):
    """Raised when an action is outside the immutable permission snapshot."""


RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    "DRAFT": frozenset({"PLANNING", "CANCELLED"}),
    "PLANNING": frozenset({"RUNNING", "CANCELLED", "FAILED"}),
    "RUNNING": frozenset({"EVALUATING", "FAILED", "CANCELLED"}),
    "EVALUATING": frozenset(
        {"SHADOW", "PENDING_APPROVAL", "INSUFFICIENT_DATA", "FAILED"}
    ),
    "SHADOW": frozenset({"PENDING_APPROVAL", "REJECTED", "ROLLED_BACK"}),
    "PENDING_APPROVAL": frozenset({"ACTIVE", "REJECTED"}),
    "ACTIVE": frozenset({"ROLLED_BACK"}),
    "INSUFFICIENT_DATA": frozenset(),
    "FAILED": frozenset(),
    "REJECTED": frozenset(),
    "CANCELLED": frozenset(),
    "ROLLED_BACK": frozenset(),
}

LAB_ACTIONS = frozenset(
    {
        "CREATE_EXPERIMENT",
        "TRAIN_MODEL",
        "RUN_OOT_BACKTEST",
        "RUN_POLYMARKET_OOT",
        "PROMOTE_TO_SHADOW",
        "STOP_EXPERIMENT",
        "REQUEST_ACTIVATION",
        "REQUEST_ROLLBACK",
    }
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_run_transition(current: str, target: str) -> None:
    current = str(current).upper()
    target = str(target).upper()
    if target not in RUN_TRANSITIONS.get(current, frozenset()):
        raise AIRunTransitionError(
            f"invalid AI Lab run transition: {current} -> {target}"
        )


def validate_permission(permission: AIPermission | None, action: str) -> None:
    if permission is None:
        raise AIPermissionError("run has no permission snapshot")
    if not permission.enabled:
        raise AIPermissionError(f"permission profile {permission.profile_name!r} is disabled")
    action = str(action).upper()
    if action not in LAB_ACTIONS:
        raise AIPermissionError(f"unknown AI Lab action: {action}")
    allowed = {str(item).upper() for item in (permission.allowed_actions or [])}
    if action not in allowed:
        raise AIPermissionError(
            f"action {action} is not allowed by {permission.profile_name} v{permission.version}"
        )


async def create_run(
    session: AsyncSession,
    *,
    objective: str,
    scope: Mapping[str, Any],
    autonomy_level: str = "EXPERIMENT",
    budget_experiments: int = 0,
    budget_seconds: int = 0,
    created_by: str = "system",
    permission: AIPermission | None = None,
) -> AIOptimizationRun:
    if not objective or not objective.strip():
        raise AILabError("objective must not be empty")
    if autonomy_level not in {"OBSERVE", "EXPERIMENT", "SHADOW", "LIVE_PROPOSE"}:
        raise AILabError(f"unsupported autonomy level: {autonomy_level}")
    if budget_experiments < 0 or budget_seconds < 0:
        raise AILabError("budgets must be non-negative")
    if permission is not None:
        validate_permission(permission, "CREATE_EXPERIMENT")
    row = AIOptimizationRun(
        objective=objective.strip(),
        scope=dict(scope),
        autonomy_level=autonomy_level,
        budget_experiments=budget_experiments,
        budget_seconds=budget_seconds,
        created_by=created_by,
        permission_id=permission.id if permission is not None else None,
        created_at=utc_now(),
    )
    session.add(row)
    await session.flush()
    return row


async def transition_run(
    session: AsyncSession,
    run: AIOptimizationRun,
    target: str,
    *,
    reason: str | None = None,
) -> AIOptimizationRun:
    target = str(target).upper()
    validate_run_transition(run.status, target)
    now = utc_now()
    if target == "RUNNING":
        run.started_at = now
    if target in {"FAILED", "CANCELLED", "INSUFFICIENT_DATA", "REJECTED", "ROLLED_BACK"}:
        run.error = reason
    if target in {"ACTIVE", "SHADOW", "PENDING_APPROVAL"} and reason:
        run.summary = reason
    if target in {"ACTIVE", "ROLLED_BACK", "FAILED", "CANCELLED", "INSUFFICIENT_DATA", "REJECTED"}:
        run.finished_at = now
    run.status = target
    await session.flush()
    return run


async def append_step(
    session: AsyncSession,
    run_id: int,
    *,
    step_index: int,
    step_type: str,
    status: str = "SUCCEEDED",
    hypothesis: str | None = None,
    action: str | None = None,
    input_payload: Mapping[str, Any] | None = None,
    output_payload: Mapping[str, Any] | None = None,
    summary: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AIRunStep:
    if step_index < 0:
        raise AILabError("step_index must be non-negative")
    status = status.upper()
    if status not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED"}:
        raise AILabError(f"unsupported step status: {status}")
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    now = utc_now()
    row = AIRunStep(
        run_id=run_id,
        step_index=step_index,
        step_type=step_type,
        status=status,
        hypothesis=hypothesis,
        action=action,
        input_payload=dict(input_payload) if input_payload is not None else None,
        output_payload=dict(output_payload) if output_payload is not None else None,
        summary=summary,
        error_code=error_code,
        error_message=error_message,
        created_at=now,
        started_at=now if status in {"RUNNING", "SUCCEEDED", "FAILED"} else None,
        finished_at=now if status in {"SUCCEEDED", "FAILED", "SKIPPED"} else None,
    )
    session.add(row)
    await session.flush()
    return row


async def create_experiment_config(
    session: AsyncSession,
    *,
    name: str,
    model_family: str,
    feature_set: str,
    feature_pipeline_version: str,
    model_params: Mapping[str, Any],
    strategy_params: Mapping[str, Any],
    backtest_params: Mapping[str, Any],
    asset: str | None = None,
    regime: str | None = None,
    description: str | None = None,
    created_by: str = "system",
    parent_id: int | None = None,
) -> AIExperimentConfig:
    payload = {
        "name": name,
        "asset": asset,
        "regime": regime,
        "model_family": model_family,
        "feature_set": feature_set,
        "feature_pipeline_version": feature_pipeline_version,
        "model_params": dict(model_params),
        "strategy_params": dict(strategy_params),
        "backtest_params": dict(backtest_params),
        "parent_id": parent_id,
    }
    config_hash = compute_manifest_hash(payload)
    row = AIExperimentConfig(
        **payload,
        description=description,
        config_hash=config_hash,
        created_by=created_by,
        created_at=utc_now(),
    )
    session.add(row)
    await session.flush()
    return row


async def request_approval(
    session: AsyncSession,
    *,
    run_id: int | None,
    target_type: str,
    target_id: str,
    requested_action: str,
    diff: Mapping[str, Any],
) -> AIApprovalRequest:
    if requested_action.upper() not in {"ACTIVATE", "ROLLBACK", "CHANGE_LIVE_POLICY"}:
        raise AILabError(f"unsupported approval action: {requested_action}")
    row = AIApprovalRequest(
        run_id=run_id,
        target_type=target_type,
        target_id=target_id,
        requested_action=requested_action.upper(),
        diff=dict(diff),
        status="PENDING",
        requested_at=utc_now(),
    )
    session.add(row)
    await session.flush()
    return row


async def get_run_detail(session: AsyncSession, run_id: int) -> dict[str, Any] | None:
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        return None
    steps = (
        await session.execute(
            select(AIRunStep)
            .where(AIRunStep.run_id == run_id)
            .order_by(AIRunStep.step_index)
        )
    ).scalars().all()
    results = (
        await session.execute(
            select(ExperimentResult)
            .where(ExperimentResult.run_id == run_id)
            .order_by(ExperimentResult.created_at, ExperimentResult.id)
        )
    ).scalars().all()
    return {
        "run": run,
        "steps": list(steps),
        "results": list(results),
    }
