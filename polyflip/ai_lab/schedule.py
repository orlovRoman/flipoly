from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any, Mapping

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.service import create_run, transition_run
from polyflip.db.models import AIPermission, AIOptimizationRun


@dataclass(frozen=True)
class ScheduleDecision:
    allowed: bool
    reason: str
    run_id: int | None = None


def evaluate_schedule_start(
    *,
    enabled: bool,
    mode: str,
    trading_enabled: bool,
    active_runs: int,
    runs_today: int,
    max_concurrent_runs: int = 1,
    max_daily_runs: int = 1,
) -> ScheduleDecision:
    """Decide whether cron may create a run without touching execution state."""
    if not enabled:
        return ScheduleDecision(False, "schedule_disabled")
    if str(mode).upper() != "RESEARCH":
        return ScheduleDecision(False, "schedule_requires_research_mode")
    if trading_enabled:
        return ScheduleDecision(False, "research_schedule_blocked_while_trading_enabled")
    if active_runs >= max_concurrent_runs:
        return ScheduleDecision(False, "active_run_limit_reached")
    if runs_today >= max_daily_runs:
        return ScheduleDecision(False, "daily_run_limit_reached")
    return ScheduleDecision(True, "schedule_run_allowed")


async def create_scheduled_research_run(
    session: AsyncSession,
    *,
    enabled: bool,
    mode: str,
    trading_enabled: bool,
    objective: str,
    scope: Mapping[str, Any],
    budget_experiments: int,
    max_concurrent_runs: int = 1,
    max_daily_runs: int = 1,
) -> ScheduleDecision:
    """Create one idempotent PAPER/RESEARCH run for the durable agent worker.

    The scheduler only queues a run; it never claims jobs, trains a model, or
    touches deployment state. A permission snapshot with the research actions
    is required so the later worker cannot bypass the allowlist.
    """
    now = datetime.now(timezone.utc)
    day_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    active_statuses = (
        "DRAFT", "QUEUED", "PLANNING", "RUNNING", "EVALUATING", "PAUSED",
        "SHADOW", "PENDING_APPROVAL",
    )
    active_runs = int(
        (await session.execute(
            select(func.count(AIOptimizationRun.id)).where(
                AIOptimizationRun.status.in_(active_statuses)
            )
        )).scalar_one()
        or 0
    )
    runs_today = int(
        (await session.execute(
            select(func.count(AIOptimizationRun.id)).where(
                AIOptimizationRun.created_by == "ai_lab_scheduler",
                AIOptimizationRun.created_at >= day_start,
            )
        )).scalar_one()
        or 0
    )
    decision = evaluate_schedule_start(
        enabled=enabled,
        mode=mode,
        trading_enabled=trading_enabled,
        active_runs=active_runs,
        runs_today=runs_today,
        max_concurrent_runs=max_concurrent_runs,
        max_daily_runs=max_daily_runs,
    )
    if not decision.allowed:
        return decision

    permissions = (
        await session.execute(
            select(AIPermission)
            .where(
                AIPermission.enabled.is_(True),
                AIPermission.is_current.is_(True),
            )
            .order_by(AIPermission.id.desc())
        )
    ).scalars().all()
    required = {
        "CREATE_EXPERIMENT",
        "TRAIN_MODEL",
        "RUN_OOT_BACKTEST",
        "RUN_POLYMARKET_OOT",
        "PROMOTE_TO_SHADOW",
    }
    permission = next(
        (
            row for row in permissions
            if required.issubset({str(item).upper() for item in (row.allowed_actions or [])})
        ),
        None,
    )
    if permission is None:
        return ScheduleDecision(False, "research_permission_profile_missing")

    run = await create_run(
        session,
        objective=objective,
        scope=dict(scope),
        autonomy_level="AUTONOMOUS_SHADOW",
        budget_experiments=max(1, int(budget_experiments)),
        permission=permission,
        created_by="ai_lab_scheduler",
        mode="RESEARCH",
    )
    await transition_run(session, run, "QUEUED", reason="scheduled research run")
    await session.commit()
    return ScheduleDecision(True, "scheduled_research_run_created", run.id)
