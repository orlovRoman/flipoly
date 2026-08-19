"""Pure safety policy for optional recurring AI Lab research scheduling."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleDecision:
    allowed: bool
    reason: str


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
