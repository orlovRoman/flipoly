"""Safe executor for autonomous AI Lab experiment steps.

This module is deliberately an adapter boundary.  It can execute only the
three offline actions planned by the AI Lab orchestrator.  A caller must
explicitly register adapters; there are no built-in imports of trainers,
backtest runners, Polymarket gateways, RuntimeSettings or live execution.

The database transaction that claims a step is committed before an adapter is
called.  Long-running training/backtests therefore never hold a row lock.
Every adapter outcome is persisted through the same audited result path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Mapping, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.orchestrator import (
    claim_next_step,
    record_result,
)
from polyflip.ai_lab.jobs import claim_job, complete_job, ensure_job
from polyflip.ai_lab.service import AILabError, utc_now
from polyflip.db.models import (
    AIExperimentConfig,
    AIOptimizationRun,
    AIRunStep,
    AIStepAuditLog,
)

ACTION_TO_EVALUATION_KIND: dict[str, str] = {
    "TRAIN_MODEL": "TRAIN",
    "RUN_OOT_BACKTEST": "OOT",
    "RUN_POLYMARKET_OOT": "POLYMARKET_OOT",
}
OFFLINE_ACTIONS = frozenset(ACTION_TO_EVALUATION_KIND)
FORBIDDEN_ACTIONS = frozenset(
    {
        "ACTIVATE_MODEL",
        "CHANGE_LIVE_POLICY",
        "EXECUTE_LIVE",
        "PLACE_ORDER",
        "LIVE",
    }
)
MAX_ERROR_MESSAGE_LENGTH = 4000


class AdapterCallable(Protocol):
    def __call__(
        self, context: "StepContext"
    ) -> "AdapterResult | Awaitable[AdapterResult]":
        ...


@dataclass(frozen=True)
class StepContext:
    """Immutable, serializable inputs handed to one offline adapter."""

    run_id: int
    step_id: int
    action: str
    config_id: int
    config_hash: str
    objective: str
    scope: Mapping[str, Any]
    input_payload: Mapping[str, Any]
    # Immutable AIExperimentConfig snapshot passed to offline adapters.
    model_family: str = ""
    feature_set: str = "A"
    asset: str | None = None
    regime: str | None = None
    interval: str = "15m"
    model_params: Mapping[str, Any] = field(default_factory=dict)
    calibration_params: Mapping[str, Any] = field(default_factory=dict)
    strategy_params: Mapping[str, Any] = field(default_factory=dict)
    backtest_params: Mapping[str, Any] = field(default_factory=dict)


class ExecutionBatchError(RuntimeError):
    """Raised after a bounded batch fails while preserving completed outcomes."""

    def __init__(
        self,
        cause: Exception,
        completed: list["ExecutionOutcome"],
    ) -> None:
        self.cause = cause
        self.completed = tuple(completed)
        super().__init__(
            f"AI Lab executor batch stopped after {len(self.completed)} "
            f"completed step(s): {cause}"
        )


@dataclass(frozen=True)
class AdapterResult:
    """Structured output accepted from a registered adapter."""

    evaluation_kind: str
    status: str = "SUCCEEDED"
    metrics: Mapping[str, Any] = field(default_factory=dict)
    slices: Mapping[str, Any] = field(default_factory=dict)
    trade_count: int | None = None
    net_pnl: float | None = None
    max_drawdown: float | None = None
    artifact_id: int | None = None
    code_sha: str | None = None
    dataset_fingerprint: str | None = None
    train_window_start: Any | None = None
    train_window_end: Any | None = None
    oot_window_start: Any | None = None
    oot_window_end: Any | None = None
    summary: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    def validate_for(self, action: str) -> "AdapterResult":
        normalized_action = action.strip().upper()
        expected_kind = ACTION_TO_EVALUATION_KIND.get(normalized_action)
        if expected_kind is None:
            raise AILabError(f"unsupported executor action: {normalized_action}")
        kind = self.evaluation_kind.strip().upper()
        if kind != expected_kind:
            raise AILabError(
                f"adapter returned {kind} for {normalized_action}; "
                f"expected {expected_kind}"
            )
        status = self.status.strip().upper()
        if status not in {"SUCCEEDED", "FAILED", "INSUFFICIENT_DATA"}:
            raise AILabError(f"unsupported adapter result status: {status}")
        if self.trade_count is not None and self.trade_count < 0:
            raise AILabError("adapter trade_count must be non-negative")
        return self


@dataclass(frozen=True)
class ExecutionOutcome:
    """Small result returned to a worker after the DB commit."""

    run_id: int
    step_id: int
    action: str
    evaluation_kind: str | None
    status: str
    result_id: int | None
    error_code: str | None = None
    config_id: int | None = None


class AdapterRegistry:
    """Explicit allow-list of offline adapters.

    Registering a live or unknown action is rejected instead of silently
    accepting an adapter that could bypass the safety boundary.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, AdapterCallable] = {}

    def register(
        self,
        action: str,
        adapter: AdapterCallable,
        *,
        replace: bool = False,
    ) -> "AdapterRegistry":
        normalized = action.strip().upper()
        if normalized in FORBIDDEN_ACTIONS or normalized not in OFFLINE_ACTIONS:
            raise AILabError(
                f"only offline AI Lab actions can be registered: {normalized}"
            )
        if not callable(adapter):
            raise TypeError("adapter must be callable")
        if normalized in self._adapters and not replace:
            raise AILabError(
                f"adapter for {normalized} is already registered; "
                "unregister it or pass replace=True explicitly"
            )
        self._adapters[normalized] = adapter
        return self

    def unregister(self, action: str) -> AdapterCallable:
        normalized = action.strip().upper()
        try:
            return self._adapters.pop(normalized)
        except KeyError as exc:
            raise AILabError(f"no adapter is registered for {normalized}") from exc

    def get(self, action: str | None) -> AdapterCallable | None:
        if not action:
            return None
        return self._adapters.get(action.strip().upper())

    def actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def _short_error(value: Any) -> str:
    message = str(value).strip() or value.__class__.__name__
    return message[:MAX_ERROR_MESSAGE_LENGTH]


def _close_step(
    step: AIRunStep,
    *,
    summary: str,
    error_code: str,
    error_message: str,
) -> None:
    step.status = "FAILED"
    step.finished_at = utc_now()
    step.summary = summary[:MAX_ERROR_MESSAGE_LENGTH]
    step.error_code = error_code[:64]
    step.error_message = error_message[:MAX_ERROR_MESSAGE_LENGTH]


async def _run_adapter(
    adapter: AdapterCallable,
    context: StepContext,
) -> AdapterResult:
    value = adapter(context)
    if inspect.isawaitable(value):
        value = await value
    if not isinstance(value, AdapterResult):
        raise AILabError("registered adapter must return AdapterResult")
    return value.validate_for(context.action)


async def execute_next_step(
    session: AsyncSession,
    run_id: int,
    registry: AdapterRegistry,
) -> ExecutionOutcome | None:
    """Claim and execute one offline step.

    The claim transaction is committed before adapter execution.  Missing
    adapters are recorded as a deterministic FAILED result, allowing the
    worker to keep draining the queue without crashing or touching LIVE.
    """

    step = await claim_next_step(session, run_id)
    if step is None:
        await session.rollback()
        return None

    action = (step.action or "").strip().upper()
    payload = step.input_payload if isinstance(step.input_payload, Mapping) else {}
    raw_config_id = payload.get("config_id")
    try:
        config_id = int(raw_config_id) if raw_config_id is not None else None
    except (TypeError, ValueError):
        config_id = None
    if config_id is not None and config_id <= 0:
        config_id = None

    run = await session.get(AIOptimizationRun, run_id)
    config = (
        await session.get(AIExperimentConfig, config_id)
        if config_id is not None
        else None
    )
    if (
        run is None
        or config is None
        or not action
        or action not in ACTION_TO_EVALUATION_KIND
    ):
        error_code = "INVALID_STEP_INPUT"
        message = "claimed step has no valid run, config_id or offline action"
        _close_step(
            step,
            summary="Step could not be executed because its inputs are invalid.",
            error_code=error_code,
            error_message=message,
        )
        session.add(
            AIStepAuditLog(
                run_id=run_id,
                step_id=step.id,
                config_id=config_id,
                action=action or None,
                error_code=error_code,
                error_message=message,
                payload={"raw_config_id": raw_config_id},
                created_at=utc_now(),
            )
        )
        await session.commit()
        return ExecutionOutcome(
            run_id=run_id,
            step_id=step.id,
            action=action,
            evaluation_kind=ACTION_TO_EVALUATION_KIND.get(action),
            status="FAILED",
            result_id=None,
            error_code=error_code,
            config_id=config_id,
        )

    context = StepContext(
        run_id=run_id,
        step_id=step.id,
        action=action,
        config_id=config_id,
        config_hash=str(config.config_hash),
        objective=str(run.objective),
        scope=dict(run.scope or {}),
        input_payload=dict(payload),
        model_family=str(getattr(config, "model_family", "") or ""),
        feature_set=str(getattr(config, "feature_set", "A") or "A"),
        asset=getattr(config, "asset", None),
        regime=getattr(config, "regime", None),
        interval=str(payload.get("interval") or "15m"),
        model_params=dict(getattr(config, "model_params", None) or {}),
        calibration_params=dict(
            getattr(config, "calibration_params", None)
            or payload.get("calibration", {})
            or {}
        ),
        strategy_params=dict(getattr(config, "strategy_params", None) or {}),
        backtest_params=dict(getattr(config, "backtest_params", None) or {}),
    )
    # Create and claim a durable job for this exact step before executing the
    # potentially long-running adapter. The run-level AIWorkerLease remains
    # the owner lease; this job record makes restart/recovery auditable.
    job_key = f"ai-lab:{run_id}:{step.id}:{action}"
    job = await ensure_job(
        session,
        run_id=run_id,
        step_id=step.id,
        operation=action,
        idempotency_key=job_key,
    )
    if job.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        await session.rollback()
        return None
    if await claim_job(session, job_key) is None:
        await session.rollback()
        return None

    # Do not hold the claim row lock during model training/backtesting.
    await session.commit()

    adapter = registry.get(action)
    result: AdapterResult
    if adapter is None:
        result = AdapterResult(
            evaluation_kind=ACTION_TO_EVALUATION_KIND[action],
            status="FAILED",
            summary=f"No adapter is registered for {action}.",
            error_code="ADAPTER_NOT_REGISTERED",
            error_message=(
                "The step was claimed safely, but the worker registry has no "
                f"adapter for {action}."
            ),
        )
    else:
        try:
            result = await _run_adapter(adapter, context)
        except Exception as exc:  # adapter failures become auditable results
            result = AdapterResult(
                evaluation_kind=ACTION_TO_EVALUATION_KIND[action],
                status="FAILED",
                summary=f"{action} adapter failed.",
                error_code="ADAPTER_EXECUTION_FAILED",
                error_message=_short_error(exc),
            )

    try:
        persisted = await record_result(
            session,
            run_id=run_id,
            config_id=config_id,
            evaluation_kind=result.evaluation_kind,
            status=result.status,
            metrics=result.metrics,
            slices=result.slices,
            trade_count=result.trade_count,
            net_pnl=result.net_pnl,
            max_drawdown=result.max_drawdown,
            artifact_id=result.artifact_id,
            step_id=step.id,
            code_sha=result.code_sha,
            dataset_fingerprint=result.dataset_fingerprint,
            train_window_start=result.train_window_start,
            train_window_end=result.train_window_end,
            oot_window_start=result.oot_window_start,
            oot_window_end=result.oot_window_end,
            summary=result.summary,
            error_code=result.error_code,
            error_message=result.error_message,
        )
        await complete_job(
            session,
            job.id,
            status="SUCCEEDED" if result.status == "SUCCEEDED" else "FAILED",
            error=result.error_message,
        )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        try:
            async with session.begin():
                await complete_job(
                    session,
                    job.id,
                    status="FAILED",
                    error=_short_error(exc),
                )
        except Exception:
            await session.rollback()
        raise

    return ExecutionOutcome(
        run_id=run_id,
        step_id=step.id,
        action=action,
        evaluation_kind=result.evaluation_kind.strip().upper(),
        status=result.status.strip().upper(),
        result_id=persisted.id,
        error_code=result.error_code,
        config_id=config_id,
    )


async def execute_steps(
    session: AsyncSession,
    run_id: int,
    registry: AdapterRegistry,
    *,
    max_steps: int = 1,
) -> list[ExecutionOutcome]:
    """Drain at most max_steps; the bound prevents an unbounded worker loop."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    outcomes: list[ExecutionOutcome] = []
    for _ in range(max_steps):
        try:
            outcome = await execute_next_step(session, run_id, registry)
        except Exception as exc:
            # execute_next_step already rolls back its failed result write.
            # Preserve earlier committed outcomes for the worker/agent instead
            # of losing them behind a bare exception.
            raise ExecutionBatchError(exc, outcomes) from exc
        if outcome is None:
            break
        outcomes.append(outcome)
    return outcomes
