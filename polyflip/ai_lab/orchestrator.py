"""Safe experiment orchestration for AI Lab phase 3.

The orchestrator creates and evaluates reproducible experiment work, but never
changes active models, RuntimeSettings, live execution policy or open orders.
Workers/agents claim pending steps and submit their results through this module.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.service import (
    AILabError,
    authorize_run_action,
    transition_run,
    utc_now,
)
from polyflip.db.models import (
    AIExperimentConfig,
    AIModelArtifact,
    AIOptimizationRun,
    AIShadowAssignment,
    AIRunStep,
    ExperimentResult,
)

PLAN_ACTIONS = (
    ("TRAIN_MODEL", "TRAIN_MODEL"),
    ("RUN_OOT_BACKTEST", "RUN_OOT_BACKTEST"),
    ("RUN_POLYMARKET_OOT", "RUN_POLYMARKET_OOT"),
)

RESULT_ACTIONS = {
    "TRAIN": "TRAIN_MODEL",
    "OOT": "RUN_OOT_BACKTEST",
    "POLYMARKET_OOT": "RUN_POLYMARKET_OOT",
}


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _median(values: Sequence[Any]) -> float | None:
    numbers = sorted(
        value for value in (_finite(item) for item in values) if value is not None
    )
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return float(numbers[middle])
    return float((numbers[middle - 1] + numbers[middle]) / 2)


def default_plan_steps(config_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Return the deterministic three-stage plan for each candidate config."""
    normalized = [int(config_id) for config_id in config_ids]
    return [
        {
            "step_index": index,
            "step_type": step_type,
            "action": action,
            "config_id": config_id,
        }
        for config_id in normalized
        for index, (step_type, action) in enumerate(PLAN_ACTIONS)
    ]


def build_experiment_report(
    results: Sequence[ExperimentResult | Mapping[str, Any]],
    *,
    min_trades: int = 3,
) -> dict[str, Any]:
    """Build an advisory report from persisted OOT/Polymarket-OOT results.

    Selection is based on median net PnL and drawdown only when a candidate has
    enough trades. AUC/ECE are reported as diagnostics and never substitute
    for missing PnL evidence.
    """
    if min_trades < 0:
        raise AILabError("min_trades must be non-negative")

    grouped: dict[int, list[Any]] = defaultdict(list)
    for result in results:
        kind = str(_value(result, "evaluation_kind", "")).upper()
        if kind in {"OOT", "POLYMARKET_OOT"}:
            grouped[int(_value(result, "config_id"))].append(result)

    rows: list[dict[str, Any]] = []
    for config_id, candidate_results in sorted(grouped.items()):
        metric_values: dict[str, list[Any]] = defaultdict(list)
        pnl_values: list[Any] = []
        trade_values: list[Any] = []
        drawdown_values: list[Any] = []
        artifact_ids: set[int] = set()
        for result in candidate_results:
            metrics = _value(result, "metrics", {}) or {}
            for metric_name in ("auc", "ece", "brier", "log_loss", "win_rate"):
                metric_values[metric_name].append(metrics.get(metric_name))
            pnl_values.append(_value(result, "net_pnl", metrics.get("net_pnl")))
            trade_values.append(
                _value(result, "trade_count", metrics.get("n_trades"))
            )
            drawdown_values.append(
                _value(result, "max_drawdown", metrics.get("max_drawdown"))
            )
            artifact_id = _value(result, "artifact_id")
            if artifact_id is not None:
                artifact_ids.add(int(artifact_id))

        median_pnl = _median(pnl_values)
        median_trades = _median(trade_values)
        median_drawdown = _median(drawdown_values)
        row = {
            "config_id": config_id,
            "evaluation_count": len(candidate_results),
            "artifact_ids": sorted(artifact_ids),
            "median_oot_pnl": median_pnl,
            "median_oot_trades": int(round(median_trades))
            if median_trades is not None
            else 0,
            "median_oot_drawdown": median_drawdown,
            "auc": _median(metric_values["auc"]),
            "ece": _median(metric_values["ece"]),
            "brier": _median(metric_values["brier"]),
            "log_loss": _median(metric_values["log_loss"]),
            "win_rate": _median(metric_values["win_rate"]),
        }
        row["eligible_for_shadow"] = (
            median_pnl is not None
            and row["median_oot_trades"] >= min_trades
        )
        rows.append(row)

    eligible = [row for row in rows if row["eligible_for_shadow"]]
    winner = (
        max(
            eligible,
            key=lambda row: (
                row["median_oot_pnl"],
                -(abs(row["median_oot_drawdown"] or 0.0)),
                row["median_oot_trades"],
            ),
        )
        if eligible
        else None
    )
    if winner:
        status = "READY_FOR_SHADOW"
        reason = (
            "Highest median OOT net PnL among candidates meeting the minimum "
            "trade count; verify in SHADOW before any human activation."
        )
    elif rows:
        status = "NO_PNL_SAMPLE"
        reason = (
            "No candidate has enough OOT trades with a finite PnL; no winner "
            "is recommended."
        )
    else:
        status = "NO_RESULTS"
        reason = "No OOT or Polymarket-OOT results have been recorded."

    return {
        "rows": rows,
        "result_count": sum(len(items) for items in grouped.values()),
        "min_trades": min_trades,
        "recommended_config_id": winner["config_id"] if winner else None,
        "recommendation_status": status,
        "recommendation_reason": reason,
    }


async def plan_run(
    session: AsyncSession,
    run_id: int,
    config_ids: Sequence[int],
) -> list[AIRunStep]:
    """Create an idempotent, bounded training/backtest plan for a run."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.status not in {"DRAFT", "PLANNING"}:
        raise AILabError(f"run {run_id} cannot be planned from {run.status}")

    normalized_ids = list(dict.fromkeys(int(config_id) for config_id in config_ids))
    if not normalized_ids:
        raise AILabError("at least one experiment config is required")
    if run.budget_experiments and len(normalized_ids) > run.budget_experiments:
        raise AILabError(
            f"experiment budget {run.budget_experiments} is smaller than "
            f"{len(normalized_ids)} requested configs"
        )
    await authorize_run_action(session, run_id, "CREATE_EXPERIMENT")

    existing = (
        await session.execute(
            select(AIRunStep.id).where(AIRunStep.run_id == run_id).limit(1)
        )
    ).first()
    if existing is not None:
        raise AILabError("run already has a plan")

    configs = (
        await session.execute(
            select(AIExperimentConfig).where(
                AIExperimentConfig.id.in_(normalized_ids)
            )
        )
    ).scalars().all()
    found_ids = {config.id for config in configs}
    missing = sorted(set(normalized_ids).difference(found_ids))
    if missing:
        raise AILabError(f"experiment configs not found: {missing}")

    if run.status == "DRAFT":
        await transition_run(session, run, "PLANNING", reason="experiment plan created")

    steps: list[AIRunStep] = []
    step_index = 0
    for config_id in normalized_ids:
        for step_type, action in PLAN_ACTIONS:
            step = AIRunStep(
                run_id=run_id,
                step_index=step_index,
                step_type=step_type,
                status="PENDING",
                action=action,
                input_payload={"config_id": config_id},
                created_at=utc_now(),
            )
            session.add(step)
            steps.append(step)
            step_index += 1
    run.summary = f"Planned {len(normalized_ids)} configs ({len(steps)} steps)."
    await session.flush()
    return steps


async def claim_next_step(
    session: AsyncSession,
    run_id: int,
) -> AIRunStep | None:
    """Atomically claim the next pending step for a worker or AI agent."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.status not in {"PLANNING", "RUNNING"}:
        raise AILabError(f"run {run_id} cannot claim work from {run.status}")

    step = (
        await session.execute(
            select(AIRunStep)
            .where(
                AIRunStep.run_id == run_id,
                AIRunStep.status == "PENDING",
            )
            .order_by(AIRunStep.step_index)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
    ).scalar_one_or_none()
    if step is None:
        return None

    if step.action:
        await authorize_run_action(session, run_id, step.action)
    if run.status == "PLANNING":
        await transition_run(session, run, "RUNNING", reason="first plan step claimed")
    now = utc_now()
    step.status = "RUNNING"
    step.started_at = now
    await session.flush()
    return step


async def record_result(
    session: AsyncSession,
    *,
    run_id: int,
    config_id: int,
    evaluation_kind: str,
    status: str = "SUCCEEDED",
    metrics: Mapping[str, Any] | None = None,
    slices: Mapping[str, Any] | None = None,
    trade_count: int | None = None,
    net_pnl: float | None = None,
    max_drawdown: float | None = None,
    artifact_id: int | None = None,
    step_id: int | None = None,
) -> ExperimentResult:
    kind = evaluation_kind.strip().upper()
    if kind not in {"TRAIN", "OOT", "POLYMARKET_OOT", "SHADOW"}:
        raise AILabError(f"unsupported evaluation kind: {kind}")
    status = status.strip().upper()
    if status not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "INSUFFICIENT_DATA"}:
        raise AILabError(f"unsupported result status: {status}")
    if trade_count is not None and trade_count < 0:
        raise AILabError("trade_count must be non-negative")

    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    action = RESULT_ACTIONS.get(kind)
    if action:
        await authorize_run_action(session, run_id, action)
    config = await session.get(AIExperimentConfig, config_id)
    if config is None:
        raise AILabError(f"experiment config {config_id} not found")
    if artifact_id is not None and await session.get(AIModelArtifact, artifact_id) is None:
        raise AILabError(f"model artifact {artifact_id} not found")

    step = None
    if step_id is not None:
        step = await session.get(AIRunStep, step_id)
        if step is None or step.run_id != run_id:
            raise AILabError(f"run step {step_id} not found")
        if step.action and action and step.action != action:
            raise AILabError(
                f"step {step_id} action {step.action} does not match {action}"
            )

    result = ExperimentResult(
        run_id=run_id,
        config_id=config_id,
        artifact_id=artifact_id,
        evaluation_kind=kind,
        status=status,
        metrics=dict(metrics) if metrics is not None else None,
        slices=dict(slices) if slices is not None else None,
        trade_count=trade_count,
        net_pnl=net_pnl,
        max_drawdown=max_drawdown,
        created_at=utc_now(),
    )
    session.add(result)
    if step is not None and status in {"SUCCEEDED", "FAILED", "INSUFFICIENT_DATA"}:
        step.status = status if status in {"SUCCEEDED", "FAILED"} else "SKIPPED"
        step.finished_at = utc_now()
        step.output_payload = {
            "result_status": status,
            "evaluation_kind": kind,
            "result_id": result.id,
        }
    await session.flush()
    return result


async def evaluate_run(
    session: AsyncSession,
    run_id: int,
) -> dict[str, Any]:
    """Persist an advisory median-OOT report and move the run to EVALUATING."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    await authorize_run_action(session, run_id, "RUN_OOT_BACKTEST")
    if run.status == "RUNNING":
        await transition_run(session, run, "EVALUATING", reason="evaluation started")
    elif run.status != "EVALUATING":
        raise AILabError(f"run {run_id} cannot be evaluated from {run.status}")

    results = (
        await session.execute(
            select(ExperimentResult)
            .where(ExperimentResult.run_id == run_id)
            .order_by(ExperimentResult.created_at, ExperimentResult.id)
        )
    ).scalars().all()
    report = build_experiment_report(
        results,
        min_trades=int((run.scope or {}).get("min_trades", 3)),
    )
    run.summary = json.dumps(report, sort_keys=True, separators=(",", ":"))
    await session.flush()
    return report


async def promote_to_shadow(
    session: AsyncSession,
    *,
    run_id: int,
    candidate_artifact_id: int,
    baseline_artifact_id: int | None,
    asset: str,
    regime: str | None = None,
) -> AIShadowAssignment:
    """Assign a candidate to SHADOW without touching active execution."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.status != "EVALUATING":
        raise AILabError(f"run {run_id} must be EVALUATING before SHADOW")
    await authorize_run_action(session, run_id, "PROMOTE_TO_SHADOW")
    if await session.get(AIModelArtifact, candidate_artifact_id) is None:
        raise AILabError(f"model artifact {candidate_artifact_id} not found")
    if baseline_artifact_id is not None and await session.get(
        AIModelArtifact, baseline_artifact_id
    ) is None:
        raise AILabError(f"baseline artifact {baseline_artifact_id} not found")

    assignment = AIShadowAssignment(
        run_id=run_id,
        candidate_artifact_id=candidate_artifact_id,
        baseline_artifact_id=baseline_artifact_id,
        asset=asset.strip().upper(),
        regime=regime.strip().lower() if regime else None,
        status="PENDING",
        created_at=utc_now(),
    )
    session.add(assignment)
    await transition_run(session, run, "SHADOW", reason="candidate assigned to SHADOW")
    await session.flush()
    return assignment
