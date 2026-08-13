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
    "SHADOW": "PROMOTE_TO_SHADOW",
}

TERMINAL_STEP_STATUSES = {"SUCCEEDED", "FAILED", "SKIPPED"}


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
    normalized = list(dict.fromkeys(int(config_id) for config_id in config_ids))
    steps: list[dict[str, Any]] = []
    step_index = 0
    for config_id in normalized:
        for step_type, action in PLAN_ACTIONS:
            steps.append(
                {
                    "step_index": step_index,
                    "step_type": step_type,
                    "action": action,
                    "config_id": config_id,
                }
            )
            step_index += 1
    return steps


def build_experiment_report(
    results: Sequence[ExperimentResult | Mapping[str, Any]],
    *,
    min_trades: int = 3,
) -> dict[str, Any]:
    """Build an advisory report from persisted OOT evaluations.

    Generic OOT results remain visible as diagnostics. A candidate is eligible
    for SHADOW only when it has a finite PnL/trade sample from a real
    POLYMARKET_OOT evaluation. AUC/ECE are reported but never replace PnL
    evidence.
    """
    if min_trades < 0:
        raise AILabError("min_trades must be non-negative")

    grouped: dict[int, dict[str, list[Any]]] = defaultdict(
        lambda: {"OOT": [], "POLYMARKET_OOT": []}
    )
    for result in results:
        kind = str(_value(result, "evaluation_kind", "")).upper()
        if kind in {"OOT", "POLYMARKET_OOT"}:
            config_id = _value(result, "config_id")
            if config_id is not None:
                grouped[int(config_id)][kind].append(result)

    rows: list[dict[str, Any]] = []
    for config_id, by_kind in sorted(grouped.items()):
        all_results = by_kind["OOT"] + by_kind["POLYMARKET_OOT"]
        successful_results = [
            result
            for result in all_results
            if str(_value(result, "status", "SUCCEEDED")).upper() == "SUCCEEDED"
        ]
        polymarket_results = [
            result
            for result in by_kind["POLYMARKET_OOT"]
            if str(_value(result, "status", "SUCCEEDED")).upper() == "SUCCEEDED"
        ]
        metric_values: dict[str, list[Any]] = defaultdict(list)
        pnl_values: list[Any] = []
        trade_values: list[Any] = []
        drawdown_values: list[Any] = []
        artifact_ids: set[int] = set()
        for result in successful_results:
            metrics = _value(result, "metrics", {}) or {}
            for metric_name in ("auc", "ece", "brier", "log_loss", "win_rate"):
                metric_values[metric_name].append(metrics.get(metric_name))
            artifact_id = _value(result, "artifact_id")
            if artifact_id is not None:
                artifact_ids.add(int(artifact_id))

        for result in polymarket_results:
            metrics = _value(result, "metrics", {}) or {}
            pnl_values.append(_value(result, "net_pnl", metrics.get("net_pnl")))
            trade_values.append(
                _value(result, "trade_count", metrics.get("n_trades"))
            )
            drawdown_values.append(
                _value(result, "max_drawdown", metrics.get("max_drawdown"))
            )

        median_pnl = _median(pnl_values)
        median_trades = _median(trade_values)
        median_drawdown = _median(drawdown_values)
        row = {
            "config_id": config_id,
            "evaluation_count": len(all_results),
            "oot_evaluation_count": len(by_kind["OOT"]),
            "polymarket_oot_evaluation_count": len(polymarket_results),
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
            bool(polymarket_results)
            and median_pnl is not None
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
            "Highest median Polymarket-OOT net PnL among candidates meeting the "
            "minimum trade count; verify in SHADOW before any human activation."
        )
    elif rows:
        status = "NO_PNL_SAMPLE"
        reason = (
            "No candidate has enough real Polymarket-OOT trades with finite PnL; "
            "AUC/ECE do not substitute for this evidence."
        )
    else:
        status = "NO_RESULTS"
        reason = "No OOT or Polymarket-OOT results have been recorded."

    return {
        "rows": rows,
        "result_count": sum(len(items) for by_kind in grouped.values() for items in by_kind.values()),
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
    for payload in default_plan_steps(normalized_ids):
        step = AIRunStep(
            run_id=run_id,
            step_index=payload["step_index"],
            step_type=payload["step_type"],
            status="PENDING",
            action=payload["action"],
            input_payload={"config_id": payload["config_id"]},
            summary=f"Queued {payload['step_type']} for config {payload['config_id']}.",
            created_at=utc_now(),
        )
        session.add(step)
        steps.append(step)
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
    code_sha: str | None = None,
    dataset_fingerprint: str | None = None,
    train_window_start: Any | None = None,
    train_window_end: Any | None = None,
    oot_window_start: Any | None = None,
    oot_window_end: Any | None = None,
) -> ExperimentResult:
    kind = evaluation_kind.strip().upper()
    if kind not in RESULT_ACTIONS:
        raise AILabError(f"unsupported evaluation kind: {kind}")
    status = status.strip().upper()
    if status not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "INSUFFICIENT_DATA"}:
        raise AILabError(f"unsupported result status: {status}")
    if trade_count is not None and trade_count < 0:
        raise AILabError("trade_count must be non-negative")
    if net_pnl is not None and _finite(net_pnl) is None:
        raise AILabError("net_pnl must be finite when supplied")
    if max_drawdown is not None and _finite(max_drawdown) is None:
        raise AILabError("max_drawdown must be finite when supplied")

    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    await authorize_run_action(session, run_id, RESULT_ACTIONS[kind])
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
        if step.action and step.action != RESULT_ACTIONS[kind]:
            raise AILabError(
                f"step {step_id} action {step.action} does not match "
                f"{RESULT_ACTIONS[kind]}"
            )
        if step.status in TERMINAL_STEP_STATUSES:
            raise AILabError(f"step {step_id} is already terminal")

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
        code_sha=code_sha,
        dataset_fingerprint=dataset_fingerprint,
        train_window_start=train_window_start,
        train_window_end=train_window_end,
        oot_window_start=oot_window_start,
        oot_window_end=oot_window_end,
        created_at=utc_now(),
    )
    session.add(result)
    await session.flush()
    if step is not None and status in TERMINAL_STEP_STATUSES:
        step.status = status if status in {"SUCCEEDED", "FAILED"} else "SKIPPED"
        step.finished_at = utc_now()
        step.output_payload = {
            "result_id": result.id,
            "result_status": status,
            "evaluation_kind": kind,
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
    """Assign the recommended candidate to SHADOW without active execution."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.status != "EVALUATING":
        raise AILabError(f"run {run_id} must be EVALUATING before SHADOW")
    await authorize_run_action(session, run_id, "PROMOTE_TO_SHADOW")

    results = (
        await session.execute(
            select(ExperimentResult)
            .where(ExperimentResult.run_id == run_id)
        )
    ).scalars().all()
    report = build_experiment_report(
        results,
        min_trades=int((run.scope or {}).get("min_trades", 3)),
    )
    recommended_config_id = report["recommended_config_id"]
    if recommended_config_id is None:
        raise AILabError("cannot promote without a real Polymarket-OOT winner")
    winner = next(
        row for row in report["rows"]
        if row["config_id"] == recommended_config_id
    )
    if candidate_artifact_id not in winner["artifact_ids"]:
        raise AILabError(
            "candidate artifact is not attached to the recommended experiment"
        )
    if await session.get(AIModelArtifact, candidate_artifact_id) is None:
        raise AILabError(f"model artifact {candidate_artifact_id} not found")
    if baseline_artifact_id is not None and await session.get(
        AIModelArtifact, baseline_artifact_id
    ) is None:
        raise AILabError(f"baseline artifact {baseline_artifact_id} not found")

    existing = (
        await session.execute(
            select(AIShadowAssignment.id).where(
                AIShadowAssignment.run_id == run_id,
                AIShadowAssignment.asset == asset.strip().upper(),
                AIShadowAssignment.regime
                == (regime.strip().lower() if regime else None),
                AIShadowAssignment.status.in_({"PENDING", "RUNNING"}),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise AILabError("an active SHADOW assignment already exists for this scope")

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
    await transition_run(session, run, "SHADOW", reason="recommended candidate assigned to SHADOW")
    await session.flush()
    return assignment
