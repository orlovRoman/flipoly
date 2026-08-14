"""Safe experiment orchestration for AI Lab.

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
    AIStepAuditLog,
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
# Result status INSUFFICIENT_DATA is persisted on ExperimentResult, but the
# corresponding queue step must still be closed as SKIPPED rather than orphaned.
RESULT_CLOSING_STATUSES = {"SUCCEEDED", "FAILED", "INSUFFICIENT_DATA"}

MIN_TOTAL_TRADES = 50
MIN_WINDOWS = 3


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


def _oot_window_key(result: Any) -> tuple[str, str] | None:
    metrics = _value(result, "metrics", {}) or {}
    start = _value(
        result,
        "oot_window_start",
        metrics.get("oot_window_start", metrics.get("window_start")),
    )
    end = _value(
        result,
        "oot_window_end",
        metrics.get("oot_window_end", metrics.get("window_end")),
    )
    if start is not None and end is not None:
        return (str(start), str(end))
    return None


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


def evaluate_finalization_gate(
    row: Mapping[str, Any],
    *,
    min_trades: int = MIN_TOTAL_TRADES,
    min_windows: int = MIN_WINDOWS,
) -> dict[str, Any]:
    """Evaluate whether a candidate config passes strict finalization gates."""
    rejection_reasons: list[str] = []
    polymarket_count = int(row.get("polymarket_oot_evaluation_count") or 0)
    if polymarket_count == 0:
        rejection_reasons.append("NO_PNL_SAMPLE")

    window_count = int(row.get("window_count") or 0)
    total_trades = int(row.get("total_trades") or 0)
    median_pnl = row.get("median_oot_pnl")
    median_drawdown = row.get("median_oot_drawdown")

    if median_pnl is None or not math.isfinite(median_pnl):
        rejection_reasons.append("INVALID_RESULT")
    elif median_pnl <= 0.0:
        rejection_reasons.append("NON_POSITIVE_PNL")

    if median_drawdown is not None and not math.isfinite(median_drawdown):
        if "INVALID_RESULT" not in rejection_reasons:
            rejection_reasons.append("INVALID_RESULT")

    if total_trades < min_trades:
        rejection_reasons.append("INSUFFICIENT_TRADES")

    if window_count < min_windows:
        rejection_reasons.append("INSUFFICIENT_WINDOWS")

    is_eligible = len(rejection_reasons) == 0
    return {
        "eligible": is_eligible,
        "rejection_reasons": rejection_reasons,
        "window_count": window_count,
        "total_trades": total_trades,
        "median_pnl": (
            median_pnl
            if (median_pnl is not None and math.isfinite(median_pnl))
            else None
        ),
    }


def build_experiment_report(
    results: Sequence[ExperimentResult | Mapping[str, Any]],
    *,
    min_trades: int = MIN_TOTAL_TRADES,
    min_windows: int = MIN_WINDOWS,
) -> dict[str, Any]:
    """Build an advisory report from persisted OOT evaluations.

    Generic OOT results remain visible as diagnostics. A candidate is eligible
    for SHADOW only when it has a finite PnL/trade sample from a real
    POLYMARKET_OOT evaluation passing the strict gate:
      - At least min_windows (default 3) distinct non-empty OOT windows;
      - At least min_trades (default 50) total trades;
      - Strictly positive median net PnL (median_net_pnl > 0.0);
      - Finite numeric PnL and drawdown values.
    AUC/ECE are reported but never replace PnL evidence.
    """
    if min_trades < 0:
        raise AILabError("min_trades must be non-negative")
    if min_windows < 0:
        raise AILabError("min_windows must be non-negative")

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
        unique_windows: set[tuple[str, str]] = set()
        window_details: list[dict[str, Any]] = []

        for result in successful_results:
            metrics = _value(result, "metrics", {}) or {}
            for metric_name in ("auc", "ece", "brier", "log_loss", "win_rate"):
                metric_values[metric_name].append(metrics.get(metric_name))
            artifact_id = _value(result, "artifact_id")
            if artifact_id is not None:
                artifact_ids.add(int(artifact_id))

        for result in polymarket_results:
            metrics = _value(result, "metrics", {}) or {}
            res_pnl = _value(result, "net_pnl", metrics.get("net_pnl"))
            res_trades = _value(result, "trade_count", metrics.get("n_trades"))
            res_dd = _value(result, "max_drawdown", metrics.get("max_drawdown"))

            pnl_values.append(res_pnl)
            trade_values.append(res_trades)
            drawdown_values.append(res_dd)

            w_key = _oot_window_key(result)
            if w_key is not None:
                unique_windows.add(w_key)
                window_details.append(
                    {
                        "oot_window_start": w_key[0],
                        "oot_window_end": w_key[1],
                        "net_pnl": _finite(res_pnl),
                        "trade_count": int(_finite(res_trades) or 0)
                        if _finite(res_trades) is not None
                        else 0,
                        "max_drawdown": _finite(res_dd),
                    }
                )

        median_pnl = _median(pnl_values)
        median_trades = _median(trade_values)
        median_drawdown = _median(drawdown_values)
        total_trades = sum(
            int(_finite(t) or 0) for t in trade_values if _finite(t) is not None
        )

        row: dict[str, Any] = {
            "config_id": config_id,
            "evaluation_count": len(all_results),
            "oot_evaluation_count": len(by_kind["OOT"]),
            "polymarket_oot_evaluation_count": len(polymarket_results),
            "artifact_ids": sorted(artifact_ids),
            "window_count": len(unique_windows),
            "total_trades": total_trades,
            "median_oot_pnl": median_pnl,
            "median_oot_trades": int(round(median_trades))
            if median_trades is not None
            else 0,
            "median_oot_drawdown": median_drawdown,
            "windows": window_details,
            "auc": _median(metric_values["auc"]),
            "ece": _median(metric_values["ece"]),
            "brier": _median(metric_values["brier"]),
            "log_loss": _median(metric_values["log_loss"]),
            "win_rate": _median(metric_values["win_rate"]),
        }

        gate_res = evaluate_finalization_gate(
            row, min_trades=min_trades, min_windows=min_windows
        )
        row["eligible_for_shadow"] = gate_res["eligible"]
        row["rejection_reasons"] = gate_res["rejection_reasons"]
        rows.append(row)

    eligible = [row for row in rows if row["eligible_for_shadow"]]
    winner = (
        max(
            eligible,
            key=lambda row: (
                row["median_oot_pnl"],
                -(abs(row["median_oot_drawdown"] or 0.0)),
                row["total_trades"],
            ),
        )
        if eligible
        else None
    )

    if winner:
        status = "READY_FOR_SHADOW"
        rejection_reasons: list[str] = []
        reason = (
            "Highest median Polymarket-OOT net PnL among candidates meeting the "
            "minimum trade count and window criteria; verify in SHADOW before any human activation."
        )
    elif rows:
        rejection_reasons = sorted(
            list(
                dict.fromkeys(
                    reason
                    for row in rows
                    for reason in row.get("rejection_reasons", [])
                )
            )
        )
        if not rejection_reasons:
            rejection_reasons = ["NO_PNL_SAMPLE"]
        if "NO_PNL_SAMPLE" in rejection_reasons:
            status = "NO_PNL_SAMPLE"
            reason = (
                "No candidate has enough real Polymarket-OOT trades with finite PnL; "
                "AUC/ECE do not substitute for this evidence."
            )
        elif "INSUFFICIENT_TRADES" in rejection_reasons:
            status = "INSUFFICIENT_TRADES"
            reason = (
                f"Candidates did not meet the minimum total trade threshold ({min_trades})."
            )
        elif "INSUFFICIENT_WINDOWS" in rejection_reasons:
            status = "INSUFFICIENT_WINDOWS"
            reason = (
                f"Candidates did not meet the minimum OOT windows threshold ({min_windows})."
            )
        elif "NON_POSITIVE_PNL" in rejection_reasons:
            status = "NON_POSITIVE_PNL"
            reason = "Candidates produced non-positive median net PnL."
        elif "INVALID_RESULT" in rejection_reasons:
            status = "INVALID_RESULT"
            reason = "Candidates produced non-finite or invalid metric results."
        else:
            status = rejection_reasons[0]
            reason = f"Candidate rejected by finalization gate: {', '.join(rejection_reasons)}"
    else:
        status = "NO_RESULTS"
        rejection_reasons = ["NO_RESULTS"]
        reason = "No OOT or Polymarket-OOT results have been recorded."

    top_window_count = (
        winner["window_count"]
        if winner
        else (rows[0]["window_count"] if rows else 0)
    )
    top_total_trades = (
        winner["total_trades"]
        if winner
        else (rows[0]["total_trades"] if rows else 0)
    )
    top_median_pnl = (
        winner["median_oot_pnl"]
        if winner
        else (rows[0]["median_oot_pnl"] if rows else None)
    )

    return {
        "rows": rows,
        "result_count": sum(
            len(items)
            for by_kind in grouped.values()
            for items in by_kind.values()
        ),
        "min_trades": min_trades,
        "min_windows": min_windows,
        "recommended_config_id": winner["config_id"] if winner else None,
        "recommendation_status": status,
        "rejection_reasons": rejection_reasons,
        "window_count": top_window_count,
        "total_trades": top_total_trades,
        "median_pnl": top_median_pnl,
        "recommendation_reason": reason,
    }


async def plan_run(
    session: AsyncSession,
    *,
    run_id: int,
    config_ids: Sequence[int],
) -> list[AIRunStep]:
    """Populate the bounded queue for one optimization run."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.status not in {"DRAFT", "PLANNING"}:
        raise AILabError(f"run {run_id} cannot be planned from {run.status}")
    await authorize_run_action(session, run_id, "CREATE_EXPERIMENT")

    configs = (
        await session.execute(
            select(AIExperimentConfig).where(
                AIExperimentConfig.id.in_(list(config_ids))
            )
        )
    ).scalars().all()
    found_ids = {config.id for config in configs}
    missing = set(config_ids) - found_ids
    if missing:
        raise AILabError(f"experiment configs not found: {sorted(missing)}")

    existing_steps = (
        await session.execute(
            select(AIRunStep.id).where(AIRunStep.run_id == run_id).limit(1)
        )
    ).scalar_one_or_none()
    if existing_steps is not None:
        raise AILabError(f"run {run_id} already has planned steps")

    planned = default_plan_steps(config_ids)
    limit = int(run.experiment_budget or len(planned))
    if len(planned) > limit:
        raise AILabError(
            f"plan exceeds experiment budget ({len(planned)} > {limit})"
        )

    db_steps: list[AIRunStep] = []
    for item in planned:
        step = AIRunStep(
            run_id=run_id,
            step_index=item["step_index"],
            action=item["action"],
            status="PENDING",
            input_payload={
                "step_type": item["step_type"],
                "config_id": item["config_id"],
            },
            created_at=utc_now(),
        )
        session.add(step)
        db_steps.append(step)

    if run.status == "DRAFT":
        await transition_run(
            session, run, "PLANNING", reason="experiment plan created"
        )
    await session.flush()
    return db_steps


async def claim_next_step(
    session: AsyncSession,
    run_id: int,
) -> AIRunStep | None:
    """Atomically claim the next pending step using SELECT FOR UPDATE SKIP LOCKED."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.status not in {"PLANNING", "RUNNING"}:
        raise AILabError(f"run {run_id} cannot execute steps from {run.status}")

    stmt = (
        select(AIRunStep)
        .where(
            AIRunStep.run_id == run_id,
            AIRunStep.status == "PENDING",
        )
        .order_by(AIRunStep.step_index, AIRunStep.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    step = (await session.execute(stmt)).scalar_one_or_none()
    if step is None:
        return None

    await authorize_run_action(session, run_id, step.action)
    if run.status == "PLANNING":
        await transition_run(session, run, "RUNNING", reason="first step claimed")
    step.status = "RUNNING"
    step.started_at = utc_now()
    await session.flush()
    return step


async def record_result(
    session: AsyncSession,
    *,
    run_id: int,
    config_id: int,
    evaluation_kind: str,
    status: str = "SUCCEEDED",
    step_id: int | None = None,
    artifact_id: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    slices: Mapping[str, Any] | None = None,
    trade_count: int | None = None,
    net_pnl: float | None = None,
    max_drawdown: float | None = None,
    code_sha: str | None = None,
    dataset_fingerprint: str | None = None,
    train_window_start: Any | None = None,
    train_window_end: Any | None = None,
    oot_window_start: Any | None = None,
    oot_window_end: Any | None = None,
    summary: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ExperimentResult:
    """Record an immutable result and atomically update the queue step."""
    kind = evaluation_kind.upper()
    action = RESULT_ACTIONS.get(kind)
    if action is None:
        raise AILabError(f"unsupported evaluation kind: {evaluation_kind}")
    await authorize_run_action(session, run_id, action)

    step: AIRunStep | None = None
    if step_id is not None:
        step = await session.get(AIRunStep, step_id)
        if step is None or step.run_id != run_id:
            raise AILabError(f"step {step_id} does not belong to run {run_id}")
        if step.status not in {"RUNNING", "PENDING"}:
            raise AILabError(f"step {step_id} cannot be closed from {step.status}")

    result = ExperimentResult(
        run_id=run_id,
        config_id=config_id,
        artifact_id=artifact_id,
        evaluation_kind=kind,
        status=status,
        metrics=dict(metrics or {}),
        slices=dict(slices or {}),
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
    if step is not None and status in RESULT_CLOSING_STATUSES:
        step.status = status if status in {"SUCCEEDED", "FAILED"} else "SKIPPED"
        step.finished_at = utc_now()
        step.output_payload = {
            "result_id": result.id,
            "result_status": status,
            "evaluation_kind": kind,
        }
        step.summary = summary[:4000] if summary else step.summary
        step.error_code = error_code[:64] if error_code else None
        step.error_message = error_message[:4000] if error_message else None
        await session.flush()
    return result


async def evaluate_run(
    session: AsyncSession,
    run_id: int,
) -> dict[str, Any]:
    """Persist a strict median-OOT report and move the run to EVALUATING."""
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

    scope = run.scope or {}
    min_trades = max(
        MIN_TOTAL_TRADES, int(scope.get("min_trades", MIN_TOTAL_TRADES))
    )
    min_windows = max(
        MIN_WINDOWS, int(scope.get("min_windows", MIN_WINDOWS))
    )

    report = build_experiment_report(
        results,
        min_trades=min_trades,
        min_windows=min_windows,
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

    scope = run.scope or {}
    min_trades = max(
        MIN_TOTAL_TRADES, int(scope.get("min_trades", MIN_TOTAL_TRADES))
    )
    min_windows = max(
        MIN_WINDOWS, int(scope.get("min_windows", MIN_WINDOWS))
    )
    report = build_experiment_report(
        results,
        min_trades=min_trades,
        min_windows=min_windows,
    )
    recommended_config_id = report["recommended_config_id"]
    if (
        recommended_config_id is None
        or report["recommendation_status"] != "READY_FOR_SHADOW"
    ):
        raise AILabError(
            "cannot promote without a real Polymarket-OOT winner meeting strict gate"
        )
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
            select(AIShadowAssignment).where(
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
        if existing.candidate_artifact_id == candidate_artifact_id:
            return existing
        raise AILabError(
            "an active SHADOW assignment already exists for this scope with a different artifact"
        )

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
    await transition_run(
        session, run, "SHADOW", reason="recommended candidate assigned to SHADOW"
    )
    await session.flush()
    return assignment


async def finalize_run(
    session: AsyncSession,
    run_id: int,
    *,
    auto_shadow: bool = True,
    asset: str | None = None,
    regime: str | None = None,
    candidate_artifact_id: int | None = None,
    baseline_artifact_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate a completed run and optionally assign its winner to SHADOW.

    This is the autonomous laboratory boundary: it may select and assign a
    candidate for passive observation, but it never transitions to ACTIVE or
    changes RuntimeSettings/live execution. The report is retained in the run
    summary together with the assignment provenance.
    """
    report = await evaluate_run(session, run_id)
    assignment: AIShadowAssignment | None = None
    if auto_shadow and report["recommendation_status"] == "READY_FOR_SHADOW":
        config_id = report["recommended_config_id"]
        config = await session.get(AIExperimentConfig, config_id)
        if config is None:
            raise AILabError(
                f"recommended experiment config {config_id} no longer exists"
            )
        resolved_asset = (asset or config.asset or "").strip()
        if not resolved_asset:
            raise AILabError(
                "asset is required for automatic SHADOW assignment when the "
                "experiment config has no asset"
            )
        winner = next(
            row for row in report["rows"] if row["config_id"] == config_id
        )
        resolved_artifact = candidate_artifact_id
        if resolved_artifact is None and winner["artifact_ids"]:
            resolved_artifact = int(winner["artifact_ids"][0])
        if resolved_artifact is None:
            raise AILabError(
                "recommended experiment has no model artifact for SHADOW"
            )
        assignment = await promote_to_shadow(
            session,
            run_id=run_id,
            candidate_artifact_id=resolved_artifact,
            baseline_artifact_id=baseline_artifact_id,
            asset=resolved_asset,
            regime=regime or config.regime,
        )
        session_run = await session.get(AIOptimizationRun, run_id)
        if session_run is not None:
            session_run.summary = json.dumps(
                {
                    "report": report,
                    "status": "READY_FOR_SHADOW",
                    "shadow_assignment": {
                        "assignment_id": assignment.id,
                        "candidate_artifact_id": assignment.candidate_artifact_id,
                        "baseline_artifact_id": assignment.baseline_artifact_id,
                        "asset": assignment.asset,
                        "regime": assignment.regime,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            await session.flush()
    else:
        # Rejection or report-only mode: record durable summary & audit log
        session_run = await session.get(AIOptimizationRun, run_id)
        if session_run is not None:
            session_run.summary = json.dumps(
                {
                    "report": report,
                    "status": report.get("recommendation_status"),
                    "rejection_reasons": report.get("rejection_reasons", []),
                    "shadow_assignment": None,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            await session.flush()

        if report.get("recommendation_status") != "READY_FOR_SHADOW":
            last_step = (
                await session.execute(
                    select(AIRunStep)
                    .where(AIRunStep.run_id == run_id)
                    .order_by(AIRunStep.step_index.desc(), AIRunStep.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if last_step is not None:
                reasons_str = (
                    ", ".join(report.get("rejection_reasons", []))
                    or report.get("recommendation_status", "UNKNOWN")
                )
                audit_entry = AIStepAuditLog(
                    run_id=run_id,
                    step_id=last_step.id,
                    config_id=report.get("recommended_config_id"),
                    action="FINALIZE_RUN",
                    error_code=f"GATE_REJECTED_{report.get('recommendation_status')}",
                    error_message=f"Finalization rejected candidate: {reasons_str}",
                    payload={
                        "recommendation_status": report.get("recommendation_status"),
                        "rejection_reasons": report.get("rejection_reasons", []),
                        "window_count": report.get("window_count"),
                        "total_trades": report.get("total_trades"),
                        "median_pnl": report.get("median_pnl"),
                    },
                    created_at=utc_now(),
                )
                session.add(audit_entry)
                await session.flush()

    return {"report": report, "assignment": assignment}
