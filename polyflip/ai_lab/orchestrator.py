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
    raw_metrics = _value(result, "metrics", {})
    metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
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
    if start is None or end is None:
        return None
    start_text = str(start).strip()
    end_text = str(end).strip()
    if not start_text or not end_text or start_text == end_text:
        return None
    return (start_text, end_text)


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
    polymarket_count = int(_finite(row.get("polymarket_oot_evaluation_count")) or 0)
    window_count = int(_finite(row.get("window_count")) or 0)
    total_trades = int(_finite(row.get("total_trades")) or 0)
    median_pnl = _finite(row.get("median_oot_pnl"))
    median_drawdown_raw = row.get("median_oot_drawdown")
    invalid_count = int(_finite(row.get("invalid_result_count")) or 0)

    if polymarket_count == 0:
        rejection_reasons.append("NO_PNL_SAMPLE")
    if invalid_count > 0:
        rejection_reasons.append("INVALID_RESULT")
    if median_pnl is None:
        if "INVALID_RESULT" not in rejection_reasons:
            rejection_reasons.append("INVALID_RESULT")
    elif median_pnl <= 0.0:
        rejection_reasons.append("NON_POSITIVE_PNL")
    if median_drawdown_raw is not None and _finite(median_drawdown_raw) is None:
        if "INVALID_RESULT" not in rejection_reasons:
            rejection_reasons.append("INVALID_RESULT")
    if total_trades < min_trades:
        rejection_reasons.append("INSUFFICIENT_TRADES")
    if window_count < min_windows:
        rejection_reasons.append("INSUFFICIENT_WINDOWS")

    status = (
        "READY_FOR_SHADOW"
        if not rejection_reasons
        else rejection_reasons[0]
    )
    return {
        "eligible": len(rejection_reasons) == 0,
        "recommendation_status": status,
        "rejection_reasons": rejection_reasons,
    }


def build_experiment_report(
    results: Sequence[Any],
    *,
    min_trades: int = MIN_TOTAL_TRADES,
    min_windows: int = MIN_WINDOWS,
) -> dict[str, Any]:
    """Aggregate per-config metrics across training, OOT backtests, and Polymarket OOT.

    Only successful POLYMARKET_OOT evaluations count toward trade volume and
    positive median PnL requirements. Other result types provide audit context
    without bypassing the evaluation gate.
    """
    by_config: dict[int, list[Any]] = defaultdict(list)
    for result in results:
        by_config[int(_value(result, "config_id"))].append(result)

    rows: list[dict[str, Any]] = []
    eligible_candidates: list[dict[str, Any]] = []
    best_candidate_seen: dict[str, Any] | None = None
    all_rejection_reasons: list[str] = []

    for config_id, config_results in by_config.items():
        polymarket_results: list[Any] = []
        oot_results: list[Any] = []
        train_results: list[Any] = []
        pnls: list[float] = []
        drawdowns: list[float] = []
        trades_by_window: dict[tuple[str, str], int] = {}
        seen_windows: set[tuple[str, str]] = set()
        artifact_ids: list[int] = []
        invalid_results = 0

        for res in config_results:
            kind = str(_value(res, "evaluation_kind", "")).upper()
            status = str(_value(res, "status", "")).upper()
            artifact = _value(res, "artifact_id")
            if artifact is not None:
                artifact_ids.append(int(artifact))

            if kind == "TRAIN":
                train_results.append(res)
            elif kind == "OOT":
                oot_results.append(res)
            elif kind == "POLYMARKET_OOT":
                polymarket_results.append(res)
                if status == "SUCCEEDED":
                    pnl = _finite(_value(res, "net_pnl"))
                    dd = _finite(_value(res, "max_drawdown"))
                    trade_cnt = _finite(_value(res, "trade_count"))
                    w_key = _oot_window_key(res)

                    if pnl is None or trade_cnt is None:
                        invalid_results += 1
                        continue

                    if w_key is not None:
                        if w_key in seen_windows:
                            # Duplicate window entry is an anomaly that corrupts the sample.
                            invalid_results += 1
                            continue
                        seen_windows.add(w_key)
                        trades_by_window[w_key] = int(trade_cnt)
                    else:
                        # Window boundaries are mandatory for Polymarket OOT.
                        invalid_results += 1
                        continue

                    pnls.append(pnl)
                    if dd is not None:
                        drawdowns.append(dd)

        total_trades = sum(trades_by_window.values())
        window_count = len(trades_by_window)
        median_pnl = _median(pnls)
        median_dd = _median(drawdowns)

        row_data = {
            "config_id": config_id,
            "evaluation_count": len(config_results),
            "polymarket_oot_evaluation_count": len(polymarket_results),
            "train_evaluation_count": len(train_results),
            "oot_backtest_evaluation_count": len(oot_results),
            "window_count": window_count,
            "total_trades": total_trades,
            "median_oot_pnl": median_pnl,
            "median_oot_drawdown": median_dd,
            "invalid_result_count": invalid_results,
            "artifact_ids": list(dict.fromkeys(artifact_ids)),
        }

        gate_res = evaluate_finalization_gate(
            row_data,
            min_trades=min_trades,
            min_windows=min_windows,
        )
        row_data["recommendation_status"] = gate_res["recommendation_status"]
        row_data["rejection_reasons"] = gate_res["rejection_reasons"]
        all_rejection_reasons.extend(gate_res["rejection_reasons"])
        rows.append(row_data)

        if gate_res["eligible"]:
            eligible_candidates.append(row_data)
        elif best_candidate_seen is None and row_data["polymarket_oot_evaluation_count"] > 0:
            best_candidate_seen = row_data

    # Rank strictly by median Polymarket PnL descending; tie-break by drawdown ascending.
    eligible_candidates.sort(
        key=lambda r: (
            -(r["median_oot_pnl"] or -float("inf")),
            r["median_oot_drawdown"] if r["median_oot_drawdown"] is not None else float("inf"),
            -r["total_trades"],
        )
    )

    if eligible_candidates:
        winner = eligible_candidates[0]
        return {
            "recommended_config_id": winner["config_id"],
            "recommendation_status": "READY_FOR_SHADOW",
            "rejection_reasons": [],
            "window_count": winner["window_count"],
            "total_trades": winner["total_trades"],
            "median_pnl": winner["median_oot_pnl"],
            "median_drawdown": winner["median_oot_drawdown"],
            "rows": rows,
        }

    # No candidate passed all criteria. Report reasons transparently.
    status = (
        best_candidate_seen.get("recommendation_status", "NO_PNL_SAMPLE")
        if best_candidate_seen
        else (all_rejection_reasons[0] if all_rejection_reasons else "NO_PNL_SAMPLE")
    )
    return {
        "recommended_config_id": None,
        "recommendation_status": status,
        "rejection_reasons": list(dict.fromkeys(all_rejection_reasons)) or [status],
        "rows": rows,
    }


async def plan_run(
    session: AsyncSession,
    run_id: int,
    config_ids: Sequence[int],
) -> list[AIRunStep]:
    """Populate an optimization run with plan steps for the given configs."""
    await authorize_run_action(session, run_id, "PLAN_RUN")
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"Optimization run {run_id} not found")

    if run.status in {"ACTIVE", "SHADOW", "FAILED", "REJECTED", "CANCELLED", "ROLLED_BACK"}:
        raise AILabError(f"Cannot add plan steps to run {run_id} in state {run.status}")

    # Verify configs exist in the immutable AI experiment configs registry.
    stmt = select(AIExperimentConfig.id).where(AIExperimentConfig.id.in_(config_ids))
    existing = set((await session.execute(stmt)).scalars().all())
    missing = set(config_ids) - existing
    if missing:
        raise AILabError(f"Configs {sorted(missing)} do not exist in ai_experiment_configs")

    # Fetch current max step_index for this run to append smoothly.
    max_idx_stmt = (
        select(AIRunStep.step_index)
        .where(AIRunStep.run_id == run_id)
        .order_by(AIRunStep.step_index.desc())
        .limit(1)
    )
    current_max = (await session.execute(max_idx_stmt)).scalar_one_or_none()
    start_index = 0 if current_max is None else current_max + 1

    plan = default_plan_steps(config_ids)
    created_steps: list[AIRunStep] = []
    for offset, item in enumerate(plan):
        step = AIRunStep(
            run_id=run_id,
            step_index=start_index + offset,
            step_type=item["step_type"],
            action=item["action"],
            status="PENDING",
            input_payload={"config_id": item["config_id"]},
            created_at=utc_now(),
        )
        session.add(step)
        created_steps.append(step)

    await transition_run(session, run, "PLANNING", reason="Generated experiment plan")
    await session.flush()
    return created_steps


async def claim_next_step(
    session: AsyncSession,
    run_id: int,
) -> AIRunStep | None:
    """Atomic step reservation for workers."""
    stmt = (
        select(AIRunStep)
        .where(AIRunStep.run_id == run_id, AIRunStep.status == "PENDING")
        .order_by(AIRunStep.step_index.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    step = (await session.execute(stmt)).scalar_one_or_none()
    if step is None:
        return None

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
    metrics: dict[str, Any] | None = None,
    slices: dict[str, Any] | None = None,
    trade_count: int | None = None,
    net_pnl: float | None = None,
    max_drawdown: float | None = None,
    artifact_id: int | None = None,
    step_id: int | None = None,
    code_sha: str | None = None,
    dataset_fingerprint: str | None = None,
    train_window_start: Any = None,
    train_window_end: Any = None,
    oot_window_start: Any = None,
    oot_window_end: Any = None,
    summary: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ExperimentResult:
    """Persist an immutable result row and update associated run step state."""
    action = RESULT_ACTIONS.get(evaluation_kind.upper(), "RECORD_EXPERIMENT_RESULT")
    await authorize_run_action(session, run_id, action)

    # Persist the immutable result entry.
    result = ExperimentResult(
        run_id=run_id,
        config_id=config_id,
        artifact_id=artifact_id,
        evaluation_kind=evaluation_kind.upper(),
        status=status.upper(),
        metrics=metrics,
        slices=slices,
        trade_count=trade_count,
        net_pnl=net_pnl,
        max_drawdown=max_drawdown,
        code_sha=code_sha,
        dataset_fingerprint=dataset_fingerprint,
        train_window_start=train_window_start,
        train_window_end=train_window_end,
        oot_window_start=oot_window_start,
        oot_window_end=oot_window_end,
        summary=summary,
        error_code=error_code,
        error_message=error_message,
        created_at=utc_now(),
    )
    session.add(result)
    await session.flush()

    # If linked to a queue step, close the step deterministically.
    target_step: AIRunStep | None = None
    if step_id is not None:
        target_step = await session.get(AIRunStep, step_id)
    else:
        # Fallback: locate matching running step for this config.
        stmt = (
            select(AIRunStep)
            .where(
                AIRunStep.run_id == run_id,
                AIRunStep.action == action,
                AIRunStep.status == "RUNNING",
            )
            .order_by(AIRunStep.step_index.asc())
            .limit(1)
        )
        target_step = (await session.execute(stmt)).scalar_one_or_none()

    if target_step is not None:
        if status.upper() == "SUCCEEDED":
            target_step.status = "SUCCEEDED"
        elif status.upper() == "INSUFFICIENT_DATA":
            target_step.status = "SKIPPED"
        else:
            target_step.status = "FAILED"

        target_step.finished_at = utc_now()
        target_step.output_payload = {
            "result_id": result.id,
            "status": status.upper(),
            "metrics": metrics,
            "net_pnl": net_pnl,
            "trade_count": trade_count,
        }
        target_step.summary = summary
        target_step.error_code = error_code
        target_step.error_message = error_message
        await session.flush()

    return result


async def evaluate_run(
    session: AsyncSession,
    run_id: int,
) -> dict[str, Any]:
    """Aggregate run metrics and update optimization run summary."""
    await authorize_run_action(session, run_id, "EVALUATE_RUN")
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"Optimization run {run_id} not found")

    stmt = select(ExperimentResult).where(ExperimentResult.run_id == run_id)
    results = (await session.execute(stmt)).scalars().all()
    report = build_experiment_report(results)

    run.summary = json.dumps(
        {
            "report": report,
            "status": report.get("recommendation_status"),
            "rejection_reasons": report.get("rejection_reasons", []),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    await transition_run(session, run, "EVALUATING", reason="Generated experiment evaluation report")
    await session.flush()
    return report


async def promote_to_shadow(
    session: AsyncSession,
    *,
    run_id: int,
    candidate_artifact_id: int,
    baseline_artifact_id: int | None = None,
    asset: str,
    regime: str | None = None,
) -> AIShadowAssignment:
    """Create a passive shadow evaluation assignment."""
    await authorize_run_action(session, run_id, "PROMOTE_TO_SHADOW")
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"Optimization run {run_id} not found")

    artifact = await session.get(AIModelArtifact, candidate_artifact_id)
    if artifact is None:
        raise AILabError(f"Candidate artifact {candidate_artifact_id} not found")

    assignment = AIShadowAssignment(
        run_id=run_id,
        candidate_artifact_id=candidate_artifact_id,
        baseline_artifact_id=baseline_artifact_id,
        asset=asset,
        regime=regime,
        status="RUNNING",
        started_at=utc_now(),
        created_at=utc_now(),
    )
    session.add(assignment)
    await transition_run(session, run, "SHADOW", reason=f"Promoted artifact {candidate_artifact_id} to passive shadow")
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
    """Evaluate run results, enforce gates, and optionally promote the winner to SHADOW."""
    await authorize_run_action(session, run_id, "FINALIZE_RUN")
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"Optimization run {run_id} not found")

    stmt = select(ExperimentResult).where(ExperimentResult.run_id == run_id)
    results = (await session.execute(stmt)).scalars().all()
    report = build_experiment_report(results)

    assignment: AIShadowAssignment | None = None
    config_id = report.get("recommended_config_id")

    if config_id is not None and auto_shadow:
        config = await session.get(AIExperimentConfig, config_id)
        if config is None:
            raise AILabError(f"Config {config_id} not found in ai_experiment_configs")

        resolved_asset = asset or config.asset
        if not resolved_asset:
            raise AILabError("Asset must be provided to promote winner to SHADOW")

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
        # Rejection or report-only mode: retain any prior assignment provenance.
        session_run = await session.get(AIOptimizationRun, run_id)
        previous_assignment = None
        if session_run is not None:
            try:
                previous_summary = json.loads(session_run.summary or "{}")
            except (TypeError, ValueError):
                previous_summary = {}
            previous_assignment = previous_summary.get("shadow_assignment")
            session_run.summary = json.dumps(
                {
                    "report": report,
                    "status": report.get("recommendation_status"),
                    "rejection_reasons": report.get("rejection_reasons", []),
                    "shadow_assignment": previous_assignment,
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
            if last_step is None:
                # AIStepAuditLog.step_id is mandatory. Create a durable
                # finalization step when a run has no queued steps yet.
                last_step = AIRunStep(
                    run_id=run_id,
                    step_index=0,
                    step_type="FINALIZE",
                    status="FAILED",
                    action="FINALIZE_RUN",
                    summary="Finalization gate rejected the run before a plan step existed.",
                    error_code=f"GATE_REJECTED_{report.get('recommendation_status')}",
                    created_at=utc_now(),
                    finished_at=utc_now(),
                )
                session.add(last_step)
                await session.flush()
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
