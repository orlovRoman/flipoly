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
    """Return a stable key for one persisted OOT window."""
    raw_metrics = _value(result, "metrics", {}) or {}
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
    if start is not None and end is not None:
        start_text = str(start).strip()
        end_text = str(end).strip()
        if start_text and end_text and start_text != end_text:
            return ("timestamp", f"{start_text}/{end_text}")
    ordinal = _value(result, "window")
    if ordinal is not None:
        source = _value(result, "source")
        token = f"{source}:{ordinal}" if source is not None else str(ordinal)
        return ("ordinal", token)
    return None


def _result_oot_windows(result: Any) -> list[Mapping[str, Any]]:
    """Extract per-window summaries persisted in slices or metrics."""
    for field in ("slices", "metrics"):
        raw = _value(result, field, {}) or {}
        if isinstance(raw, Mapping):
            windows = raw.get("oot_windows")
            if isinstance(windows, list):
                return [item for item in windows if isinstance(item, Mapping)]
    return [result] if _oot_window_key(result) is not None else []


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
    mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate finalization gates; RESEARCH keeps valid low-evidence rows provisional."""
    from polyflip.config import settings
    research_mode = str(mode or getattr(settings, "AI_LAB_MODE", "STANDARD")).upper() == "RESEARCH"
    polymarket_count = int(_finite(row.get("polymarket_oot_evaluation_count")) or 0)
    window_count = int(_finite(row.get("window_count")) or 0)
    total_trades = int(_finite(row.get("total_trades")) or 0)
    median_pnl = _finite(row.get("median_oot_pnl"))
    median_drawdown_raw = row.get("median_oot_drawdown")
    max_drawdown_limit = 25.0
    invalid_count = int(_finite(row.get("invalid_result_count")) or 0)

    if polymarket_count == 0:
        return {
            "eligible": False,
            "recommendation_status": "NO_PNL_SAMPLE",
            "rejection_reasons": ["NO_PNL_SAMPLE"],
            "window_count": window_count,
            "total_trades": total_trades,
            "median_pnl": median_pnl,
            "median_drawdown": None,
        }

    rejection_reasons: list[str] = []
    if invalid_count > 0:
        rejection_reasons.append("INVALID_RESULT")
    if median_pnl is None:
        if "INVALID_RESULT" not in rejection_reasons:
            rejection_reasons.append("INVALID_RESULT")
    elif median_pnl <= 0.0:
        rejection_reasons.append("NON_POSITIVE_PNL")
    drawdown_num = _finite(median_drawdown_raw) if median_drawdown_raw is not None else None
    if median_drawdown_raw is not None and drawdown_num is None:
        if "INVALID_RESULT" not in rejection_reasons:
            rejection_reasons.append("INVALID_RESULT")
    elif drawdown_num is not None:
        if drawdown_num < 0:
            rejection_reasons.append("INVALID_RESULT")
        elif drawdown_num > max_drawdown_limit:
            rejection_reasons.append("EXCESSIVE_DRAWDOWN")
    if total_trades < min_trades:
        rejection_reasons.append("INSUFFICIENT_TRADES")
    if window_count < min_windows:
        rejection_reasons.append("INSUFFICIENT_WINDOWS")

    reasons = list(dict.fromkeys(rejection_reasons))
    hard_reasons = {"INVALID_RESULT", "EXCESSIVE_DRAWDOWN"}
    if research_mode and not any(reason in hard_reasons for reason in reasons):
        status = "READY_FOR_SHADOW" if not reasons else "RESEARCH_PROVISIONAL"
        eligible = True
    else:
        status = "READY_FOR_SHADOW" if not reasons else reasons[0]
        eligible = not reasons
    return {
        "eligible": eligible,
        "recommendation_status": status,
        "rejection_reasons": reasons,
        "window_count": window_count,
        "total_trades": total_trades,
        "median_pnl": median_pnl,
        "median_drawdown": drawdown_num,
    }

def build_experiment_report(
    results: Sequence[ExperimentResult | Mapping[str, Any]],
    *,
    min_trades: int = MIN_TOTAL_TRADES,
    min_windows: int = MIN_WINDOWS,
    mode: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-config metrics across training, OOT backtests, and Polymarket OOT.

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
            raw_metrics = _value(result, "metrics", {}) or {}
            metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
            for metric_name in ("auc", "ece", "brier", "log_loss", "win_rate"):
                metric_values[metric_name].append(metrics.get(metric_name))
            artifact_id = _value(result, "artifact_id")
            if artifact_id is not None:
                artifact_ids.add(int(artifact_id))

        invalid_result_count = 0
        for result_index, result in enumerate(polymarket_results):
            windows = _result_oot_windows(result)
            if not windows:
                invalid_result_count += 1
                continue
            for window in windows:
                window_data = window if isinstance(window, Mapping) else {}
                w_key = _oot_window_key(window_data) or _oot_window_key(result)
                if w_key is None:
                    invalid_result_count += 1
                    continue
                if w_key in unique_windows:
                    # A retry/duplicate for the same OOT interval must not
                    # inflate total trades or median samples.
                    invalid_result_count += 1
                    continue
                unique_windows.add(w_key)

                raw_metrics = _value(result, "metrics", {}) or {}
                result_metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
                res_pnl = _value(
                    window_data,
                    "net_pnl",
                    window_data.get("net_profit", window_data.get("pnl")),
                )
                if res_pnl is None:
                    res_pnl = _value(result, "net_pnl", result_metrics.get("net_pnl"))
                res_trades = _value(
                    window_data,
                    "trade_count",
                    window_data.get("n_trades"),
                )
                if res_trades is None:
                    res_trades = _value(result, "trade_count", result_metrics.get("n_trades"))
                res_dd = _value(
                    window_data,
                    "max_drawdown",
                    window_data.get("max_drawdown_usdc"),
                )
                if res_dd is None:
                    res_dd = _value(result, "max_drawdown", result_metrics.get("max_drawdown"))
                pnl_num = _finite(res_pnl)
                trades_num = _finite(res_trades)
                dd_num = _finite(res_dd) if res_dd is not None else None
                if res_pnl is None or pnl_num is None:
                    invalid_result_count += 1
                if (
                    res_trades is None
                    or trades_num is None
                    or trades_num < 0
                    or not trades_num.is_integer()
                ):
                    invalid_result_count += 1
                if res_dd is not None and dd_num is None:
                    invalid_result_count += 1

                if pnl_num is not None:
                    pnl_values.append(pnl_num)
                if (
                    trades_num is not None
                    and trades_num >= 0
                    and trades_num.is_integer()
                ):
                    trade_values.append(trades_num)
                if dd_num is not None:
                    drawdown_values.append(dd_num)

                window_details.append(
                    {
                        "window_key": f"{w_key[0]}:{w_key[1]}",
                        "net_pnl": pnl_num,
                        "trade_count": (
                            int(trades_num)
                            if trades_num is not None
                            and trades_num >= 0
                            and trades_num.is_integer()
                            else 0
                        ),
                        "max_drawdown": dd_num,
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
            "invalid_result_count": invalid_result_count,
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
            row, min_trades=min_trades, min_windows=min_windows, mode=mode
        )
        row["eligible_for_shadow"] = gate_res["eligible"]
        row["recommendation_status"] = gate_res["recommendation_status"]
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
        status = winner.get("recommendation_status", "READY_FOR_SHADOW")
        rejection_reasons: list[str] = []
        reason = (
            "Highest median Polymarket-OOT net PnL among candidates meeting the "
            "minimum trade count and window criteria; verify in SHADOW before any human activation."
        )
    elif rows:
        rejection_reasons = list(
            dict.fromkeys(
                reason
                for row in rows
                for reason in row.get("rejection_reasons", [])
            )
        )
        status = rejection_reasons[0] if rejection_reasons else "REJECTED"
        reason = (
            f"No candidate met the criteria for SHADOW promotion ({status})."
        )
    else:
        status = "NO_PNL_SAMPLE"
        rejection_reasons = ["NO_PNL_SAMPLE"]
        reason = "No OOT evaluations found for any configuration."

    summary = {
        "status": status,
        "recommended_config_id": winner["config_id"] if winner else None,
        "recommendation_status": status,
        "reason": reason,
        "rejection_reasons": rejection_reasons,
        "evaluated_config_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "rows": rows,
    }
    if winner:
        summary.update(
            {
                "window_count": winner["window_count"],
                "total_trades": winner["total_trades"],
                "median_pnl": winner["median_oot_pnl"],
                "median_drawdown": winner["median_oot_drawdown"],
            }
        )
    return summary


async def plan_run(
    session: AsyncSession,
    run_id: int,
    config_ids: Sequence[int],
) -> list[AIRunStep]:
    """Populate an optimization run with plan steps for the given configs."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.status in {"ACTIVE", "SHADOW", "FAILED", "REJECTED", "CANCELLED", "ROLLED_BACK"}:
        raise AILabError(
            f"cannot add plan steps to run {run_id} in terminal or active state "
            f"{run.status}"
        )
    await authorize_run_action(session, run_id, "CREATE_EXPERIMENT")

    cleaned_ids = [
        int(config_id) for config_id in config_ids if config_id is not None
    ]
    if not cleaned_ids:
        raise AILabError("at least one config_id is required to plan a run")

    for config_id in cleaned_ids:
        if await session.get(AIExperimentConfig, config_id) is None:
            raise AILabError(f"experiment config {config_id} not found")

    planned_steps = default_plan_steps(cleaned_ids)
    existing_indices = (
        await session.execute(
            select(AIRunStep.step_index).where(AIRunStep.run_id == run_id)
        )
    ).scalars().all()
    step_offset = max(existing_indices) + 1 if existing_indices else 0
    created_steps: list[AIRunStep] = []
    for step_data in planned_steps:
        step = AIRunStep(
            run_id=run_id,
            step_index=step_offset + step_data["step_index"],
            step_type=step_data["step_type"],
            action=step_data["action"],
            status="PENDING",
            input_payload={"config_id": step_data["config_id"]},
            created_at=utc_now(),
        )
        session.add(step)
        created_steps.append(step)

    await transition_run(
        session, run, "PLANNING", reason="planned experiment steps generated"
    )
    await session.flush()
    return created_steps


async def claim_next_step(
    session: AsyncSession,
    run_id: int,
) -> AIRunStep | None:
    """Acquire the next unassigned task step in order.

    Locks the row for update with skip_locked so worker processes do not
    collide on the same step.
    """
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")

    stmt = (
        select(AIRunStep)
        .where(
            AIRunStep.run_id == run_id,
            AIRunStep.status == "PENDING",
        )
        .order_by(AIRunStep.step_index.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    step = (await session.execute(stmt)).scalar_one_or_none()
    if step is None:
        return None

    await authorize_run_action(session, run_id, step.action or "CLAIM_STEP")
    step.status = "RUNNING"
    step.started_at = utc_now()
    if run.status == "PLANNING":
        await transition_run(
            session, run, "RUNNING", reason=f"claimed step {step.step_index}"
        )
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
        summary=summary,
        error_code=error_code,
        error_message=error_message,
        created_at=utc_now(),
    )
    session.add(result)
    await session.flush()
    if step is not None and status in RESULT_CLOSING_STATUSES:
        step.status = status if status in {"SUCCEEDED", "FAILED"} else "SKIPPED"
        step.finished_at = utc_now()
        step.output_payload = {
            "result_id": result.id,
            "status": status,
            "evaluation_kind": kind,
        }
        if summary:
            step.summary = summary
        if error_code:
            step.error_code = error_code
        if error_message:
            step.error_message = error_message
        await session.flush()
    return result


async def evaluate_run(
    session: AsyncSession,
    run_id: int,
) -> dict[str, Any]:
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    await authorize_run_action(session, run_id, "RUN_OOT_BACKTEST")

    results = (
        await session.execute(
            select(ExperimentResult).where(ExperimentResult.run_id == run_id)
        )
    ).scalars().all()
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
    if run.status != "EVALUATING":
        await transition_run(
            session, run, "EVALUATING", reason="evaluation report generated"
        )
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
    """Register the candidate artifact for real-time passive shadow evaluation.

    Shadow evaluation records parallel inference without mutating open orders
    or live trade execution.
    """
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    await authorize_run_action(session, run_id, "PROMOTE_TO_SHADOW")

    if await session.get(AIModelArtifact, candidate_artifact_id) is None:
        raise AILabError(f"candidate artifact {candidate_artifact_id} not found")
    if baseline_artifact_id is not None and await session.get(AIModelArtifact, baseline_artifact_id) is None:
        raise AILabError(f"baseline artifact {baseline_artifact_id} not found")

    normalized_asset = asset.strip()
    if not normalized_asset:
        raise AILabError("asset is required for shadow assignment")
    normalized_regime = regime.strip() if regime else None

    assignment = AIShadowAssignment(
        run_id=run_id,
        candidate_artifact_id=candidate_artifact_id,
        baseline_artifact_id=baseline_artifact_id,
        asset=normalized_asset,
        regime=normalized_regime,
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
    if auto_shadow and report["recommendation_status"] in {"READY_FOR_SHADOW", "RESEARCH_PROVISIONAL"}:
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
                    "status": report.get("recommendation_status"),
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
