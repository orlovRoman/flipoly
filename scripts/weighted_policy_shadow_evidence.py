#!/usr/bin/env python3
"""Collect read-only evidence for weighted-policy shadow activation."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARM_NAMES = (
    "MARKET_ONLY",
    "MARKET_LOGREG",
    "MARKET_LGBM",
    "FULL_WEIGHTED",
    "FULL_WEIGHTED_MRF",
    "OUTSIDER_AGREE_ONLY",
    "LEGACY",
)


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        result = value
    else:
        text_value = str(value or "")
        if not text_value:
            return None
        if text_value.endswith("Z"):
            text_value = text_value[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text_value)
        except (TypeError, ValueError):
            return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _outcome_yes(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) in (0.0, 1.0):
            return bool(value)
    text_value = str(value or "").strip().upper()
    if text_value in {"YES", "UP", "TRUE", "1", "WIN", "WON"}:
        return True
    if text_value in {"NO", "DOWN", "FALSE", "0", "LOSS", "LOST"}:
        return False
    return None


def _side(value: Any) -> Optional[str]:
    text_value = str(value or "").strip().upper()
    if text_value in {"BUY_YES", "YES", "UP"}:
        return "BUY_YES"
    if text_value in {"BUY_NO", "NO", "DOWN"}:
        return "BUY_NO"
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _legacy_probability_yes(row: Mapping[str, Any]) -> Optional[float]:
    """Return legacy LogReg probability on the canonical YES axis."""
    explicit = _float(row.get("p_legacy_yes"))
    if explicit is not None:
        return explicit
    explicit = _float(row.get("p_logreg_yes"))
    if explicit is not None:
        return explicit
    candidate_win = _float(row.get("p_logreg_win"))
    side = _side(row.get("candidate_side"))
    if candidate_win is None or side is None:
        return None
    return candidate_win if side == "BUY_YES" else 1.0 - candidate_win


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("weighted_benchmark_json")
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _ask_for_side(
    row: Mapping[str, Any],
    side: str,
    summary: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    if summary:
        if "policy_eligible" in summary and _bool(summary.get("policy_eligible")):
            value = _float(summary.get("policy_selected_ask"))
            if value is not None:
                return value
        value = _float(summary.get("selected_ask"))
        if value is not None:
            return value
        side_key = "yes_ask" if side == "BUY_YES" else "no_ask"
        value = _float(summary.get(side_key))
        if value is not None:
            return value
    for key in (
        "legacy_ask",
        "candidate_ask",
        "weighted_expected_execution_price",
    ):
        if key in row and key in {"legacy_ask", "candidate_ask"}:
            value = _float(row.get(key))
            if value is not None:
                return value
    key = "yes_ask" if side == "BUY_YES" else "no_ask"
    return _float(row.get(key))


def _pnl(
    row: Mapping[str, Any],
    side: Optional[str],
    outcome_yes: Optional[bool],
    summary: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    if side is None or outcome_yes is None:
        return None
    ask = _ask_for_side(row, side, summary)
    if ask is None or not 0.0 < ask < 1.0:
        return None
    cost = None
    if summary:
        if "policy_eligible" in summary and _bool(summary.get("policy_eligible")):
            cost = _float(summary.get("policy_selected_cost_per_share"))
        if cost is None:
            cost = _float(summary.get("selected_cost_per_share"))
        if cost is None:
            side_key = (
                "yes_cost_per_share" if side == "BUY_YES" else "no_cost_per_share"
            )
            cost = _float(summary.get(side_key))
    if cost is None:
        cost = _float(row.get("observed_cost_per_share"))
    if cost is None:
        cost = _float(row.get("weighted_cost_per_share")) or 0.0
    won = outcome_yes if side == "BUY_YES" else not outcome_yes
    return (1.0 if won else 0.0) - ask - max(0.0, cost)


def _cluster_ci_lower(
    evaluations: Iterable[tuple[str, datetime, float]],
    *,
    iterations: int = 1000,
    seed: int = 20260902,
) -> Optional[float]:
    grouped: dict[str, list[float]] = {}
    for market_id, timestamp, pnl in evaluations:
        cluster = f"{market_id}|{timestamp.date().isoformat()}"
        grouped.setdefault(cluster, []).append(float(pnl))
    if not grouped:
        return None
    keys = list(grouped)
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [
            sum(sum(grouped[keys[index]]) for index in rng.integers(0, len(keys), len(keys)))
            for _ in range(max(1, int(iterations)))
        ],
        dtype=float,
    )
    return round(float(np.quantile(samples, 0.025)), 10)


def _mean(values: list[float]) -> Optional[float]:
    return round(float(np.mean(values)), 10) if values else None


def summarize_shadow_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize persisted shadow telemetry without mutating the database."""
    resolved_markets: set[str] = set()
    timestamps = [
        timestamp
        for timestamp in (_dt(row.get("created_at", row.get("timestamp")) ) for row in rows)
        if timestamp is not None
    ]
    arm_predictions: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    arm_outcomes: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    arm_pnls: dict[str, list[tuple[str, datetime, float]]] = {
        name: [] for name in ARM_NAMES
    }
    candidate_trades = 0
    raw_candidate_trades = 0
    telemetry_rows = 0
    arm_coverage = {name: 0 for name in ARM_NAMES}
    policy_ids = sorted(
        {
            str(row.get("weighted_policy_id") or "").strip()
            for row in rows
            if str(row.get("weighted_policy_id") or "").strip()
        }
    )
    for row in rows:
        outcome_yes = _outcome_yes(row.get("outcome_yes"))
        market_id = str(row.get("market_id") or "")
        timestamp = _dt(row.get("created_at", row.get("timestamp"))) or datetime.min.replace(
            tzinfo=timezone.utc
        )
        if outcome_yes is not None and market_id:
            resolved_markets.add(market_id)
        payload = _payload(row)
        if payload:
            telemetry_rows += 1
        for name in ARM_NAMES:
            summary = payload.get(name)
            if isinstance(summary, Mapping):
                arm_coverage[name] += 1
            else:
                summary = None
            if name == "MARKET_ONLY":
                probability = _float(row.get("weighted_p_market_yes"))
                if probability is None:
                    probability = _float(row.get("p_market_yes"))
            elif name == "LEGACY":
                probability = _legacy_probability_yes(row)
            elif name == "FULL_WEIGHTED_MRF":
                probability = _float(row.get("weighted_p_final_yes"))
            else:
                probability = None
            if summary:
                probability = _float(summary.get("p_final_yes")) or probability
            if probability is not None and outcome_yes is not None:
                arm_predictions[name].append(probability)
                arm_outcomes[name].append(float(outcome_yes))
            selected_side = _side(
                summary.get("selected_side") if summary else None
            )
            if name == "LEGACY":
                selected_side = _side(
                    row.get("legacy_action", row.get("final_action", row.get("candidate_side")))
                )
            if name == "FULL_WEIGHTED_MRF":
                raw_selected_side = selected_side or _side(row.get("weighted_selected_side"))
                if raw_selected_side is not None:
                    raw_candidate_trades += 1
                if summary and "policy_eligible" in summary:
                    selected_side = (
                        _side(summary.get("policy_selected_side"))
                        if _bool(summary.get("policy_eligible"))
                        else None
                    )
                elif str(row.get("weighted_policy_id") or "").strip():
                    # An identified policy without eligibility telemetry is not
                    # enough to claim an executable candidate.
                    selected_side = None
                else:
                    # Backwards-compatible handling for unversioned/manual
                    # fixtures created before policy eligibility was persisted.
                    selected_side = raw_selected_side
                if selected_side is not None:
                    candidate_trades += 1
            pnl = _pnl(row, selected_side, outcome_yes, summary)
            if pnl is not None:
                arm_pnls[name].append((market_id, timestamp, pnl))

    def arm_result(name: str) -> dict[str, Any]:
        predictions = arm_predictions[name]
        outcomes = arm_outcomes[name]
        pnls = arm_pnls[name]
        brier = (
            _mean([(prediction - outcome) ** 2 for prediction, outcome in zip(predictions, outcomes)])
            if predictions
            else None
        )
        net_pnl = round(sum(item[2] for item in pnls), 10) if pnls else None
        return {
            "observations": len(predictions),
            "trades": len(pnls),
            "brier": brier,
            "net_pnl": net_pnl,
            "pnl_ci_lower": _cluster_ci_lower(pnls),
        }

    weighted_predictions = arm_predictions["FULL_WEIGHTED_MRF"]
    weighted_outcomes = arm_outcomes["FULL_WEIGHTED_MRF"]
    calibration_error = (
        round(abs(float(np.mean(weighted_predictions)) - float(np.mean(weighted_outcomes))), 10)
        if weighted_predictions
        else None
    )
    duration_days = (
        round((max(timestamps) - min(timestamps)).total_seconds() / 86400.0, 10)
        if len(timestamps) >= 2
        else 0.0
    )
    arms = {name: arm_result(name) for name in ARM_NAMES}
    return {
        "policy_id": policy_ids[0] if len(policy_ids) == 1 else None,
        "policy_ids": policy_ids,
        "shadow_days": duration_days,
        "shadow_resolved_markets": len(resolved_markets),
        "shadow_candidate_trades": candidate_trades,
        "shadow_raw_candidate_trades": raw_candidate_trades,
        "pnl_ci_lower": arms["FULL_WEIGHTED_MRF"]["pnl_ci_lower"],
        "weighted_brier": arms["FULL_WEIGHTED_MRF"]["brier"],
        "market_brier": arms["MARKET_ONLY"]["brier"],
        "legacy_brier": arms["LEGACY"]["brier"],
        "weighted_net_pnl": arms["FULL_WEIGHTED_MRF"]["net_pnl"],
        "market_net_pnl": arms["MARKET_ONLY"]["net_pnl"],
        "legacy_net_pnl": arms["LEGACY"]["net_pnl"],
        "calibration_error": calibration_error,
        "telemetry": {
            "rows": len(rows),
            "rows_with_benchmark": telemetry_rows,
            "arm_coverage": arm_coverage,
            "arms": arms,
        },
    }


def summarize_live_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    drags: list[float] = []
    policy_ids = sorted(
        {
            str(row.get("weighted_policy_id") or "").strip()
            for row in rows
            if str(row.get("weighted_policy_id") or "").strip()
        }
    )
    for row in rows:
        expected = _float(row.get("weighted_expected_execution_price"))
        realized = _float(row.get("executed_price"))
        if expected is not None and realized is not None:
            drags.append(abs(realized - expected))
    return {
        "policy_id": policy_ids[0] if len(policy_ids) == 1 else None,
        "policy_ids": policy_ids,
        "live_fills": len(rows),
        "execution_drag": _mean(drags),
        "expected_realized_samples": len(drags),
    }


async def _columns(connection, table_name: str) -> set[str]:
    result = await connection.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = :table_name"
        ),
        {"table_name": table_name},
    )
    return {str(row[0]) for row in result.fetchall()}


def _expr(columns: set[str], alias: str, names: Iterable[str], fallback: str = "NULL") -> str:
    """Build a NULL-tolerant compatibility expression for exported columns."""
    expressions = [f"{alias}.{name}" for name in names if name in columns]
    if not expressions:
        return fallback
    if fallback == "NULL":
        return expressions[0] if len(expressions) == 1 else f"COALESCE({', '.join(expressions)})"
    return f"COALESCE({', '.join((*expressions, fallback))})"


async def _fetch_shadow_rows(
    connection,
    days: int,
    policy_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    requested_policy_id = str(policy_id or "").strip()
    funnel = await _columns(connection, "decision_funnel_log")
    markets = await _columns(connection, "live_markets")
    if not {"market_id", "created_at"}.issubset(funnel):
        return []
    if not {"weighted_policy_mode", "weighted_benchmark_json"}.issubset(funnel):
        return []
    # A requested policy cannot be verified when the persisted identity column
    # is absent. Return no rows so activation fails closed with missing evidence
    # instead of silently mixing historical policies.
    if requested_policy_id and "weighted_policy_id" not in funnel:
        return []
    f = lambda names, fallback="NULL": _expr(funnel, "d", names, fallback)
    outcome_join = ""
    outcome = "NULL"
    if {"market_id", "final_outcome"}.issubset(markets):
        outcome_join = " LEFT JOIN live_markets lm ON lm.market_id = d.market_id "
        outcome = "lm.final_outcome"
    selected = [
        f(["market_id"]) + " AS market_id",
        f(["created_at", "timestamp"]) + " AS created_at",
        f(["weighted_policy_mode"]) + " AS weighted_policy_mode",
        f(["weighted_policy_id"]) + " AS weighted_policy_id",
        f(["execution_mode"]) + " AS execution_mode",
        f(["asset"]) + " AS asset",
        f(["candidate_side"]) + " AS candidate_side",
        f(["candidate_ask"]) + " AS candidate_ask",
        f(["final_action"]) + " AS final_action",
        f(["p_market_yes", "weighted_p_market_yes"]) + " AS p_market_yes",
        f(["p_logreg_yes", "weighted_p_logreg_yes"]) + " AS p_logreg_yes",
        f(["p_logreg_win"]) + " AS p_logreg_win",
        f(["p_lgbm_yes", "weighted_p_lgbm_yes"]) + " AS p_lgbm_yes",
        f(["weighted_p_market_yes"]) + " AS weighted_p_market_yes",
        f(["weighted_p_final_yes"]) + " AS weighted_p_final_yes",
        f(["weighted_selected_side"]) + " AS weighted_selected_side",
        f(["weighted_cost_per_share"]) + " AS weighted_cost_per_share",
        f(["weighted_expected_execution_price"]) + " AS weighted_expected_execution_price",
        f(["weighted_benchmark_json"]) + " AS weighted_benchmark_json",
        f(["weighted_net_ev_per_share"]) + " AS weighted_net_ev_per_share",
        f(["weighted_min_net_ev"]) + " AS weighted_min_net_ev",
        f(["mrf_asset_phase"]) + " AS mrf_asset_phase",
        outcome + " AS outcome_yes",
    ]
    mode_conditions = [
        "d.weighted_policy_mode IN ('WEIGHTED_SHADOW', 'SHADOW')",
        "d.execution_mode IN ('SHADOW', 'PAPER')",
    ]
    available_modes = [
        condition
        for condition in mode_conditions
        if (
            condition.startswith("d.weighted_policy_mode") and "weighted_policy_mode" in funnel
        )
        or (
            condition.startswith("d.execution_mode") and "execution_mode" in funnel
        )
    ]
    if not available_modes:
        return []
    policy_condition = ""
    params: dict[str, Any] = {"days": max(1, int(days))}
    if requested_policy_id:
        policy_condition = " AND d.weighted_policy_id = :policy_id"
        params["policy_id"] = requested_policy_id
    query = text(
        "SELECT "
        + ", ".join(selected)
        + " FROM decision_funnel_log d "
        + outcome_join
        + " WHERE d.created_at >= now() - (:days * interval '1 day') "
        + policy_condition
        + " AND ("
        + " OR ".join(available_modes)
        + ") ORDER BY d.created_at ASC"
    )
    result = await connection.execute(query, params)
    return [dict(row._mapping) for row in result.fetchall()]


async def _fetch_live_rows(
    connection,
    days: int,
    policy_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    requested_policy_id = str(policy_id or "").strip()
    trades = await _columns(connection, "trade_history")
    required = {"market_id", "created_at", "mode", "executed_price"}
    if not required.issubset(trades) or "weighted_policy_mode" not in trades:
        return []
    if requested_policy_id and "weighted_policy_id" not in trades:
        return []
    t = lambda names, fallback="NULL": _expr(trades, "t", names, fallback)
    status_condition = (
        "(t.status IS NULL OR t.status IN ('FILLED', 'PARTIAL'))"
        if "status" in trades
        else "TRUE"
    )
    query = text(
        "SELECT "
        + ", ".join(
            [
                t(["market_id"]) + " AS market_id",
                t(["created_at", "timestamp"]) + " AS created_at",
                t(["mode"]) + " AS mode",
                t(["status"]) + " AS status",
                t(["weighted_policy_mode"]) + " AS weighted_policy_mode",
                t(["weighted_policy_id"]) + " AS weighted_policy_id",
                t(["weighted_expected_execution_price"]) + " AS weighted_expected_execution_price",
                t(["executed_price"]) + " AS executed_price",
                t(["weighted_p_final_yes"]) + " AS weighted_p_final_yes",
                t(["weighted_net_ev_per_share"]) + " AS weighted_net_ev_per_share",
                t(["pnl"]) + " AS pnl",
            ]
        )
        + " FROM trade_history t WHERE t.created_at >= now() - (:days * interval '1 day') "
        + " AND t.mode = 'LIVE' "
        + " AND t.weighted_policy_mode = 'WEIGHTED_ACTIVE' "
        + (" AND t.weighted_policy_id = :policy_id " if requested_policy_id else "")
        + " AND " + status_condition
        + " ORDER BY t.created_at ASC"
    )
    params: dict[str, Any] = {"days": max(1, int(days))}
    if requested_policy_id:
        params["policy_id"] = requested_policy_id
    result = await connection.execute(query, params)
    return [dict(row._mapping) for row in result.fetchall()]


async def collect(
    database_url: str,
    days: int,
    repeat_oot_reports: int,
    policy_id: Optional[str] = None,
) -> dict[str, Any]:
    requested_policy_id = str(policy_id or "").strip() or None
    url = database_url
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    engine = create_async_engine(url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            shadow_rows = await _fetch_shadow_rows(
                connection, days, policy_id=requested_policy_id
            )
            live_rows = await _fetch_live_rows(
                connection, days, policy_id=requested_policy_id
            )
    finally:
        await engine.dispose()
    shadow = summarize_shadow_rows(shadow_rows)
    live = summarize_live_rows(live_rows)
    evidence = {
        key: shadow.get(key)
        for key in (
            "shadow_days",
            "shadow_resolved_markets",
            "shadow_candidate_trades",
            "shadow_raw_candidate_trades",
            "pnl_ci_lower",
            "weighted_brier",
            "market_brier",
            "legacy_brier",
            "weighted_net_pnl",
            "market_net_pnl",
            "legacy_net_pnl",
            "calibration_error",
        )
    }
    observed_policy_ids = sorted(
        set(shadow.get("policy_ids", [])) | set(live.get("policy_ids", []))
    )
    observed_policy_id = observed_policy_ids[0] if len(observed_policy_ids) == 1 else None
    evidence["policy_id"] = observed_policy_id
    evidence["policy_ids"] = observed_policy_ids
    evidence["repeat_oot_reports"] = max(0, int(repeat_oot_reports))
    evidence["live_fills"] = live["live_fills"]
    evidence["execution_drag"] = live["execution_drag"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": max(1, int(days)),
        "policy_id": observed_policy_id,
        "requested_policy_id": requested_policy_id,
        "evidence": evidence,
        "shadow": shadow,
        "live": live,
    }


async def run(args: argparse.Namespace) -> int:
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    payload = await collect(
        database_url,
        args.days,
        args.repeat_oot_reports,
        policy_id=args.policy_id,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--database-url")
    result.add_argument("--days", type=int, default=30)
    result.add_argument("--repeat-oot-reports", type=int, default=0)
    result.add_argument(
        "--policy-id",
        help="restrict evidence to one immutable weighted policy ID",
    )
    result.add_argument("--output", required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
