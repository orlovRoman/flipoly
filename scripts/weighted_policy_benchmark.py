"""Run a read-only weighted-policy benchmark from JSON or PostgreSQL.

The PostgreSQL path exports one row per market_id from Decision Funnel and
joins quotes/outcomes without writing to the database.  Use --input for a
repeatable fixture or a previously exported dataset.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from polyflip.trading.policy_artifact import save_policy_artifact
from polyflip.trading.weighted_benchmark import (
    BenchmarkConfig,
    MarketObservation,
    benchmark,
    create_policy_artifact_from_benchmark,
)
from polyflip.trading.weighted_policy import WeightedPolicyConfig


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _expr(columns: set[str], alias: str, names: Iterable[str], fallback: str = "NULL") -> str:
    for name in names:
        if name in columns:
            return f"{alias}.{name}"
    return fallback


async def _columns(connection, table_name: str) -> set[str]:
    result = await connection.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = :table_name"
        ),
        {"table_name": table_name},
    )
    return {str(row[0]) for row in result.fetchall()}


async def _fetch_funnel_rows(connection, days: int) -> list[dict[str, Any]]:
    funnel = await _columns(connection, "decision_funnel_log")
    snapshots = await _columns(connection, "market_snapshots")
    markets = await _columns(connection, "live_markets")
    if not {"market_id", "created_at"}.issubset(funnel):
        return []

    f = lambda names, fallback="NULL": _expr(funnel, "f", names, fallback)
    s = lambda names, fallback="NULL": _expr(snapshots, "q", names, fallback)
    m = lambda names, fallback="NULL": _expr(markets, "lm", names, fallback)
    final_expr = m(["final_outcome"])
    quote_join = ""
    outcome_join = ""
    quote_yes = "NULL"
    quote_no = "NULL"
    spread = "0.0"
    if "market_id" in snapshots:
        snapshot_time = _expr(snapshots, "q", ["recorded_at", "market_timestamp"])
        quote_columns = [
            f"{s(['poly_up_best_ask'])} AS snapshot_yes_ask",
            f"{s(['poly_down_best_ask'])} AS snapshot_no_ask",
            f"{s(['spread'])} AS snapshot_spread",
        ]
        quote_join = (
            " LEFT JOIN LATERAL (SELECT "
            + ", ".join(quote_columns)
            + " FROM market_snapshots q WHERE q.market_id = f.market_id"
        )
        if snapshot_time != "NULL":
            quote_join += f" AND {snapshot_time} <= f.created_at"
            quote_join += f" ORDER BY {snapshot_time} DESC"
        quote_join += " LIMIT 1) quote ON TRUE"
        quote_yes = "quote.snapshot_yes_ask"
        quote_no = "quote.snapshot_no_ask"
        spread = "COALESCE(quote.snapshot_spread, 0.0)"
        if "final_outcome" in snapshots:
            outcome_time = _expr(snapshots, "o", ["recorded_at", "market_timestamp"])
            outcome_join = (
                " LEFT JOIN LATERAL (SELECT o.final_outcome AS resolved_outcome "
                "FROM market_snapshots o WHERE o.market_id = f.market_id "
                "AND o.final_outcome IN ('YES', 'NO')"
            )
            if outcome_time != "NULL":
                outcome_join += f" ORDER BY {outcome_time} DESC"
            outcome_join += " LIMIT 1) outcome ON TRUE"
            final_expr = "COALESCE(" + final_expr + ", outcome.resolved_outcome)"
    candidate_yes = (
        f"CASE WHEN {f(['candidate_side'])} = 'BUY_YES' "
        f"THEN {f(['candidate_ask'])} ELSE NULL END"
    )
    candidate_no = (
        f"CASE WHEN {f(['candidate_side'])} = 'BUY_NO' "
        f"THEN {f(['candidate_ask'])} ELSE NULL END"
    )
    yes_ask = f"COALESCE({quote_yes}, {candidate_yes})"
    no_ask = f"COALESCE({quote_no}, {candidate_no})"
    p_market_raw = f(["p_market_yes", "weighted_p_market_yes"])
    market_fallback = (
        "CASE "
        f"WHEN {yes_ask} IS NOT NULL AND {no_ask} IS NOT NULL "
        f"THEN {yes_ask} / NULLIF(({yes_ask} + {no_ask}), 0) "
        f"WHEN {yes_ask} IS NOT NULL THEN {yes_ask} "
        f"WHEN {no_ask} IS NOT NULL THEN 1.0 - {no_ask} "
        "ELSE NULL END"
    )
    p_market = f"COALESCE({p_market_raw}, {market_fallback})"
    p_logreg = f(["p_logreg_yes", "weighted_p_logreg_yes", "p_logreg_win"])
    p_lgbm = f(["p_lgbm_yes", "weighted_p_lgbm_yes"])
    mrf = f(["weighted_mrf_evidence", "mrf_regime_evidence"])
    fee_rate = f(["weighted_fee_rate", "fee_rate"])
    fee_exponent = f(["weighted_fee_exponent", "fee_exponent"])
    fee_source = f(["weighted_fee_source", "fee_source"], "'CONFIG_DEFAULT'")
    execution_role = f(
        ["weighted_execution_role", "execution_role", "trade_role"],
        "'TAKER'",
    )
    observed_cost = f(
        ["weighted_cost_per_share", "observed_cost_per_share", "cost_per_share"]
    )
    phase = f(
        ["entry_model_phase", "phase", "market_phase", "mrf_phase"]
    )
    asset_phase = f(["mrf_asset_phase", "asset_phase"])
    time_left = f(["time_left_sec", "time_left_seconds"])
    if time_left == "NULL" and "time_left_min" in funnel:
        time_left = f"({f(['time_left_min'])} * 60.0)"
    group = f(["group", "market_group"], f(["asset"], "''"))
    horizon = f(["horizon", "market_horizon", "timeframe"])
    role = f(["market_role"], "CASE "
              f"WHEN {f(['candidate_ask'])} < 0.5 THEN 'OUTSIDER' "
              f"WHEN {f(['candidate_ask'])} IS NOT NULL THEN 'FAVORITE' ELSE NULL END")
    horizon_key = f"COALESCE(CAST({horizon} AS TEXT), '')"
    actionable_order = (
        f"CASE WHEN {f(['candidate_side'])} IN ('BUY_YES', 'BUY_NO') "
        f"AND {f(['candidate_ask'])} IS NOT NULL THEN 0 ELSE 1 END"
    )
    query = text(
        f"SELECT DISTINCT ON (f.market_id, {horizon_key}) "
        f"{f(['market_id'])} AS market_id, {f(['created_at'])} AS timestamp, "
        f"{f(['asset'])} AS asset, {yes_ask} AS yes_ask, {no_ask} AS no_ask, "
        f"{final_expr} AS outcome_yes, {p_market} AS p_market_yes, "
        f"{p_logreg} AS p_logreg_yes, {p_lgbm} AS p_lgbm_yes, "
        f"{mrf} AS mrf_evidence, {spread} AS spread, {role} AS market_role, "
        f"{fee_rate} AS fee_rate, {fee_exponent} AS fee_exponent, "
        f"{fee_source} AS fee_source, {execution_role} AS execution_role, "
        f"{observed_cost} AS observed_cost_per_share, {phase} AS phase, "
        f"{asset_phase} AS asset_phase, {time_left} AS time_left_sec, "
        f"{group} AS \"group\", {horizon} AS horizon, "
        f"{f(['candidate_side'])} AS candidate_side, "
        f"{f(['strategy_type'])} AS strategy_type, "
        f"{f(['final_action'])} AS legacy_action, "
        f"{f(['candidate_ask'])} AS legacy_ask "
        "FROM decision_funnel_log f "
        + quote_join
        + outcome_join
        + " LEFT JOIN live_markets lm ON lm.market_id = f.market_id "
        "WHERE f.created_at >= now() - (:days * interval '1 day') "
        f"ORDER BY f.market_id, {horizon_key}, {actionable_order}, f.created_at ASC, f.id ASC"
    )
    result = await connection.execute(query, {"days": max(1, int(days))})
    return [dict(row._mapping) for row in result.fetchall()]


async def _fetch_trade_rows(connection, days: int) -> list[dict[str, Any]]:
    columns = await _columns(connection, "trade_history")
    if not {"market_id", "created_at"}.issubset(columns):
        return []
    selected = [
        "market_id", "created_at", "asset", "outcome_bought", "executed_price",
        "settlement_outcome", "pnl", "fee", "p_candidate_win", "p_logreg_win",
        "p_market_yes", "p_logreg_yes", "p_lgbm_yes", "weighted_p_market_yes",
        "weighted_p_logreg_yes", "weighted_p_lgbm_yes", "weighted_mrf_evidence",
        "weighted_cost_per_share", "weighted_fee_rate", "weighted_fee_exponent",
        "weighted_fee_source", "weighted_execution_role",
        "weighted_spread_per_share", "weighted_expected_execution_price",
        "weighted_edge_lower_bound", "weighted_size_multiplier",
        "market_role", "strategy_type", "trade_role",
    ]
    selected = [name for name in selected if name in columns]
    result = await connection.execute(
        text(
            "SELECT " + ", ".join(selected) + " FROM trade_history "
            "WHERE created_at >= now() - (:days * interval '1 day') "
            "AND outcome_bought IN ('YES', 'NO') "
            "ORDER BY created_at ASC"
        ),
        {"days": max(1, int(days))},
    )
    rows = []
    for row in result.fetchall():
        item = dict(row._mapping)
        price = item.get("executed_price")
        side = str(item.get("outcome_bought") or "").upper()
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        item["yes_ask"] = price if side == "YES" else 1.0 - price
        item["no_ask"] = 1.0 - price if side == "YES" else price
        item["timestamp"] = item.get("created_at")
        item["fee_rate"] = item.get("weighted_fee_rate")
        item["fee_exponent"] = item.get("weighted_fee_exponent")
        item["fee_source"] = item.get("weighted_fee_source") or "OBSERVED_TRADE"
        item["execution_role"] = (
            item.get("weighted_execution_role")
            or item.get("trade_role")
            or "TAKER"
        )
        item["observed_cost_per_share"] = item.get("weighted_cost_per_share")
        item["spread"] = item.get("weighted_spread_per_share") or 0.0
        item["outcome_yes"] = item.get("settlement_outcome")
        item["legacy_action"] = "BUY_YES" if side == "YES" else "BUY_NO"
        item["legacy_ask"] = price
        item["group"] = str(item.get("asset") or "")
        rows.append(item)
    return rows


async def load_from_database(database_url: str, days: int) -> tuple[list[dict[str, Any]], str]:
    if database_url.startswith("postgresql://"):
        database_url = "postgresql+asyncpg://" + database_url[len("postgresql://"):]
    engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            rows = await _fetch_funnel_rows(connection, days)
            if rows:
                return rows, "decision_funnel_log"
            return await _fetch_trade_rows(connection, days), "trade_history"
    finally:
        await engine.dispose()


def _policy_config() -> WeightedPolicyConfig:
    def number(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except (TypeError, ValueError):
            return default
    return WeightedPolicyConfig(
        market_weight=number("WEIGHTED_MARKET_WEIGHT", 0.90),
        logreg_weight=number("WEIGHTED_LOGREG_WEIGHT", 0.05),
        lgbm_weight=number("WEIGHTED_LGBM_WEIGHT", 0.05),
        mrf_beta=number("WEIGHTED_MRF_BETA", 0.0),
        intercept=number("WEIGHTED_INTERCEPT", 0.0),
        fee_rate=number("WEIGHTED_FEE_RATE", 0.07),
        maker_fee_rate=number("WEIGHTED_MAKER_FEE_RATE", 0.0),
        fee_exponent=number("WEIGHTED_FEE_EXPONENT", 1.0),
        slippage_rate=number("WEIGHTED_SLIPPAGE_RATE", 0.005),
        latency_buffer=number("WEIGHTED_LATENCY_BUFFER", 0.0),
        execution_role=os.getenv("WEIGHTED_EXECUTION_ROLE", "TAKER"),
        policy_id=os.getenv("WEIGHTED_POLICY_ID", "UNVERSIONED")[:64],
        mrf_extreme_veto_threshold=number("WEIGHTED_MRF_EXTREME_VETO_THRESHOLD", -1.0),
    )


def _report_hash(report: dict[str, Any]) -> str:
    payload = json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def run(args: argparse.Namespace) -> int:
    source = "json"
    if args.input:
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
        raw_rows = raw.get("observations", raw) if isinstance(raw, dict) else raw
    else:
        database_url = args.database_url or os.getenv("DATABASE_URL")
        if not database_url:
            raise SystemExit("DATABASE_URL or --database-url is required without --input")
        raw_rows, source = await load_from_database(database_url, args.days)
    observations = [MarketObservation.from_mapping(row) for row in raw_rows]
    cfg = BenchmarkConfig(
        policy_config=_policy_config(),
        min_net_ev=args.min_net_ev,
        train_min_rows=args.train_min_rows,
        test_size=args.test_size,
        purge_gap=args.purge_gap,
        ridge_lambda=args.ridge_lambda,
        hierarchical_min_segment_rows=args.hierarchical_min_segment_rows,
        hierarchical_shrinkage=args.hierarchical_shrinkage,
        coefficient_bound=args.coefficient_bound,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    result = benchmark(observations, config=cfg)
    report = result.as_dict()
    report["input"] = {
        "source": source,
        "requested_days": args.days,
        "raw_rows": len(raw_rows),
        "market_observations": len(observations),
        "resolved": sum(item.outcome_yes is not None for item in observations),
    }
    if args.export_observations:
        Path(args.export_observations).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export_observations).write_text(
            json.dumps(
                {"generated_at": datetime.now(timezone.utc).isoformat(),
                 "observations": [item.as_dict() for item in observations]},
                ensure_ascii=False, indent=2, default=_json_default,
            ) + "\n",
            encoding="utf-8",
        )
    if args.artifact:
        selected_tuning = {
            str(item["parameter"]): item.get("selected")
            for item in result.tuning
            if item.get("parameter") and item.get("selected") is not None
        }
        artifact = create_policy_artifact_from_benchmark(
            observations,
            result,
            version=args.policy_version,
            policy_config=cfg.policy_config,
            thresholds={
                "min_net_ev": args.min_net_ev,
                "selected_tuning": selected_tuning,
            },
            source_report_hash=_report_hash(report),
        )
        save_policy_artifact(args.artifact, artifact)
        report["policy_artifact_id"] = artifact.artifact_id
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", help="JSON fixture/export instead of PostgreSQL")
    result.add_argument("--database-url")
    result.add_argument("--days", type=int, default=30)
    result.add_argument("--output")
    result.add_argument("--export-observations")
    result.add_argument("--artifact")
    result.add_argument("--policy-version", default="weighted-policy-v1")
    result.add_argument("--train-min-rows", type=int, default=300)
    result.add_argument("--test-size", type=int, default=100)
    result.add_argument("--purge-gap", type=int, default=0)
    result.add_argument("--ridge-lambda", type=float, default=1.0)
    result.add_argument("--coefficient-bound", type=float, default=5.0)
    result.add_argument("--min-net-ev", type=float, default=0.0)
    result.add_argument("--bootstrap-iterations", type=int, default=1000)
    result.add_argument("--hierarchical-min-segment-rows", type=int, default=300)
    result.add_argument("--hierarchical-shrinkage", type=float, default=300.0)
    return result


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
