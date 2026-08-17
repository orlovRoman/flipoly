"""Run the read-only LogReg candidate replay on local CPU workers."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polyflip.crypto.logreg_replay import (
    classification_metrics,
    first_snapshot_per_market,
    split_market_windows,
)
from polyflip.crypto.oof_artifact import deserialize_oof_artifact
from polyflip.crypto.polymarket_backtest import compute_oof_polymarket_backtest
from polyflip.scripts.replay_logreg_candidates import write_reports


COMMIT = "02c9ed7517c70cd4e3ac56fd0b12cdc9a22d8b4c"
PROTOCOL = {
    "protocol_version": "polymarket_logreg_eval_v1",
    "strategy_branch": "COMBINED",
    "min_edge": 0.03,
    "fee_rate": 0.02,
    "slippage_pct": 0.0,
    "cost_buffer": 0.0,
    "stake_usdc": 1.0,
    "min_price": 0.05,
    "max_price": 0.95,
    "outsider_max_price": 0.45,
}
BRANCHES = ("COMBINED", "FAVORITE_ONLY", "OUTSIDER_ONLY")
WINDOWS = ("COMBINED", "T1", "T2", "T3")
_CLOSE_MAP: dict[str, Any] = {}
_SOURCE_BY_MARKET: dict[str, str] = {}


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _close_info(markets: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    close_map: dict[str, Any] = {}
    source_by_market: dict[str, str] = {}
    for row in markets:
        market_id = row.get("market_id")
        if market_id is None:
            continue
        market_id = str(market_id)
        for source, key in (
            ("end_date", "end_date"),
            ("resolved_at", "resolved_at"),
            ("end_time_est", "end_time_est"),
        ):
            value = row.get(key)
            if value not in (None, ""):
                close_map[market_id] = value
                source_by_market[market_id] = source
                break
    return close_map, source_by_market


def _initialise_worker(close_map: dict[str, Any], source_by_market: dict[str, str]) -> None:
    global _CLOSE_MAP, _SOURCE_BY_MARKET
    _CLOSE_MAP = close_map
    _SOURCE_BY_MARKET = source_by_market


def _evaluate(
    frame: pd.DataFrame,
    scores: np.ndarray,
    quotes: pd.DataFrame,
) -> tuple[dict[str, Any], int]:
    selected, p_yes = first_snapshot_per_market(frame, scores)
    selected["market_id"] = selected["market_id"].astype(str)
    quote_frame = quotes.copy()
    if not quote_frame.empty:
        quote_frame["market_id"] = quote_frame["market_id"].astype(str)
    partitions = split_market_windows(selected, _CLOSE_MAP)
    output: dict[str, Any] = {}
    for window in WINDOWS:
        market_ids = (
            set(selected["market_id"])
            if window == "COMBINED"
            else partitions[window]
        )
        mask = selected["market_id"].isin(market_ids).to_numpy()
        sub_frame = selected.loc[mask].reset_index(drop=True)
        sub_scores = p_yes[mask]
        sub_quotes = (
            quote_frame[quote_frame["market_id"].isin(market_ids)].reset_index(drop=True)
            if not quote_frame.empty
            else quote_frame
        )
        output[window] = {}
        for branch in BRANCHES:
            result = compute_oof_polymarket_backtest(
                sub_frame,
                sub_scores,
                sub_quotes,
                strategy_branch=branch,
                min_edge=PROTOCOL["min_edge"],
                fee_rate=PROTOCOL["fee_rate"],
                slippage_pct=PROTOCOL["slippage_pct"],
                cost_buffer=PROTOCOL["cost_buffer"],
                stake_usdc=PROTOCOL["stake_usdc"],
                min_price=PROTOCOL["min_price"],
                max_price=PROTOCOL["max_price"],
                outsider_max_price=PROTOCOL["outsider_max_price"],
            )
            classification = classification_metrics(sub_frame, sub_scores)
            output[window][branch] = {
                "coverage_pct": result.get("coverage_pct"),
                "n_trades": result.get("n_trades"),
                "win_rate": result.get("win_rate"),
                "net_profit": result.get("net_profit"),
                "roi_pct": result.get("roi_pct"),
                "max_drawdown_usdc": result.get("max_drawdown_usdc"),
                "brier": classification.get("brier"),
                "ece": classification.get("ece"),
                "log_loss": classification.get("log_loss"),
            }
    return output, len(selected)


def _replay_one(item: dict[str, Any]) -> dict[str, Any]:
    blob = base64.b64decode(item["artifact_blob_b64"])
    artifact = deserialize_oof_artifact(blob)
    frame = artifact["frame"]
    quotes = artifact["quotes"]
    raw_eval, market_count = _evaluate(frame, artifact["raw_oof_scores"], quotes)
    platt_eval, _ = _evaluate(frame, artifact["oof_scores"], quotes)
    return {
        "model_registry_id": item["model_registry_id"],
        "model_version": item["model_version"],
        "oof_artifact_id": item["oof_artifact_id"],
        "artifact_schema_version": item["artifact_schema_version"],
        "artifact_row_count": item["artifact_row_count"],
        "artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "artifact_status": "VALID_FOR_REPLAY",
        "invalid_reason": None,
        "market_count": market_count,
        "close_time_recovery": {
            "join_key": "market_id",
            "source_counts": dict(
                Counter(
                    _SOURCE_BY_MARKET.get(str(market_id))
                    for market_id in frame["market_id"].dropna().astype(str).unique()
                    if _SOURCE_BY_MARKET.get(str(market_id)) is not None
                )
            ),
            "missing_count": sum(
                1
                for market_id in frame["market_id"].dropna().astype(str).unique()
                if market_id not in _CLOSE_MAP
            ),
            "ambiguous_count": 0,
        },
        "evaluation_commit": COMMIT,
        "metrics_schema_version": "canonical_pnl_v1",
        "evaluation_protocol_version": PROTOCOL["protocol_version"],
        "backtest_parameters": PROTOCOL,
        "evaluations": {
            "RAW": {"windows": raw_eval},
            "PLATT": {"windows": platt_eval},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    close_map, source_by_market = _close_info(payload["live_markets"])
    workers = args.workers or int(
        os.environ.get("REPLAY_WORKERS", str(min(os.cpu_count() or 2, 6)))
    )
    workers = max(1, min(workers, len(payload["records"])))
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_worker,
        initargs=(close_map, source_by_market),
    ) as pool:
        futures = [pool.submit(_replay_one, item) for item in payload["records"]]
        for index, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if index % 5 == 0 or index == len(futures):
                sys.stderr.write(f"LOCAL_REPLAY_PROGRESS {index}/{len(futures)} workers={workers}\n")
                sys.stderr.flush()
    records.sort(key=lambda item: item["model_registry_id"])
    report = {
        "report_version": "logreg_candidate_replay_v2",
        "candidate_id_range": [820, 879],
        "candidate_count_expected": 60,
        "candidate_count_observed": len(records),
        "read_only": True,
        "retrained": False,
        "evaluation_commit": COMMIT,
        "evaluation_protocol_version": PROTOCOL["protocol_version"],
        "metrics_schema_version": "canonical_pnl_v1",
        "backtest_parameters": PROTOCOL,
        "records": records,
    }
    report_dir = Path(__file__).resolve().parents[2] / "reports"
    write_reports(_native(report), report_dir, overwrite=True)
    sys.stderr.write(f"LOCAL_REPLAY_OK candidates={len(records)} workers={workers}\n")
    return 0


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
