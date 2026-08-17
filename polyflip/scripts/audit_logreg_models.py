"""Stage 4: Read-only LogReg model auditor.

Evaluates active or specified LogReg models against canonical Polymarket OOF
backtest accounting without modifying any database or runtime state.
Computes Brier, ECE, Log Loss, OOT PnL, 3 chronological windows (T1/T2/T3),
COMBINED / FAVORITE / OUTSIDER branches, and directional breakdowns.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import select

from polyflip.crypto.logreg_polymarket_backtest import compute_logreg_polymarket_backtest
from polyflip.crypto.oof_artifact import deserialize_oof_artifact
from polyflip.db.connection import async_session
from polyflip.db.models import ModelRegistry, ModelRegistryOOFArtifact


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_t = y_true[valid].astype(int)
    y_p = y_prob[valid]
    if len(y_t) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total = len(y_t)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return float(ece)


def _compute_drawdown_series(trade_pnls: Sequence[float]) -> tuple[float, float]:
    """Return max drawdown (USDC) and max drawdown (%) from trade PnL sequence."""
    if not trade_pnls:
        return 0.0, 0.0
    cumulative = np.cumsum(trade_pnls)
    running_max = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))
    drawdowns = running_max[1:] - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    return max_dd, 0.0


def _split_chronological_windows(
    frame: pd.DataFrame,
    p_yes: np.ndarray,
    quotes: pd.DataFrame | None,
    strategy_branch: str,
    **backtest_kwargs: Any,
) -> dict[str, Any]:
    """Split canonical evaluated markets into 3 chronological windows T1/T2/T3 by market_close_at."""
    if frame.empty or len(p_yes) != len(frame):
        return {
            "T1": {"status": "EMPTY", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "T2": {"status": "EMPTY", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "T3": {"status": "EMPTY", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "median_pnl": 0.0,
            "non_negative_windows_count": 0,
        }

    working = frame.copy()
    # Determine market_close_at
    close_col = None
    for candidate in ("market_close_at", "resolved_at", "end_time_est", "market_start", "recorded_at"):
        if candidate in working.columns and working[candidate].notna().any():
            close_col = candidate
            break

    if close_col:
        working["_close_time"] = pd.to_datetime(working[close_col], utc=True, errors="coerce")
    else:
        working["_close_time"] = pd.to_datetime(working["recorded_at"], utc=True, errors="coerce")

    working["_p_yes"] = p_yes
    working = working.sort_values("_close_time", kind="stable").reset_index(drop=True)

    n = len(working)
    if n < 3:
        res = compute_logreg_polymarket_backtest(
            working, working["_p_yes"].to_numpy(), quotes, strategy_branch=strategy_branch, **backtest_kwargs
        )
        return {
            "T1": {"status": "SPARSE", "n_markets": n, "n_trades": res.get("n_trades", 0), "total_pnl": res.get("total_pnl", 0.0)},
            "T2": {"status": "SPARSE", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "T3": {"status": "SPARSE", "n_markets": 0, "n_trades": 0, "total_pnl": 0.0},
            "median_pnl": 0.0,
            "non_negative_windows_count": 1 if res.get("total_pnl", 0.0) >= 0 else 0,
        }

    q33 = working["_close_time"].quantile(1.0 / 3.0)
    q67 = working["_close_time"].quantile(2.0 / 3.0)

    w1_mask = working["_close_time"] <= q33
    w2_mask = (working["_close_time"] > q33) & (working["_close_time"] <= q67)
    w3_mask = working["_close_time"] > q67

    # Ensure no window is completely empty due to duplicate quantiles
    if not w1_mask.any() or not w2_mask.any() or not w3_mask.any():
        idx1 = n // 3
        idx2 = 2 * n // 3
        w1_mask = np.zeros(n, dtype=bool)
        w2_mask = np.zeros(n, dtype=bool)
        w3_mask = np.zeros(n, dtype=bool)
        w1_mask[:idx1] = True
        w2_mask[idx1:idx2] = True
        w3_mask[idx2:] = True

    windows = {}
    pnls = []
    for label, mask in (("T1", w1_mask), ("T2", w2_mask), ("T3", w3_mask)):
        sub = working[mask].reset_index(drop=True)
        sub_n = len(sub)
        if sub_n == 0:
            windows[label] = {
                "status": "EMPTY",
                "n_markets": 0,
                "n_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
            }
            pnls.append(0.0)
            continue

        sub_quotes = (
            quotes[quotes["market_id"].isin(sub["market_id"])].reset_index(drop=True)
            if quotes is not None and not quotes.empty and "market_id" in quotes.columns
            else None
        )
        res = compute_logreg_polymarket_backtest(
            sub, sub["_p_yes"].to_numpy(), sub_quotes, strategy_branch=strategy_branch, **backtest_kwargs
        )
        trades = res.get("trades", [])
        trade_pnls = [t.get("pnl", 0.0) for t in trades]
        max_dd, _ = _compute_drawdown_series(trade_pnls)
        tot_pnl = float(res.get("total_pnl", 0.0))
        pnls.append(tot_pnl)
        windows[label] = {
            "status": "SPARSE" if sub_n < 30 else "OK",
            "n_markets": sub_n,
            "n_trades": int(res.get("n_trades", 0)),
            "total_pnl": round(tot_pnl, 4),
            "win_rate": round(float(res.get("win_rate", 0.0)), 4),
            "roi": round(float(res.get("roi", 0.0)), 4),
            "max_drawdown": round(max_dd, 4),
            "time_start": sub["_close_time"].min().isoformat() if not sub.empty and pd.notna(sub["_close_time"].min()) else None,
            "time_end": sub["_close_time"].max().isoformat() if not sub.empty and pd.notna(sub["_close_time"].max()) else None,
        }

    windows["median_pnl"] = round(float(np.median(pnls)), 4)
    windows["non_negative_windows_count"] = int(sum(p >= 0 for p in pnls))
    return windows


def audit_single_model(
    model: ModelRegistry,
    artifact_payload: dict[str, Any] | None,
    *,
    min_edge: float = 0.03,
    fee_rate: float = 0.02,
    cost_buffer: float = 0.0,
) -> dict[str, Any]:
    """Audit one LogReg model using its deserialized OOF artifact."""
    if artifact_payload is None:
        return {
            "model_id": model.id,
            "asset": model.asset,
            "version": model.version,
            "is_active": model.is_active,
            "error": "OOF artifact missing in model_registry_oof_artifacts",
            "deployable": False,
        }

    frame = artifact_payload["frame"]
    quotes = artifact_payload.get("quotes")
    calibrated_scores = artifact_payload["oof_scores"]
    raw_scores = artifact_payload.get("raw_oof_scores", calibrated_scores)

    if frame.empty or len(calibrated_scores) == 0:
        return {
            "model_id": model.id,
            "asset": model.asset,
            "version": model.version,
            "is_active": model.is_active,
            "error": "OOF artifact frame is empty",
            "deployable": False,
        }

    # Calibration & probability metrics on target (if target is present)
    metrics: dict[str, Any] = {}
    if "target" in frame.columns:
        valid_mask = frame["target"].notna() & np.isfinite(calibrated_scores)
        y_true = frame.loc[valid_mask, "target"].astype(int).to_numpy()
        y_cal = calibrated_scores[valid_mask]
        y_raw = raw_scores[valid_mask]
        if len(y_true) > 0:
            metrics["calibrated_brier"] = round(float(brier_score_loss(y_true, y_cal)), 4)
            metrics["raw_brier"] = round(float(brier_score_loss(y_true, y_raw)), 4)
            metrics["calibrated_ece"] = round(float(_compute_ece(y_true, y_cal)), 4)
            metrics["raw_ece"] = round(float(_compute_ece(y_true, y_raw)), 4)
            try:
                metrics["log_loss"] = round(float(log_loss(y_true, np.clip(y_cal, 1e-6, 1.0 - 1e-6))), 4)
            except Exception:
                metrics["log_loss"] = None

    # Canonical Polymarket Evaluator for COMBINED, FAVORITE_ONLY, OUTSIDER_ONLY
    branches = {}
    for branch in ("COMBINED", "FAVORITE_ONLY", "OUTSIDER_ONLY"):
        res = compute_logreg_polymarket_backtest(
            frame,
            calibrated_scores,
            quotes,
            strategy_branch=branch,
            min_edge=min_edge,
            fee_rate=fee_rate,
            cost_buffer=cost_buffer,
        )
        trades = res.get("trades", [])
        trade_pnls = [t.get("pnl", 0.0) for t in trades]
        max_dd, _ = _compute_drawdown_series(trade_pnls)

        up_trades = [t for t in trades if t.get("side") == "YES"]
        down_trades = [t for t in trades if t.get("side") == "NO"]

        branches[branch] = {
            "n_markets": res.get("n_markets", 0),
            "n_trades": res.get("n_trades", 0),
            "total_pnl": round(float(res.get("total_pnl", 0.0)), 4),
            "win_rate": round(float(res.get("win_rate", 0.0)), 4),
            "roi": round(float(res.get("roi", 0.0)), 4),
            "max_drawdown": round(max_dd, 4),
            "up_pnl": round(sum(t.get("pnl", 0.0) for t in up_trades), 4),
            "down_pnl": round(sum(t.get("pnl", 0.0) for t in down_trades), 4),
        }

    # 3 Chronological OOT Windows for COMBINED
    # Get the unique first-valid-per-market frame and p_yes
    from polyflip.crypto.logreg_polymarket_backtest import _first_valid_per_market
    unique_selected, p_yes = _first_valid_per_market(frame, calibrated_scores)
    oot_windows = _split_chronological_windows(
        unique_selected,
        p_yes,
        quotes,
        strategy_branch="COMBINED",
        min_edge=min_edge,
        fee_rate=fee_rate,
        cost_buffer=cost_buffer,
    )

    combined = branches["COMBINED"]
    ece_val = metrics.get("calibrated_ece", 1.0)
    is_deployable = (
        combined["total_pnl"] > 0
        and oot_windows.get("median_pnl", -1.0) > 0
        and oot_windows.get("non_negative_windows_count", 0) >= 2
        and combined["n_trades"] >= 30
        and (ece_val is None or ece_val <= 0.10)
    )

    rejection_reasons = []
    if combined["total_pnl"] <= 0:
        rejection_reasons.append(f"COMBINED PnL <= 0 ({combined['total_pnl']})")
    if oot_windows.get("median_pnl", 0.0) <= 0:
        rejection_reasons.append(f"Median window PnL <= 0 ({oot_windows.get('median_pnl')})")
    if oot_windows.get("non_negative_windows_count", 0) < 2:
        rejection_reasons.append(f"Less than 2 non-negative windows ({oot_windows.get('non_negative_windows_count')}/3)")
    if combined["n_trades"] < 30:
        rejection_reasons.append(f"Insufficient trades ({combined['n_trades']} < 30)")
    if ece_val is not None and ece_val > 0.10:
        rejection_reasons.append(f"Excessive ECE ({ece_val} > 0.10)")

    return {
        "model_id": model.id,
        "asset": model.asset,
        "version": model.version,
        "is_active": model.is_active,
        "decision_threshold": float(model.decision_threshold) if model.decision_threshold is not None else None,
        "decision_threshold_down": float(model.decision_threshold_down) if model.decision_threshold_down is not None else None,
        "features": model.features,
        "calibration_metrics": metrics,
        "combined_branch": combined,
        "favorite_only_branch": branches["FAVORITE_ONLY"],
        "outsider_only_branch": branches["OUTSIDER_ONLY"],
        "oot_windows": oot_windows,
        "deployable": is_deployable,
        "rejection_reasons": rejection_reasons,
        "training_params": model.training_params,
        "trained_at": model.trained_at.isoformat() if model.trained_at else None,
    }


async def audit_models(
    *,
    model_ids: Sequence[int] = (),
    assets: Sequence[str] = (),
    all_active: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    async with async_session() as session:
        query = select(ModelRegistry).where(
            ModelRegistry.model_type.in_(["logreg", "logistic_regression"])
        )
        if all_active:
            query = query.where(ModelRegistry.is_active == True)
        elif model_ids:
            query = query.where(ModelRegistry.id.in_(model_ids))
        elif assets:
            query = query.where(ModelRegistry.asset.in_(assets))
        else:
            # Default to all active
            query = query.where(ModelRegistry.is_active == True)

        query = query.order_by(ModelRegistry.asset, ModelRegistry.version.desc())
        models = (await session.execute(query)).scalars().all()

        # Load OOF artifacts
        model_id_list = [m.id for m in models]
        artifacts_stmt = select(ModelRegistryOOFArtifact).where(
            ModelRegistryOOFArtifact.model_registry_id.in_(model_id_list)
        )
        artifact_rows = (await session.execute(artifacts_stmt)).scalars().all()
        artifacts_by_id = {}
        for row in artifact_rows:
            try:
                artifacts_by_id[row.model_registry_id] = deserialize_oof_artifact(row.artifact_blob)
            except Exception as e:
                artifacts_by_id[row.model_registry_id] = None

        results = []
        for m in models:
            art = artifacts_by_id.get(m.id)
            res = audit_single_model(m, art)
            results.append(res)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    audit_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_models_audited": len(results),
        "deployable_models_count": sum(1 for r in results if r.get("deployable")),
        "models": results,
    }

    if output_path is None:
        output_path = Path(f"logreg_active_audit_{date_str}.json")

    output_path.write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return audit_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LogReg models against canonical OOF criteria")
    parser.add_argument("--model-ids", nargs="*", type=int, default=(), help="Specific model IDs")
    parser.add_argument("--assets", nargs="*", type=str, default=(), help="Specific assets")
    parser.add_argument("--all-active", action="store_true", help="Audit all active LogReg models")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = asyncio.run(
        audit_models(
            model_ids=args.model_ids,
            assets=args.assets,
            all_active=args.all_active,
            output_path=args.output,
        )
    )
    print(
        f"Audited {report['total_models_audited']} LogReg models. "
        f"Deployable: {report['deployable_models_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
