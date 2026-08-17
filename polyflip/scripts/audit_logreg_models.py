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

from polyflip.crypto.logreg_polymarket_backtest import (
    compute_logreg_polymarket_backtest,
    split_chronological_oot_windows,
)
from polyflip.crypto.oof_artifact import deserialize_oof_artifact
from polyflip.crypto.polymarket_backtest import adapt_canonical_backtest_metrics
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
    """Split canonical evaluated markets into 3 chronological windows T1/T2/T3 strictly by market close time."""
    return split_chronological_oot_windows(
        frame,
        p_yes,
        quotes,
        strategy_branch=strategy_branch,
        **backtest_kwargs,
    )


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
        canonical_metrics = adapt_canonical_backtest_metrics(res)
        trades = res.get("trades", [])
        up_trades = [t for t in trades if t.get("side") == "YES"]
        down_trades = [t for t in trades if t.get("side") == "NO"]

        branches[branch] = {
            "n_markets": int(res.get("n_markets") or 0),
            "n_trades": canonical_metrics.n_trades,
            "net_profit": round(canonical_metrics.net_profit, 4),
            "total_pnl": round(canonical_metrics.net_profit, 4),
            "win_rate": round(canonical_metrics.win_rate, 4),
            "roi_pct": round(canonical_metrics.roi_pct, 4),
            "roi": round(canonical_metrics.roi_pct, 4),
            "max_drawdown": round(canonical_metrics.max_drawdown, 4),
            "max_drawdown_usdc": round(canonical_metrics.max_drawdown, 4),
            "up_pnl": round(sum(t.get("pnl", 0.0) for t in up_trades), 4),
            "down_pnl": round(sum(t.get("pnl", 0.0) for t in down_trades), 4),
        }

    # 3 Chronological OOT Windows for COMBINED strictly by market close time
    oot_windows = split_chronological_oot_windows(
        frame,
        calibrated_scores,
        quotes,
        strategy_branch="COMBINED",
        min_edge=min_edge,
        fee_rate=fee_rate,
        cost_buffer=cost_buffer,
    )

    combined = branches["COMBINED"]
    ece_val = metrics.get("calibrated_ece", 1.0)
    median_win_pnl = oot_windows.get("median_pnl")
    non_neg_windows = oot_windows.get("non_negative_windows_count", 0)

    is_deployable = (
        combined["net_profit"] > 0
        and median_win_pnl is not None
        and median_win_pnl > 0
        and non_neg_windows >= 2
        and combined["n_trades"] >= 30
        and (ece_val is None or ece_val <= 0.10)
    )

    rejection_reasons = []
    if combined["net_profit"] <= 0:
        rejection_reasons.append(f"COMBINED PnL <= 0 ({combined['net_profit']:.2f})")
    if median_win_pnl is None or median_win_pnl <= 0:
        med_str = f"{median_win_pnl:.2f}" if median_win_pnl is not None else "None"
        rejection_reasons.append(f"Median window PnL <= 0 ({med_str})")
    if non_neg_windows < 2:
        rejection_reasons.append(f"Less than 2 non-negative windows ({non_neg_windows}/3)")
    if combined["n_trades"] < 30:
        rejection_reasons.append(f"Insufficient trades ({combined['n_trades']} < 30)")
    if ece_val is not None and ece_val > 0.10:
        rejection_reasons.append(f"Excessive ECE ({ece_val:.4f} > 0.10)")

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
            except Exception:
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
