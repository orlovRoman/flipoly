"""Audit joint LightGBM thresholds from persisted OOF artifacts.

The command is read-only. It evaluates 20/40/60/80% target coverage for all
saved canonical LightGBM candidates and writes no registry or runtime state.
Run it from the repository root, for example:

    python -m polyflip.scripts.audit_lgbm_thresholds --symbols BTCUSDT ETHUSDT
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select

from polyflip.crypto.experiment_configs import normalize_experiment_config
from polyflip.crypto.oof_artifact import OOF_ARTIFACT_SCHEMA_VERSION, deserialize_oof_artifact
from polyflip.crypto.threshold_optimizer import TARGET_COVERAGES, optimize_joint_thresholds
from polyflip.db.connection import async_session
from polyflip.db.models import ModelRegistry, ModelRegistryOOFArtifact


def _backtest_options(params: dict[str, Any]) -> dict[str, Any]:
    raw = params.get("backtest_config")
    if raw is None:
        experiment = params.get("experiment_config")
        raw = experiment.get("backtest") if isinstance(experiment, dict) else None
    if raw is None:
        return {}
    return normalize_experiment_config({"backtest": raw})["backtest"]


async def audit_models(
    *,
    symbols: Sequence[str] = (),
    model_ids: Sequence[int] = (),
    selected_target_coverage: float = 0.40,
) -> dict[str, Any]:
    symbols = tuple(symbol.strip().upper() for symbol in symbols if symbol.strip())
    model_ids = tuple(int(value) for value in model_ids)
    async with async_session() as db:
        stmt = select(ModelRegistry).where(ModelRegistry.model_type == "lgbm")
        if symbols:
            assets = [f"{symbol}_{regime}" for symbol in symbols for regime in ("low_vol", "mid_vol", "high_vol")]
            stmt = stmt.where(ModelRegistry.asset.in_(assets))
        if model_ids:
            stmt = stmt.where(ModelRegistry.id.in_(model_ids))
        models = (await db.execute(stmt.order_by(ModelRegistry.asset, ModelRegistry.version.desc()))).scalars().all()
        artifact_rows = (await db.execute(
            select(ModelRegistryOOFArtifact).where(
                ModelRegistryOOFArtifact.model_registry_id.in_([model.id for model in models]),
                ModelRegistryOOFArtifact.schema_version == OOF_ARTIFACT_SCHEMA_VERSION,
            )
        )).scalars().all() if models else []
        artifacts = {artifact.model_registry_id: artifact for artifact in artifact_rows}

        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for model in models:
            artifact = artifacts.get(model.id)
            if artifact is None:
                errors.append({"model_id": model.id, "asset": model.asset, "error": "OOF_ARTIFACT_MISSING"})
                continue
            try:
                payload = deserialize_oof_artifact(artifact.artifact_blob)
                raw_scores = payload.get("raw_oof_scores")
                if raw_scores is None:
                    raw_scores = payload["oof_scores"]
                audit = optimize_joint_thresholds(
                    payload["frame"], raw_scores, payload["oof_scores"], payload["quotes"],
                    target_coverages=TARGET_COVERAGES,
                    selected_target_coverage=selected_target_coverage,
                    strategy_branches=("OUTSIDER_ONLY", "FAVORITE_ONLY", "COMBINED"),
                    **_backtest_options(model.training_params or {}),
                )
                results.append({
                    "model_id": model.id,
                    "asset": model.asset,
                    "version": model.version,
                    "is_active": bool(model.is_active),
                    "calibration_method": (model.training_params or {}).get("calibration_method"),
                    "artifact_rows": artifact.row_count,
                    **audit,
                })
            except Exception as exc:  # keep the audit going for other candidates
                errors.append({"model_id": model.id, "asset": model.asset, "error": str(exc)})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_coverages": [round(value * 100.0, 2) for value in TARGET_COVERAGES],
        "selected_target_coverage": round(selected_target_coverage * 100.0, 2),
        "read_only": True,
        "models": results,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=(), help="Optional symbols, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--model-ids", nargs="*", type=int, default=(), help="Optional ModelRegistry ids")
    parser.add_argument("--target-coverage", type=float, default=0.40, help="Selected candidate target, e.g. 0.40")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0.05 <= args.target_coverage <= 0.95:
        raise SystemExit("--target-coverage must be between 0.05 and 0.95")
    payload = asyncio.run(audit_models(
        symbols=args.symbols,
        model_ids=args.model_ids,
        selected_target_coverage=args.target_coverage,
    ))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if not payload["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
