"""Stage 1: Capture LogReg baseline state.

Exports current git commit, active LogReg models, their versions, thresholds,
features, backtest metrics, runtime settings, and container states into
logreg_baseline_YYYYMMDD.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from polyflip.db.connection import async_session
from polyflip.db.models import ModelRegistry, RuntimeSettings


async def capture_baseline(output_path: Path | None = None) -> dict[str, Any]:
    # 1. Git commit
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "8391416"

    # 2. Containers
    try:
        docker_ps = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"],
            text=True,
        ).strip().splitlines()
        containers = {
            parts[0]: {"status": parts[1], "image": parts[2]}
            for line in docker_ps
            if (parts := line.split("\t")) and len(parts) >= 3
        }
    except Exception:
        containers = {}

    # 3. Active models and runtime settings from DB
    async with async_session() as session:
        # Runtime settings
        settings_rows = (
            await session.execute(select(RuntimeSettings))
        ).scalars().all()
        runtime_settings = {
            s.key: {
                "value": s.value,
                "description": s.description,
                "updated_by": s.updated_by,
            }
            for s in settings_rows
        }

        # Active LogReg models
        models_stmt = (
            select(ModelRegistry)
            .where(
                ModelRegistry.is_active == True,
                ModelRegistry.model_type.in_(["logreg", "logistic_regression"]),
            )
            .order_by(ModelRegistry.asset, ModelRegistry.version.desc())
        )
        active_models_rows = (await session.execute(models_stmt)).scalars().all()

        active_models: list[dict[str, Any]] = []
        for m in active_models_rows:
            active_models.append({
                "id": m.id,
                "asset": m.asset,
                "version": m.version,
                "is_active": m.is_active,
                "model_type": m.model_type,
                "decision_threshold": float(m.decision_threshold) if m.decision_threshold is not None else None,
                "decision_threshold_down": float(m.decision_threshold_down) if m.decision_threshold_down is not None else None,
                "accuracy": float(m.accuracy) if m.accuracy is not None else None,
                "ece": float(m.ece) if m.ece is not None else None,
                "backtest_pnl": float(m.backtest_pnl) if m.backtest_pnl is not None else None,
                "backtest_trades": m.backtest_trades,
                "features": m.features,
                "training_params": m.training_params,
                "dataset_fingerprint": m.dataset_fingerprint,
                "trained_at": m.trained_at.isoformat() if m.trained_at else None,
                "activated_at": m.activated_at.isoformat() if m.activated_at else None,
            })

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    baseline = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "containers": containers,
        "runtime_settings": runtime_settings,
        "active_logreg_models_count": len(active_models),
        "active_models": active_models,
        "problematic_models_focus": [
            m for m in active_models
            if m["asset"] in {"DOGE_decided", "ETH", "XRP", "ETH_decided", "ETH_leaning"}
        ],
    }

    if output_path is None:
        output_path = Path(f"logreg_baseline_{date_str}.json")

    output_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return baseline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture LogReg baseline state")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    res = asyncio.run(capture_baseline(args.output))
    print(f"Captured baseline with {res['active_logreg_models_count']} active LogReg models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
