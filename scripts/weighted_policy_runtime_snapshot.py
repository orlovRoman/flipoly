#!/usr/bin/env python3
"""Write a redacted runtime snapshot for weighted-policy rollout audits."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

SAFE_ENV_KEYS = (
    "POLYFLIP_BUILD_SHA",
    "TRADING_POLICY_MODE",
    "WEIGHTED_POLICY_ID",
    "WEIGHTED_MARKET_WEIGHT",
    "WEIGHTED_LOGREG_WEIGHT",
    "WEIGHTED_LGBM_WEIGHT",
    "WEIGHTED_MRF_BETA",
    "WEIGHTED_INTERCEPT",
    "WEIGHTED_FEE_RATE",
    "WEIGHTED_MAKER_FEE_RATE",
    "WEIGHTED_FEE_EXPONENT",
    "WEIGHTED_SLIPPAGE_RATE",
    "WEIGHTED_LATENCY_BUFFER",
    "WEIGHTED_EXECUTION_ROLE",
    "WEIGHTED_MIN_NET_EV_FAVORITE",
    "WEIGHTED_MIN_NET_EV_OUTSIDER",
    "WEIGHTED_FIXED_BET_USDC",
    "WEIGHTED_MRF_EXTREME_VETO_THRESHOLD",
    "WEIGHTED_MODELS_AGREE_BETA",
    "WEIGHTED_MRF_APPLICATION",
    "WEIGHTED_MRF_SIZING_GAMMA",
    "WEIGHTED_SIZING_MODE",
    "WEIGHTED_STANDARD_ERROR",
    "WEIGHTED_KELLY_FRACTION",
    "WEIGHTED_SIZE_CAP_USDC",
    "WEIGHTED_POLICY_ARTIFACT_PATH",
    "PAPER_FEE_MODEL",
    "PAPER_FEE_RATE",
    "PAPER_FEE_EXPONENT",
    "MARKET_REGIME_FILTER_MODE",
    "LIVE_TRADING_ENABLED",
    "EXECUTION_MODE",
)

MODEL_COLUMNS = (
    "id",
    "asset",
    "version",
    "model_type",
    "features",
    "decision_threshold",
    "decision_threshold_down",
    "brier_score",
    "ece",
    "interval",
    "dataset_fingerprint",
    "trained_at",
    "quality_gate_passed",
    "activation_source",
    "activated_at",
)


def _git(command: list[str]) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", *command],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        # Production images do not include the .git directory. The deploy
        # pipeline injects immutable commit metadata for reproducible snapshots.
        if command == ["rev-parse", "HEAD"]:
            return os.getenv("POLYFLIP_BUILD_SHA") or None
        if command == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return (
                os.getenv("POLYFLIP_BUILD_BRANCH")
                or os.getenv("GIT_BRANCH")
                or None
            )
        return None


def build_snapshot() -> dict[str, Any]:
    environment = {
        key: os.getenv(key)
        for key in SAFE_ENV_KEYS
        if os.getenv(key) is not None
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "sha": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "safe_environment": environment,
        "active_models": [],
        "active_models_source": "not_requested",
        "secrets_omitted": True,
    }


async def _active_models(database_url: str) -> tuple[list[dict[str, Any]], str, str | None]:
    """Read active model metadata without loading model weights or secrets."""
    url = database_url
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    engine = create_async_engine(url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            if url.startswith("sqlite"):
                result = await connection.execute(text("PRAGMA table_info(model_registry)"))
                available = {str(row[1]) for row in result.fetchall()}
            else:
                result = await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = 'model_registry'"
                    )
                )
                available = {str(row[0]) for row in result.fetchall()}
            if not {"asset", "version", "is_active"}.issubset(available):
                return [], "unavailable", "MODEL_REGISTRY_SCHEMA_INCOMPLETE"
            selected = [column for column in MODEL_COLUMNS if column in available]
            query = (
                "SELECT "
                + ", ".join(selected)
                + " FROM model_registry WHERE is_active IS TRUE "
                + "ORDER BY asset, version DESC"
            )
            result = await connection.execute(text(query))
            rows: list[dict[str, Any]] = []
            for row in result.mappings().all():
                item = dict(row)
                for key, value in tuple(item.items()):
                    if isinstance(value, datetime):
                        item[key] = value.isoformat()
                rows.append(item)
            return rows, "database", None
    except Exception as exc:
        # Do not echo a DSN or driver message that could contain credentials.
        return [], "error", type(exc).__name__
    finally:
        await engine.dispose()


async def enrich_snapshot(snapshot: dict[str, Any], database_url: str | None) -> dict[str, Any]:
    if not database_url:
        return snapshot
    models, source, error = await _active_models(database_url)
    snapshot["active_models"] = models
    snapshot["active_models_source"] = source
    if error:
        snapshot["active_models_error"] = error
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="read-only database URL used to include active ModelRegistry versions",
    )
    parser.add_argument(
        "--assert-live-disabled",
        action="store_true",
        help="fail if the snapshot says that live trading is enabled",
    )
    args = parser.parse_args()
    snapshot = asyncio.run(enrich_snapshot(build_snapshot(), args.database_url))
    if args.assert_live_disabled:
        live_enabled = str(
            snapshot["safe_environment"].get("LIVE_TRADING_ENABLED", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if live_enabled:
            raise SystemExit("LIVE_TRADING_ENABLED is true; refusing rollout snapshot")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
