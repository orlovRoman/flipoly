"""Extended tool allowlist for AI Lab Autonomous Researcher (Phase 10).

All actions are strictly typed, bounded, audit-logged, and prevent direct SQL/shell/LIVE execution.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.ai_lab.llm import HypothesisProposal
from polyflip.ai_lab.orchestrator import promote_to_shadow
from polyflip.ai_lab.service import (
    AILabError,
    append_step,
    create_deployment_revision,
    create_experiment_config,
    propose_live_deployment,
    record_deployment_event,
    rollback_deployment,
    transition_run,
    utc_now,
)
from polyflip.db.models import (
    AIApprovalRequest,
    AIConfigOverlay,
    AIExperimentConfig,
    AIModelArtifact,
    AIOptimizationRun,
    AIRunStep,
    AIShadowAssignment,
    AIStepAuditLog,
    DeploymentRevision,
    ExperimentResult,
    LGBMTrainingJob,
    ModelRegistry,
    RuntimeSettings,
    TradeHistory,
)

logger = structlog.get_logger("polyflip.ai_lab.agent_tools")

# Allowlist of runtime settings keys that an overlay is permitted to touch
ALLOWED_OVERLAY_KEYS: set[str] = {
    "MIN_EDGE",
    "OUTSIDER_MAX_PRICE",
    "TRADE_MIN_PRICE",
    "TRADE_MAX_PRICE",
    "DEAD_ZONE_WIDTH",
    "FAVORITE_THRESHOLD",
    "BET_SIZING_MODE",
    "MAX_BET_SIZE_USDC",
    "DAILY_LOSS_LIMIT_USDC",
    "FLIP_THRESHOLD",
    "NO_MIN_EDGE",
    "FAVORITE_MIN_EDGE",
    "CRYPTO_MIN_EDGE",
    "TRADE_NO_FLIP_THRESHOLD",
    "TRADE_MIN_TIME_LEFT_SEC",
    "TRADE_MAX_TIME_LEFT_SEC",
}

# Bounds for safety verification
OVERLAY_BOUNDS: dict[str, tuple[float, float]] = {
    "MIN_EDGE": (-0.05, 0.20),
    "OUTSIDER_MAX_PRICE": (0.10, 0.50),
    "TRADE_MIN_PRICE": (0.01, 0.50),
    "TRADE_MAX_PRICE": (0.50, 0.99),
    "DEAD_ZONE_WIDTH": (0.02, 0.25),
    "FAVORITE_THRESHOLD": (0.50, 0.85),
    "MAX_BET_SIZE_USDC": (1.0, 100.0),
    "DAILY_LOSS_LIMIT_USDC": (-500.0, -10.0),
    "FLIP_THRESHOLD": (0.50, 0.90),
    "TRADE_NO_FLIP_THRESHOLD": (0.10, 0.90),
    "TRADE_MIN_TIME_LEFT_SEC": (0.0, 3600.0),
    "TRADE_MAX_TIME_LEFT_SEC": (1.0, 3600.0),
}

OVERLAY_ENUMS: dict[str, set[str]] = {
    "BET_SIZING_MODE": {"flat", "fixed", "scaled", "kelly"},
}
INTEGER_OVERLAY_KEYS = {
    "TRADE_MIN_TIME_LEFT_SEC",
    "TRADE_MAX_TIME_LEFT_SEC",
}


# ---------------------------------------------------------------------------
# Investigation & Analytics Tools (Read-Only)
# ---------------------------------------------------------------------------
async def get_active_models(session: AsyncSession, asset: str) -> list[dict[str, Any]]:
    """Retrieve currently active baseline models for a specific asset."""
    stmt = (
        select(ModelRegistry)
        .where(ModelRegistry.asset == asset, ModelRegistry.is_active.is_(True))
        .order_by(desc(ModelRegistry.id))
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "asset": r.asset,
            "version": r.version,
            "model_type": r.model_type,
            "features": r.features,
            "decision_threshold": r.decision_threshold,
            "decision_threshold_down": r.decision_threshold_down,
            "accuracy": r.accuracy,
            "backtest_pnl": r.backtest_pnl,
            "backtest_trades": r.backtest_trades,
            "interval": r.interval,
            "trained_at": r.trained_at.isoformat() if r.trained_at else None,
        }
        for r in rows
    ]


async def get_model_registry(session: AsyncSession, asset: str, limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve historical models registered for an asset."""
    stmt = (
        select(ModelRegistry)
        .where(ModelRegistry.asset == asset)
        .order_by(desc(ModelRegistry.id))
        .limit(min(limit, 50))
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "asset": r.asset,
            "version": r.version,
            "model_type": r.model_type,
            "is_active": r.is_active,
            "features": r.features,
            "backtest_pnl": r.backtest_pnl,
            "backtest_trades": r.backtest_trades,
            "trained_at": r.trained_at.isoformat() if r.trained_at else None,
        }
        for r in rows
    ]


async def get_recent_trade_statistics(session: AsyncSession, asset: str, days: int = 30) -> dict[str, Any]:
    """Aggregate recent trade metrics (win rate, total PnL, trade volume)."""
    start_date = utc_now() - timedelta(days=min(days, 90))
    stmt = select(TradeHistory).where(
        TradeHistory.asset.startswith(asset[:3]),
        TradeHistory.position_status == "CLOSED",
        TradeHistory.created_at >= start_date,
    )
    trades = (await session.execute(stmt)).scalars().all()

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if (t.pnl or 0) > 0)
    total_pnl = sum(float(t.pnl or 0) for t in trades)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "asset": asset,
        "days": days,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 4),
    }


async def get_polymarket_oot_history(session: AsyncSession, asset: str, limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve historical Polymarket OOT evaluation results."""
    stmt = (
        select(ExperimentResult)
        .join(AIExperimentConfig, ExperimentResult.config_id == AIExperimentConfig.id)
        .where(
            AIExperimentConfig.asset == asset,
            ExperimentResult.evaluation_kind == "POLYMARKET_OOT",
            ExperimentResult.status == "SUCCEEDED",
        )
        .order_by(desc(ExperimentResult.id))
        .limit(min(limit, 30))
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "result_id": r.id,
            "config_id": r.config_id,
            "net_pnl": r.net_pnl,
            "trade_count": r.trade_count,
            "max_drawdown": r.max_drawdown,
            "metrics": r.metrics,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Experiment Mutation & Creation Tools
# ---------------------------------------------------------------------------
def compute_config_hash(
    asset: str,
    regime: str,
    model_family: str,
    feature_set: str,
    feature_pipeline_version: str,
    model_params: Mapping[str, Any],
    strategy_params: Mapping[str, Any],
    backtest_params: Mapping[str, Any],
) -> str:
    """Deterministic hash of experiment parameters."""
    canonical_payload = {
        "asset": asset,
        "regime": regime,
        "model_family": model_family,
        "feature_set": feature_set,
        "feature_pipeline_version": feature_pipeline_version,
        "model_params": dict(model_params),
        "strategy_params": dict(strategy_params),
        "backtest_params": dict(backtest_params),
    }
    raw = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def create_experiment_config_from_proposal(
    session: AsyncSession,
    proposal: HypothesisProposal,
    *,
    parent_id: int | None = None,
    created_by: str = "ai_agent",
) -> AIExperimentConfig:
    """Create a reproducible immutable experiment configuration from an LLM proposal."""
    asset = proposal.asset
    regime = "DEFAULT"
    model_family = proposal.model_family
    feature_set = proposal.feature_set
    feature_pipeline_version = "1.0"
    model_params = dict(proposal.parameter_changes or {})
    strategy_params = dict(proposal.strategy_parameter_changes or {})
    backtest_params = dict(proposal.test_plan or {})

    config_hash = compute_config_hash(
        asset=asset,
        regime=regime,
        model_family=model_family,
        feature_set=feature_set,
        feature_pipeline_version=feature_pipeline_version,
        model_params=model_params,
        strategy_params=strategy_params,
        backtest_params=backtest_params,
    )

    existing = (
        await session.execute(
            select(AIExperimentConfig).where(AIExperimentConfig.config_hash == config_hash).limit(1)
        )
    ).scalar_one_or_none()

    if existing:
        return existing

    name = f"exp-{asset.lower()}-{feature_set.lower()}-{int(time.time())}"
    config = await create_experiment_config(
        session,
        name=name,
        asset=asset,
        regime=regime,
        model_family=model_family,
        feature_set=feature_set,
        feature_pipeline_version=feature_pipeline_version,
        model_params=model_params,
        strategy_params=strategy_params,
        backtest_params=backtest_params,
        config_hash=config_hash,
        parent_id=parent_id,
        created_by=created_by,
    )
    return config


# ---------------------------------------------------------------------------
# Configuration Overlay Tools (Section 10.2)
# ---------------------------------------------------------------------------
def validate_overlay_changes(changes: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that overlay changes stay within strict allowed parameters and ranges."""
    cleaned: dict[str, Any] = {}
    for key, value in changes.items():
        key = str(key).strip()
        if key not in ALLOWED_OVERLAY_KEYS:
            raise AILabError(f"Prohibited overlay parameter: '{key}'. Allowed keys: {sorted(ALLOWED_OVERLAY_KEYS)}")
        if key in OVERLAY_ENUMS:
            if not isinstance(value, str) or value.lower() not in OVERLAY_ENUMS[key]:
                raise AILabError(
                    f"Overlay parameter '{key}' must be one of {sorted(OVERLAY_ENUMS[key])}"
                )
            cleaned[key] = value.lower()
            continue
        if key in OVERLAY_BOUNDS:
            if isinstance(value, bool):
                raise AILabError(f"Overlay parameter '{key}' must be numeric, got boolean")
            min_val, max_val = OVERLAY_BOUNDS[key]
            try:
                num_val = float(value)
            except (TypeError, ValueError):
                raise AILabError(f"Overlay parameter '{key}' must be numeric, got: {value}")
            if not (min_val <= num_val <= max_val):
                raise AILabError(
                    f"Overlay parameter '{key}' value {num_val} violates safety bounds [{min_val}, {max_val}]"
                )
            if key in INTEGER_OVERLAY_KEYS and not num_val.is_integer():
                raise AILabError(f"Overlay parameter '{key}' must be an integer")
            cleaned[key] = int(num_val) if key in INTEGER_OVERLAY_KEYS else num_val
            continue
        raise AILabError(f"Overlay parameter '{key}' has no declared type or bounds")
    if "TRADE_MIN_PRICE" in cleaned and "TRADE_MAX_PRICE" in cleaned:
        if cleaned["TRADE_MIN_PRICE"] > cleaned["TRADE_MAX_PRICE"]:
            raise AILabError("TRADE_MIN_PRICE must not exceed TRADE_MAX_PRICE")
    if "TRADE_MIN_TIME_LEFT_SEC" in cleaned and "TRADE_MAX_TIME_LEFT_SEC" in cleaned:
        if cleaned["TRADE_MIN_TIME_LEFT_SEC"] > cleaned["TRADE_MAX_TIME_LEFT_SEC"]:
            raise AILabError("TRADE_MIN_TIME_LEFT_SEC must not exceed TRADE_MAX_TIME_LEFT_SEC")
    return cleaned


async def create_config_overlay(
    session: AsyncSession,
    *,
    run_id: int,
    changes: Mapping[str, Any],
    parent_overlay_id: int | None = None,
    ttl_seconds: int = 3600,
    created_by: str = "ai_agent",
) -> AIConfigOverlay:
    """Create a versioned runtime settings overlay."""
    cleaned_changes = validate_overlay_changes(changes)
    now = utc_now()
    expires_at = now + timedelta(seconds=min(ttl_seconds, 86400))

    current_rows = (
        await session.execute(
            select(RuntimeSettings).where(RuntimeSettings.key.in_(list(cleaned_changes)))
        )
    ).scalars().all()
    current_values = {row.key: row.value for row in current_rows}
    base_payload = {key: current_values.get(key) for key in sorted(cleaned_changes)}
    resulting_payload = {key: cleaned_changes[key] for key in sorted(cleaned_changes)}
    base_settings_hash = hashlib.sha256(
        json.dumps(base_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    resulting_hash = hashlib.sha256(
        json.dumps(resulting_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    overlay = AIConfigOverlay(
        run_id=run_id,
        parent_overlay_id=parent_overlay_id,
        scope={"target": "SHADOW_SIMULATION"},
        changes=cleaned_changes,
        base_settings_hash=base_settings_hash,
        resulting_settings_hash=resulting_hash,
        status="PENDING",
        created_by=created_by,
        expires_at=expires_at,
        rollback_payload={
            "previous_values": current_values,
            "keys": sorted(cleaned_changes),
            "runtime_settings_applied": False,
        },
        created_at=now,
    )
    session.add(overlay)
    await session.flush()
    return overlay


async def apply_shadow_overlay(session: AsyncSession, overlay_id: int) -> AIConfigOverlay:
    """Activate an overlay in passive shadow observation mode."""
    overlay = await session.get(AIConfigOverlay, overlay_id)
    if not overlay:
        raise AILabError(f"Overlay {overlay_id} not found")
    if overlay.status != "PENDING":
        raise AILabError(f"Overlay {overlay_id} cannot be applied from status '{overlay.status}'")

    overlay.status = "APPLIED"
    await session.flush()
    return overlay


async def rollback_overlay(session: AsyncSession, overlay_id: int) -> AIConfigOverlay:
    """Roll back a shadow overlay; runtime settings are never mutated implicitly."""
    overlay = await session.get(AIConfigOverlay, overlay_id)
    if not overlay:
        raise AILabError(f"Overlay {overlay_id} not found")
    if overlay.status not in {"PENDING", "APPLIED"}:
        raise AILabError(
            f"Overlay {overlay_id} cannot be rolled back from status {overlay.status}"
        )
    overlay.status = "ROLLED_BACK"
    await session.flush()
    return overlay


async def expire_overlays(
    session: AsyncSession,
    *,
    now: datetime | None = None,
) -> int:
    """Mark expired shadow overlays durably; no live settings are changed."""
    cutoff = now or utc_now()
    rows = (
        await session.execute(
            select(AIConfigOverlay)
            .where(
                AIConfigOverlay.status.in_({"PENDING", "APPLIED"}),
                AIConfigOverlay.expires_at.is_not(None),
                AIConfigOverlay.expires_at <= cutoff,
            )
            .with_for_update()
        )
    ).scalars().all()
    for overlay in rows:
        overlay.status = "EXPIRED"
    if rows:
        await session.flush()
    return len(rows)


# ---------------------------------------------------------------------------
# Feature Patch Prototyping (Isolated Workspace)
# ---------------------------------------------------------------------------
def generate_feature_patch(
    feature_name: str,
    expression: str,
    description: str = "",
) -> dict[str, Any]:
    """Draft a new experimental feature definition in an isolated sandbox format."""
    clean_name = feature_name.strip().lower()
    if not clean_name.replace("_", "").isalnum():
        raise AILabError(f"Invalid feature name '{feature_name}'")

    patch_payload = {
        "feature_name": clean_name,
        "expression": expression.strip(),
        "description": description,
        "created_at": utc_now().isoformat(),
        "patch_hash": hashlib.sha256(f"{clean_name}:{expression}".encode("utf-8")).hexdigest(),
    }
    return patch_payload
