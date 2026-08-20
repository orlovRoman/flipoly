"""Safe lifecycle services for the autonomous AI laboratory.

Phase 2 deliberately stops at experiment orchestration and audit persistence.
It never activates a model, changes RuntimeSettings, or submits an order.
Phase 9 adds the secure human-in-the-loop activation, revision manifest tracking,
and rollback backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from polyflip.config import normalize_ai_lab_mode, settings
from polyflip.ai_lab.manifests import (
    build_deployment_manifest,
    compute_manifest_hash,
)
from polyflip.db.models import (
    AIApprovalRequest,
    AIExperimentConfig,
    AIModelArtifact,
    AIOptimizationRun,
    AIPermission,
    AIRunStep,
    AIStepAuditLog,
    AIShadowAssignment,
    DeploymentEvent,
    DeploymentRevision,
    ExperimentResult,
    ModelRegistry,
    RuntimeSettings,
)

logger = structlog.get_logger(__name__)


class AILabError(ValueError):
    """Base error for rejected laboratory operations."""


class AIRunTransitionError(AILabError):
    """Raised when a run state transition is not allowed."""


class AIPermissionError(AILabError):
    """Raised when an action is outside the immutable permission snapshot."""


class AIResearchModeError(AILabError):
    """Raised when research mode reaches a LIVE-only deployment boundary."""


RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    "DRAFT": frozenset({"QUEUED", "PLANNING", "CANCELLED"}),
    "QUEUED": frozenset({"PLANNING", "RUNNING", "CANCELLED", "FAILED"}),
    "PLANNING": frozenset({"RUNNING", "CANCELLED", "FAILED", "PAUSED"}),
    "RUNNING": frozenset({"EVALUATING", "FAILED", "CANCELLED", "PAUSED"}),
    "EVALUATING": frozenset(
        {
            "PLANNING",
            "QUEUED",
            "SHADOW",
            "PENDING_APPROVAL",
            "INSUFFICIENT_DATA",
            "RESEARCH_PROVISIONAL",
            "INSUFFICIENT_EVIDENCE",
            "TECHNICAL_INVALID",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "PAUSED",
        }
    ),
    "PAUSED": frozenset({"PLANNING", "RUNNING", "CANCELLED", "FAILED"}),
    "SHADOW": frozenset(
        {"PENDING_APPROVAL", "REJECTED", "ROLLED_BACK", "COMPLETED"}
    ),
    "RESEARCH_PROVISIONAL": frozenset({"SHADOW"}),
    "INSUFFICIENT_EVIDENCE": frozenset(),
    "TECHNICAL_INVALID": frozenset(),
    # ACTIVE is reached only by approve_and_activate_deployment after the
    # explicit human approval row-lock transaction; generic run transitions
    # must not provide a direct activation bypass.
    "PENDING_APPROVAL": frozenset({"REJECTED", "COMPLETED"}),
    "ACTIVE": frozenset({"ROLLED_BACK"}),
    "COMPLETED": frozenset(),
    "INSUFFICIENT_DATA": frozenset(),
    "FAILED": frozenset(),
    "REJECTED": frozenset(),
    "CANCELLED": frozenset(),
    "ROLLED_BACK": frozenset(),
}

LAB_ACTIONS = frozenset(
    {
        "CREATE_EXPERIMENT",
        "TRAIN_MODEL",
        "RUN_OOT_BACKTEST",
        "RUN_POLYMARKET_OOT",
        "PROMOTE_TO_SHADOW",
        "STOP_EXPERIMENT",
        "REQUEST_ACTIVATION",
        "REQUEST_ROLLBACK",
    }
)

TRANSITION_ACTIONS: dict[str, str] = {
    "PLANNING": "CREATE_EXPERIMENT",
    "RUNNING": "TRAIN_MODEL",
    "EVALUATING": "RUN_OOT_BACKTEST",
    "SHADOW": "PROMOTE_TO_SHADOW",
    "PENDING_APPROVAL": "REQUEST_ACTIVATION",
    "INSUFFICIENT_DATA": "RUN_OOT_BACKTEST",
}


def transition_action_for_target(target: str) -> str | None:
    """Return the permission action required for a public state transition."""
    return TRANSITION_ACTIONS.get(str(target).upper())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_run_transition(current: str, target: str) -> None:
    current = str(current).upper()
    target = str(target).upper()
    if target not in RUN_TRANSITIONS.get(current, frozenset()):
        raise AIRunTransitionError(
            f"invalid AI Lab run transition: {current} -> {target}"
        )


def validate_permission(permission: AIPermission | None, action: str) -> None:
    if permission is None:
        raise AIPermissionError("run has no permission snapshot")
    if not permission.enabled:
        raise AIPermissionError(
            f"permission profile {permission.profile_name!r} is disabled"
        )
    action = str(action).upper()
    if action not in LAB_ACTIONS:
        raise AIPermissionError(f"unknown AI Lab action: {action}")
    allowed = {str(item).upper() for item in (permission.allowed_actions or [])}
    if action not in allowed:
        raise AIPermissionError(
            f"action {action} is not allowed by {permission.profile_name} v{permission.version}"
        )


async def create_run(
    session: AsyncSession,
    *,
    objective: str,
    scope: Mapping[str, Any],
    autonomy_level: str,
    budget_experiments: int,
    budget_seconds: int = 0,
    permission: AIPermission | None,
    created_by: str = "system",
    agent_thread_id: str | None = None,
    mode: str | None = None,
) -> AIOptimizationRun:
    autonomy_level = autonomy_level.upper()
    resolved_mode = normalize_ai_lab_mode(mode or settings.AI_LAB_MODE)
    if permission is None and autonomy_level != "OBSERVE":
        raise AIPermissionError(
            "permission snapshot is required for autonomous AI Lab runs"
        )
    if permission is not None and not permission.enabled:
        raise AIPermissionError("cannot bind run to a disabled permission profile")
    if budget_experiments < 1:
        raise AILabError("budget_experiments must be at least 1")
    if budget_seconds < 0:
        raise AILabError("budget_seconds must be non-negative")
    if resolved_mode == "RESEARCH":
        # Research is safe in PAPER/SHADOW. Only the explicit live gate may
        # block it; the legacy general TRADING_ENABLED flag is not sufficient.
        live_enabled = bool(getattr(settings, "LIVE_TRADING_ENABLED", False))
        if hasattr(session, "execute"):
            result = await session.execute(
                select(RuntimeSettings).where(
                    RuntimeSettings.key == "LIVE_TRADING_ENABLED"
                )
            )
            runtime_live = result.scalar_one_or_none()
            if runtime_live is not None:
                live_enabled = str(runtime_live.value).strip().lower() == "true"
        if live_enabled:
            raise AILabError(
                "AI_LAB_MODE=RESEARCH cannot be used while LIVE_TRADING_ENABLED=true"
            )
    if autonomy_level not in {
        "OBSERVE",
        "EXPERIMENT",
        "SHADOW",
        "AUTONOMOUS_SHADOW",
        "AUTONOMOUS_CONFIG",
        "LIVE_PROPOSE",
        "AUTONOMOUS_LIVE",
        "DIRECTED",
    }:
        raise AILabError(f"unsupported autonomy_level: {autonomy_level}")

    now = utc_now()
    row = AIOptimizationRun(
        objective=objective,
        scope=dict(scope),
        mode=resolved_mode,
        autonomy_level=autonomy_level,
        status="DRAFT",
        permission_id=permission.id if permission is not None else None,
        experiment_budget=budget_experiments,
        budget_experiments=budget_experiments,
        budget_seconds=budget_seconds,
        created_by=created_by,
        agent_type="AI_LAB",
        agent_thread_id=agent_thread_id,
        created_at=now,
    )
    session.add(row)
    await session.flush()
    return row


async def transition_run(
    session: AsyncSession,
    run: AIOptimizationRun,
    target: str,
    *,
    reason: str | None = None,
) -> AIOptimizationRun:
    validate_run_transition(run.status, target)
    run.status = str(target).upper()
    now = utc_now()
    if run.status == "RUNNING" and run.started_at is None:
        run.started_at = now
    if run.status in {
        "COMPLETED",
        "INSUFFICIENT_DATA",
        "RESEARCH_PROVISIONAL",
        "INSUFFICIENT_EVIDENCE",
        "TECHNICAL_INVALID",
        "FAILED",
        "REJECTED",
        "CANCELLED",
        "ROLLED_BACK",
    }:
        run.finished_at = now
    elif run.status == "SHADOW":
        # A provisional research result may be promoted into SHADOW. It is
        # no longer a finished run once passive observation starts.
        run.finished_at = None
    if reason:
        existing = run.summary or ""
        run.summary = (
            (existing + "\n" + reason).strip()[:4000]
            if existing
            else reason[:4000]
        )
    session.add(run)
    await session.flush()
    return run


async def append_step(
    session: AsyncSession,
    run_id: int,
    *,
    step_index: int,
    step_type: str,
    status: str = "SUCCEEDED",
    hypothesis: str | None = None,
    action: str | None = None,
    input_payload: Mapping[str, Any] | None = None,
    output_payload: Mapping[str, Any] | None = None,
    summary: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AIRunStep:
    if step_index < 0:
        raise AILabError("step_index must be non-negative")
    status = status.upper()
    if status not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED"}:
        raise AILabError(f"unsupported step status: {status}")
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    now = utc_now()
    row = AIRunStep(
        run_id=run_id,
        step_index=step_index,
        step_type=step_type,
        status=status,
        hypothesis=hypothesis,
        action=action,
        input_payload=dict(input_payload) if input_payload is not None else None,
        output_payload=dict(output_payload) if output_payload is not None else None,
        summary=summary,
        error_code=error_code,
        error_message=error_message,
        created_at=now,
        started_at=now if status in {"RUNNING", "SUCCEEDED", "FAILED"} else None,
        finished_at=now if status in {"SUCCEEDED", "FAILED", "SKIPPED"} else None,
    )
    session.add(row)
    await session.flush()
    return row


async def create_experiment_config(
    session: AsyncSession,
    *,
    name: str,
    model_family: str,
    feature_set: str,
    feature_pipeline_version: str,
    model_params: Mapping[str, Any],
    strategy_params: Mapping[str, Any],
    backtest_params: Mapping[str, Any],
    config_hash: str | None = None,
    asset: str | None = None,
    regime: str | None = None,
    description: str | None = None,
    created_by: str = "system",
    parent_id: int | None = None,
) -> AIExperimentConfig:
    payload = {
        "name": name,
        "asset": asset,
        "regime": regime,
        "model_family": model_family,
        "feature_set": feature_set,
        "feature_pipeline_version": feature_pipeline_version,
        "model_params": dict(model_params),
        "strategy_params": dict(strategy_params),
        "backtest_params": dict(backtest_params),
        "parent_id": parent_id,
    }
    config_hash = config_hash or compute_manifest_hash(payload)
    row = AIExperimentConfig(
        **payload,
        description=description,
        config_hash=config_hash,
        created_by=created_by,
        created_at=utc_now(),
    )
    session.add(row)
    await session.flush()
    return row


async def record_deployment_event(
    session: AsyncSession,
    *,
    revision_id: int,
    event_type: str,
    actor: str = "system",
    reason: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> DeploymentEvent:
    """Record an append-only deployment audit event in the per-revision cryptographic chain."""
    event_type = event_type.strip().upper()
    if event_type not in {
        "CREATED",
        "SHADOW_ASSIGNED",
        "APPROVED",
        "ACTIVATED",
        "REJECTED",
        "ROLLED_BACK",
    }:
        raise AILabError(f"unsupported deployment event type: {event_type}")

    # Lock the parent revision first. Locking only the latest event is
    # insufficient when the chain is empty: concurrent genesis events would
    # both observe the all-zero predecessor.
    revision = (
        await session.execute(
            select(DeploymentRevision)
            .where(DeploymentRevision.id == revision_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if revision is None:
        raise AILabError(f"deployment revision {revision_id} not found")

    last_event = (
        await session.execute(
            select(DeploymentEvent)
            .where(DeploymentEvent.revision_id == revision_id)
            .order_by(DeploymentEvent.id.desc())
            .with_for_update()
            .limit(1)
        )
    ).scalar_one_or_none()
    previous_hash = last_event.event_hash if last_event else ("0" * 64)

    now = utc_now()
    event_dict = {
        "revision_id": revision_id,
        "event_type": event_type,
        "actor": actor,
        "reason": reason,
        "payload": dict(payload or {}),
        "previous_hash": previous_hash,
        "timestamp": now.isoformat(),
    }
    event_hash = compute_manifest_hash(event_dict)

    event = DeploymentEvent(
        revision_id=revision_id,
        event_type=event_type,
        actor=actor,
        reason=reason,
        payload=dict(payload or {}),
        previous_hash=previous_hash,
        event_hash=event_hash,
        created_at=now,
    )
    session.add(event)
    await session.flush()
    return event


async def create_deployment_revision(
    session: AsyncSession,
    *,
    revision_key: str,
    manifest: Mapping[str, Any],
    parent_id: int | None = None,
    status: str = "PENDING_APPROVAL",
    created_by: str = "system",
) -> DeploymentRevision:
    """Create an immutable deployment bundle with content-addressed manifest hash.

    Idempotency only reuses active/pending revisions; completed/rolled-back revisions
    always generate a new revision instance.
    """
    status = status.strip().upper()
    if status not in {
        "DRAFT",
        "SHADOW",
        "PENDING_APPROVAL",
        "ACTIVE",
        "REJECTED",
        "ROLLED_BACK",
    }:
        raise AILabError(f"unsupported deployment revision status: {status}")

    checked_manifest = build_deployment_manifest(manifest)
    manifest_hash = checked_manifest["manifest_hash"]

    existing = (
        await session.execute(
            select(DeploymentRevision).where(
                DeploymentRevision.manifest_hash == manifest_hash,
                DeploymentRevision.status.in_(
                    {"DRAFT", "SHADOW", "PENDING_APPROVAL"}
                ),
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    revision = DeploymentRevision(
        revision_key=revision_key,
        parent_id=parent_id,
        manifest=checked_manifest,
        manifest_hash=manifest_hash,
        status=status,
        created_by=created_by,
        created_at=utc_now(),
    )
    session.add(revision)
    await session.flush()
    return revision


async def generate_deployment_diff(
    session: AsyncSession,
    *,
    run_id: int,
    candidate_config_id: int | None = None,
    candidate_artifact_id: int | None = None,
) -> dict[str, Any]:
    """Generate a structured diff comparing the candidate against the active baseline."""
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")

    resolved_config_id = candidate_config_id
    resolved_artifact_id = candidate_artifact_id
    resolved_metrics: dict[str, Any] = {}
    resolved_asset: str | None = None
    resolved_regime: str | None = None

    if run.summary:
        try:
            summary_data = (
                json.loads(run.summary)
                if isinstance(run.summary, str)
                else run.summary
            )
            report = summary_data.get("report", summary_data)
            if not resolved_config_id:
                resolved_config_id = report.get("recommended_config_id")
            if resolved_config_id and "rows" in report:
                for row in report["rows"]:
                    if row.get("config_id") == resolved_config_id:
                        resolved_metrics = {
                            "median_pnl": row.get("median_oot_pnl"),
                            "total_trades": row.get("total_trades"),
                            "max_drawdown": row.get("median_oot_drawdown"),
                            "window_count": row.get("window_count"),
                        }
                        if not resolved_artifact_id and row.get("artifact_ids"):
                            resolved_artifact_id = int(row["artifact_ids"][0])
                        break
        except Exception as exc:
            logger.warning(
                "ai_lab_diff_summary_parse_failed",
                run_id=run_id,
                error=str(exc),
            )

    if not resolved_artifact_id or not resolved_config_id:
        shadow_assign = (
            await session.execute(
                select(AIShadowAssignment)
                .where(AIShadowAssignment.run_id == run_id)
                .order_by(AIShadowAssignment.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if shadow_assign:
            resolved_artifact_id = (
                resolved_artifact_id or shadow_assign.candidate_artifact_id
            )
            resolved_asset = resolved_asset or shadow_assign.asset
            resolved_regime = resolved_regime or shadow_assign.regime

    config: AIExperimentConfig | None = None
    if resolved_config_id:
        config = await session.get(AIExperimentConfig, resolved_config_id)
    elif resolved_artifact_id:
        art = await session.get(AIModelArtifact, resolved_artifact_id)
        if art and art.artifact_metadata:
            cfg_id = art.artifact_metadata.get("config_id")
            if cfg_id:
                config = await session.get(AIExperimentConfig, int(cfg_id))

    if config is None:
        raise AILabError(
            f"could not resolve candidate experiment config for run {run_id}"
        )

    resolved_asset = resolved_asset or config.asset or "BTCUSDT"
    resolved_regime = resolved_regime or config.regime

    active_row = (
        await session.execute(
            select(ModelRegistry)
            .where(
                ModelRegistry.asset == resolved_asset,
                ModelRegistry.is_active.is_(True),
            )
            .order_by(ModelRegistry.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    baseline_info = {
        "model_registry_id": active_row.id if active_row else None,
        "version": active_row.version if active_row else None,
        "features": active_row.features if active_row else None,
        "model_type": active_row.model_type if active_row else None,
        "decision_threshold": active_row.decision_threshold if active_row else None,
        "decision_threshold_down": active_row.decision_threshold_down
        if active_row
        else None,
        "accuracy": active_row.accuracy if active_row else None,
        "backtest_pnl": active_row.backtest_pnl if active_row else None,
        "backtest_trades": active_row.backtest_trades if active_row else None,
    }

    strategy_params = config.strategy_params or {}
    candidate_info = {
        "config_id": config.id,
        "artifact_id": resolved_artifact_id,
        "model_family": config.model_family,
        "feature_set": config.feature_set,
        "feature_pipeline_version": config.feature_pipeline_version,
        "decision_threshold": strategy_params.get("decision_threshold"),
        "decision_threshold_down": strategy_params.get("decision_threshold_down"),
        "model_params": config.model_params,
        "strategy_params": config.strategy_params,
    }

    return {
        "asset": resolved_asset,
        "regime": resolved_regime,
        "candidate": candidate_info,
        "baseline": baseline_info,
        "metrics": resolved_metrics,
    }


async def propose_live_deployment(
    session: AsyncSession,
    *,
    run_id: int,
    actor: str = "system",
    reason: str | None = None,
) -> tuple[AIApprovalRequest, DeploymentRevision]:
    """Build a server-side diff and create a DeploymentRevision for human review."""
    # Serialize proposals for the same run. Without this lock, two
    # concurrent requests can both create a pending revision and approval.
    run = (
        await session.execute(
            select(AIOptimizationRun)
            .where(AIOptimizationRun.id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if str(getattr(run, "mode", settings.AI_LAB_MODE)).upper() == "RESEARCH":
        raise AIResearchModeError(
            "LIVE deployment proposals are prohibited for RESEARCH runs"
        )
    if run.status not in {"SHADOW", "PENDING_APPROVAL"}:
        raise AILabError(
            f"live deployment proposal requires run in SHADOW or PENDING_APPROVAL, got {run.status}"
        )

    # Check for existing pending approval request to maintain idempotency with early return
    existing_approval = (
        await session.execute(
            select(AIApprovalRequest).where(
                AIApprovalRequest.run_id == run_id,
                AIApprovalRequest.requested_action == "ACTIVATE",
                AIApprovalRequest.status == "PENDING",
            )
            .with_for_update()
            .limit(1)
        )
    ).scalar_one_or_none()

    if existing_approval is not None:
        existing_revision: DeploymentRevision | None = None
        if existing_approval.target_id and existing_approval.target_id.isdigit():
            existing_revision = await session.get(
                DeploymentRevision, int(existing_approval.target_id)
            )
        if existing_revision is not None:
            return existing_approval, existing_revision

    diff = await generate_deployment_diff(session, run_id=run_id)
    candidate = diff["candidate"]
    asset = diff["asset"]

    parent_revision = (
        await session.execute(
            select(DeploymentRevision)
            .where(DeploymentRevision.status == "ACTIVE")
            .order_by(DeploymentRevision.id.desc())
            .with_for_update()
            .limit(1)
        )
    ).scalar_one_or_none()
    parent_id = parent_revision.id if parent_revision else None

    manifest_payload = {
        "models": [
            {
                "asset": asset,
                "config_id": candidate["config_id"],
                "artifact_id": candidate["artifact_id"],
                "model_family": candidate["model_family"],
                "feature_set": candidate["feature_set"],
            }
        ],
        "strategy": {
            "asset": asset,
            "decision_threshold": candidate.get("decision_threshold"),
            "decision_threshold_down": candidate.get("decision_threshold_down"),
            "params": candidate.get("strategy_params", {}),
        },
        "risk_policy": {
            "max_drawdown_threshold": run.scope.get("max_drawdown")
            if run.scope
            else None,
            "min_trades_required": run.scope.get("min_trades", 50)
            if run.scope
            else 50,
        },
        "execution_policy": {
            "dry_run": False,
            "safe_switching": True,
            "preserve_open_positions": True,
        },
    }

    checked_manifest = build_deployment_manifest(manifest_payload)
    manifest_hash = checked_manifest["manifest_hash"]
    revision_key = (
        f"rev_{run_id}_{candidate['config_id']}_{manifest_hash[:12]}_"
        f"{uuid.uuid4().hex[:10]}"
    )
    revision = await create_deployment_revision(
        session,
        revision_key=revision_key,
        manifest=manifest_payload,
        parent_id=parent_id,
        status="PENDING_APPROVAL",
        created_by=actor,
    )

    approval = AIApprovalRequest(
        run_id=run_id,
        target_type="DEPLOYMENT_REVISION",
        target_id=str(revision.id),
        requested_action="ACTIVATE",
        diff=diff,
        status="PENDING",
        requested_at=utc_now(),
    )
    session.add(approval)

    if run.status == "SHADOW":
        run = (
            await session.execute(
                select(AIOptimizationRun)
                .where(AIOptimizationRun.id == run_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if run and run.status == "SHADOW":
            proposal_reason = (
                f"Deployment proposed by {actor}"
                + (f": {reason}" if reason else "")
            )
            await transition_run(
                session, run, "PENDING_APPROVAL", reason=proposal_reason
            )

    await record_deployment_event(
        session,
        revision_id=revision.id,
        event_type="CREATED",
        actor=actor,
        reason=reason or f"Deployment proposal created for run {run_id}",
        payload={"diff": diff},
    )

    await session.flush()
    return approval, revision


async def approve_and_activate_deployment(
    session: AsyncSession,
    *,
    approval_id: int,
    actor: str,
    reason: str | None = None,
) -> DeploymentRevision:
    """Administratively activate a deployment revision with row-locked pointer switching."""
    approval = (
        await session.execute(
            select(AIApprovalRequest)
            .where(AIApprovalRequest.id == approval_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if approval is None:
        raise AILabError(f"approval request {approval_id} not found")
    if approval.status != "PENDING":
        raise AILabError(
            f"approval request {approval_id} is already {approval.status}"
        )
    if approval.requested_action != "ACTIVATE" or approval.target_type != "DEPLOYMENT_REVISION":
        raise AILabError("approval request is not a deployment activation")
    if approval.run_id:
        approval_run = await session.get(AIOptimizationRun, approval.run_id)
        if approval_run is not None and str(getattr(approval_run, "mode", "STANDARD")).upper() == "RESEARCH":
            raise AIResearchModeError(
                "LIVE activation is prohibited for RESEARCH runs"
            )
    try:
        revision_id = int(approval.target_id)
    except (TypeError, ValueError) as exc:
        raise AILabError("approval target_id must reference a deployment revision") from exc
    revision = (
        await session.execute(
            select(DeploymentRevision)
            .where(DeploymentRevision.id == revision_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if revision is None:
        raise AILabError(f"deployment revision {revision_id} not found")
    if revision.status != "PENDING_APPROVAL":
        raise AILabError(
            f"deployment revision {revision_id} must be PENDING_APPROVAL, got {revision.status}"
        )

    manifest = revision.manifest or {}
    models_info = manifest.get("models", [])
    if not models_info:
        raise AILabError(f"revision {revision_id} manifest has no models defined")

    models_to_activate: list[tuple[str, ModelRegistry]] = []
    seen_assets: set[str] = set()
    for model_desc in models_info:
        asset = model_desc.get("asset")
        if not asset:
            continue
        if asset in seen_assets:
            raise AILabError(f"revision {revision.id} contains duplicate asset {asset!r}")
        seen_assets.add(asset)
        artifact_id = model_desc.get("artifact_id")
        if not artifact_id:
            raise AILabError(
                f"revision {revision.id} manifest model for asset '{asset}' has no artifact_id"
            )
        artifact = await session.get(AIModelArtifact, artifact_id)
        if artifact is None or artifact.model_registry_id is None:
            raise AILabError(
                f"artifact {artifact_id} has no linked ModelRegistry entry"
            )
        cand_model = (
            await session.execute(
                select(ModelRegistry)
                .where(ModelRegistry.id == artifact.model_registry_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if cand_model is None:
            raise AILabError(
                f"linked ModelRegistry entry {artifact.model_registry_id} for artifact {artifact_id} not found"
            )
        if cand_model.asset != asset:
            raise AILabError(
                f"artifact {artifact_id} model asset {cand_model.asset!r} does not match manifest asset {asset!r}"
            )
        models_to_activate.append((asset, cand_model))

    if not models_to_activate:
        raise AILabError(f"revision {revision.id} has no valid model entries")
    for asset, cand_model in models_to_activate:
        active_models = (
            await session.execute(
                select(ModelRegistry)
                .where(
                    ModelRegistry.asset == asset,
                    ModelRegistry.is_active.is_(True),
                )
                .with_for_update()
            )
        ).scalars().all()
        for old_model in active_models:
            old_model.is_active = False
        cand_model.is_active = True

    prev_active_revisions = (
        await session.execute(
            select(DeploymentRevision)
            .where(
                DeploymentRevision.status == "ACTIVE",
                DeploymentRevision.id != revision.id,
            )
            .with_for_update()
        )
    ).scalars().all()
    for prev_rev in prev_active_revisions:
        prev_rev.status = "SUPERSEDED"

    now = utc_now()
    revision.status = "ACTIVE"
    revision.activated_at = now

    approval.status = "APPROVED"
    approval.decided_at = now
    approval.decided_by = actor
    approval.decision_reason = reason

    if approval.run_id:
        run = (
            await session.execute(
                select(AIOptimizationRun)
                .where(AIOptimizationRun.id == approval.run_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if run and run.status == "PENDING_APPROVAL":
            activation_reason = (
                f"Activated by {actor}" + (f": {reason}" if reason else "")
            )
            # ACTIVE is a privileged state: the deployment transaction has
            # already validated the human approval and row-locked model pointers.
            # Do not route through the public transition graph, which must not
            # expose a direct PENDING_APPROVAL -> ACTIVE path.
            run.status = "ACTIVE"
            existing_summary = run.summary or ""
            run.summary = (
                (existing_summary + "\n" + activation_reason).strip()[:4000]
                if existing_summary
                else activation_reason[:4000]
            )

    await record_deployment_event(
        session,
        revision_id=revision.id,
        event_type="APPROVED",
        actor=actor,
        reason=reason or "Human administrative approval",
        payload={"approval_id": approval_id},
    )
    await record_deployment_event(
        session,
        revision_id=revision.id,
        event_type="ACTIVATED",
        actor=actor,
        reason=reason or "Human administrative activation",
        payload={"approval_id": approval_id, "manifest": manifest},
    )

    await session.flush()
    return approval, revision


async def reject_deployment_approval(
    session: AsyncSession,
    *,
    approval_id: int,
    actor: str,
    reason: str | None = None,
) -> tuple[AIApprovalRequest, DeploymentRevision | None]:
    """Reject a proposed deployment revision."""
    approval = (
        await session.execute(
            select(AIApprovalRequest)
            .where(AIApprovalRequest.id == approval_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if approval is None:
        raise AILabError(f"approval request {approval_id} not found")
    if approval.status != "PENDING":
        raise AILabError(
            f"approval request {approval_id} is already {approval.status}"
        )

    revision_id = int(approval.target_id)
    revision = (
        await session.execute(
            select(DeploymentRevision)
            .where(DeploymentRevision.id == revision_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if revision is not None:
        revision.status = "REJECTED"

    now = utc_now()
    approval.status = "REJECTED"
    approval.decided_at = now
    approval.decided_by = actor
    approval.decision_reason = reason

    if approval.run_id:
        run = (
            await session.execute(
                select(AIOptimizationRun)
                .where(AIOptimizationRun.id == approval.run_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if run and run.status == "PENDING_APPROVAL":
            rejection_reason = (
                f"Rejected by {actor}" + (f": {reason}" if reason else "")
            )
            await transition_run(
                session,
                run,
                "REJECTED",
                reason=rejection_reason,
            )

    await record_deployment_event(
        session,
        revision_id=revision_id,
        event_type="REJECTED",
        actor=actor,
        reason=reason or "Human rejected deployment proposal",
        payload={"approval_id": approval_id},
    )

    await session.flush()
    return approval, revision


async def rollback_deployment(
    session: AsyncSession,
    *,
    target_revision_id: int | None = None,
    actor: str = "admin",
    reason: str | None = None,
) -> tuple[DeploymentRevision | None, DeploymentRevision]:
    """Roll back to a parent or target deployment revision without touching open positions."""
    current_active = (
        await session.execute(
            select(DeploymentRevision)
            .where(DeploymentRevision.status == "ACTIVE")
            .order_by(DeploymentRevision.id.desc())
            .with_for_update()
            .limit(1)
        )
    ).scalar_one_or_none()

    resolved_target_id = target_revision_id
    if resolved_target_id is None:
        if current_active and current_active.parent_id:
            resolved_target_id = current_active.parent_id
        else:
            raise AILabError("no valid rollback target revision found")

    target_revision = (
        await session.execute(
            select(DeploymentRevision)
            .where(DeploymentRevision.id == resolved_target_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if target_revision is None:
        raise AILabError(f"target rollback revision {resolved_target_id} not found")
    if target_revision.status != "SUPERSEDED":
        raise AILabError(
            f"target rollback revision {resolved_target_id} is in status '{target_revision.status}', "
            "only 'SUPERSEDED' revisions can be targeted for rollback"
        )
    if current_active is not None and target_revision.id == current_active.id:
        raise AILabError("target rollback revision is already active")

    target_manifest = target_revision.manifest or {}
    target_models = target_manifest.get("models", [])
    if not target_models:
        raise AILabError(
            f"target rollback revision {resolved_target_id} has no models in manifest"
        )

    models_to_activate: list[tuple[str, ModelRegistry]] = []
    seen_assets: set[str] = set()
    for model_desc in target_models:
        asset = model_desc.get("asset")
        if not asset:
            continue
        if asset in seen_assets:
            raise AILabError(f"revision {target_revision.id} contains duplicate asset {asset!r}")
        seen_assets.add(asset)
        artifact_id = model_desc.get("artifact_id")
        if not artifact_id:
            raise AILabError(
                f"revision {target_revision.id} manifest model for asset '{asset}' has no artifact_id"
            )
        artifact = await session.get(AIModelArtifact, artifact_id)
        if artifact is None or artifact.model_registry_id is None:
            raise AILabError(
                f"artifact {artifact_id} has no linked ModelRegistry entry"
            )
        cand_model = (
            await session.execute(
                select(ModelRegistry)
                .where(ModelRegistry.id == artifact.model_registry_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if cand_model is None:
            raise AILabError(
                f"linked ModelRegistry entry {artifact.model_registry_id} for artifact {artifact_id} not found"
            )
        if cand_model.asset != asset:
            raise AILabError(
                f"artifact {artifact_id} model asset {cand_model.asset!r} does not match manifest asset {asset!r}"
            )
        models_to_activate.append((asset, cand_model))

    if not models_to_activate:
        raise AILabError(
            f"target rollback revision {target_revision.id} has no valid model entries"
        )
    for asset, cand_model in models_to_activate:
        active_models = (
            await session.execute(
                select(ModelRegistry)
                .where(
                    ModelRegistry.asset == asset,
                    ModelRegistry.is_active.is_(True),
                )
                .with_for_update()
            )
        ).scalars().all()
        for old_model in active_models:
            old_model.is_active = False
        cand_model.is_active = True

    now = utc_now()
    if current_active is not None:
        current_active.status = "ROLLED_BACK"
        current_active.rolled_back_at = now

    target_revision.status = "ACTIVE"
    target_revision.activated_at = now

    rollback_payload = {
        "rolled_back_from_id": current_active.id if current_active else None,
        "restored_revision_id": target_revision.id,
    }
    if current_active is not None:
        await record_deployment_event(
            session,
            revision_id=current_active.id,
            event_type="ROLLED_BACK",
            actor=actor,
            reason=reason or f"Rolled back from revision {current_active.id}",
            payload=rollback_payload,
        )
    await record_deployment_event(
        session,
        revision_id=target_revision.id,
        event_type="ROLLED_BACK",
        actor=actor,
        reason=reason or f"Restored revision {target_revision.id}",
        payload=rollback_payload,
    )

    await session.flush()
    return current_active, target_revision


async def request_approval(
    session: AsyncSession,
    *,
    run_id: int | None,
    target_type: str,
    target_id: str,
    requested_action: str,
    diff: Mapping[str, Any],
) -> AIApprovalRequest:
    requested_action = requested_action.upper()
    if requested_action not in {"ACTIVATE", "ROLLBACK", "CHANGE_LIVE_POLICY"}:
        raise AILabError(f"unsupported approval action: {requested_action}")
    if run_id is not None:
        run = await session.get(AIOptimizationRun, run_id)
        if run is None:
            raise AILabError(f"AI Lab run {run_id} not found")
        if requested_action == "ACTIVATE" and run.status not in {
            "SHADOW",
            "PENDING_APPROVAL",
        }:
            raise AILabError(
                f"activation approval requires SHADOW or PENDING_APPROVAL, got {run.status}"
            )
        if requested_action == "ROLLBACK" and run.status not in {
            "ACTIVE",
            "SHADOW",
            "PENDING_APPROVAL",
        }:
            raise AILabError(
                f"rollback approval requires an assigned run, got {run.status}"
            )
    row = AIApprovalRequest(
        run_id=run_id,
        target_type=target_type,
        target_id=target_id,
        requested_action=requested_action,
        diff=dict(diff),
        status="PENDING",
        requested_at=utc_now(),
    )
    session.add(row)
    await session.flush()
    return row


async def get_run_detail(
    session: AsyncSession, run_id: int
) -> dict[str, Any] | None:
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        return None
    steps = (
        await session.execute(
            select(AIRunStep)
            .where(AIRunStep.run_id == run_id)
            .order_by(AIRunStep.step_index)
        )
    ).scalars().all()
    results = (
        await session.execute(
            select(ExperimentResult)
            .where(ExperimentResult.run_id == run_id)
            .order_by(ExperimentResult.created_at, ExperimentResult.id)
        )
    ).scalars().all()
    audits = (
        await session.execute(
            select(AIStepAuditLog)
            .where(AIStepAuditLog.run_id == run_id)
            .order_by(AIStepAuditLog.created_at, AIStepAuditLog.id)
        )
    ).scalars().all()
    approvals = (
        await session.execute(
            select(AIApprovalRequest)
            .where(AIApprovalRequest.run_id == run_id)
            .order_by(AIApprovalRequest.id.desc())
        )
    ).scalars().all()
    return {
        "run": run,
        "steps": list(steps),
        "results": list(results),
        "audits": list(audits),
        "approvals": list(approvals),
    }


async def create_permission(
    session: AsyncSession,
    *,
    profile_name: str,
    allowed_actions: list[str],
    scope: Mapping[str, Any],
    limits: Mapping[str, Any],
    updated_by: str = "system",
    enabled: bool = True,
) -> AIPermission:
    """Create an immutable permission version and make it the current version."""
    if not profile_name.strip():
        raise AILabError("permission profile_name must not be empty")
    actions = {str(item).upper() for item in allowed_actions}
    unknown = actions.difference(LAB_ACTIONS)
    if unknown:
        raise AIPermissionError(
            f"unknown permission actions: {sorted(unknown)}"
        )
    if any(
        action in {"ACTIVATE_LIVE", "CHANGE_LIVE_POLICY"} for action in actions
    ):
        raise AIPermissionError(
            "live activation and live policy changes are not autonomous actions"
        )
    for attempt in range(2):
        current_rows = (
            await session.execute(
                select(AIPermission)
                .where(
                    AIPermission.profile_name == profile_name.strip(),
                    AIPermission.is_current.is_(True),
                )
                .with_for_update()
            )
        ).scalars().all()
        next_version = (
            max((row.version for row in current_rows), default=0) + 1
        )
        for current in current_rows:
            current.is_current = False
        row = AIPermission(
            profile_name=profile_name.strip(),
            version=next_version,
            is_current=True,
            allowed_actions=sorted(actions),
            scope=dict(scope),
            limits=dict(limits),
            enabled=enabled,
            updated_by=updated_by,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        session.add(row)
        try:
            await session.flush()
            return row
        except IntegrityError as exc:
            await session.rollback()
            if attempt:
                raise AILabError(
                    f"concurrent permission creation for profile "
                    f"{profile_name!r}; retry the request"
                ) from exc
    raise AILabError(
        f"permission creation failed for profile {profile_name!r}"
    )


async def authorize_run_action(
    session: AsyncSession,
    run_id: int,
    action: str,
) -> AIOptimizationRun:
    run = await session.get(AIOptimizationRun, run_id)
    if run is None:
        raise AILabError(f"AI Lab run {run_id} not found")
    if run.permission_id is None:
        raise AIPermissionError("run has no permission snapshot")
    permission = await session.get(AIPermission, run.permission_id)
    validate_permission(permission, action)
    return run
