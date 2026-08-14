"""Safe lifecycle services for the autonomous AI laboratory.

Phase 2 deliberately stops at experiment orchestration and audit persistence.
It never activates a model, changes RuntimeSettings, or submits an order.
Phase 9 adds the secure human-in-the-loop activation, revision manifest tracking,
cryptographic event hash chaining, and deterministic zero-loss rollback.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

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
    AIShadowAssignment,
    AIStepAuditLog,
    DeploymentEvent,
    DeploymentRevision,
    ExperimentResult,
    ModelRegistry,
    Order,
    Position,
    RuntimeSettings,
)

logger = structlog.get_logger("polyflip.ai_lab.service")

IMMUTABLE_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "REJECTED"}
VALID_TRANSITIONS: dict[str, set[str]] = {
    "DRAFT": {"PLANNING", "CANCELLED"},
    "PLANNING": {"DATA_PREP", "FAILED", "CANCELLED"},
    "DATA_PREP": {"TRAINING", "FAILED", "CANCELLED"},
    "TRAINING": {"EVALUATING", "FAILED", "CANCELLED"},
    "EVALUATING": {"SHADOW", "FAILED", "CANCELLED"},
    "SHADOW": {"PENDING_APPROVAL", "ACTIVE", "COMPLETED", "CANCELLED", "FAILED"},
    "PENDING_APPROVAL": {"ACTIVE", "REJECTED", "CANCELLED", "FAILED"},
    "ACTIVE": {"SUPERSEDED", "ROLLED_BACK", "CANCELLED", "COMPLETED"},
    "SUPERSEDED": {"ROLLED_BACK"},
    "ROLLED_BACK": set(),
    "REJECTED": set(),
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

DEFAULT_MAX_ACTIVE_EXPERIMENTS = 3
DEFAULT_ALLOWED_FAMILIES = [
    "LogisticRegression",
    "RandomForest",
    "LightGBM",
    "XGBoost",
]
DEFAULT_ALLOWED_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_MAX_BUDGET_USD = 100.0


class AILabError(Exception):
    """Domain error for AI Lab lifecycle rules."""


class AIPermissionError(AILabError):
    """Raised when an operation violates permission constraints."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compute_event_hash(
    *,
    revision_id: int,
    event_type: str,
    actor: str,
    timestamp: datetime,
    previous_hash: str,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Deterministic SHA-256 hash for immutable deployment event chaining."""
    serialized_payload = json.dumps(
        payload or {},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    raw = (
        f"{revision_id}|{event_type}|{actor}|{timestamp.isoformat()}|"
        f"{previous_hash}|{serialized_payload}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def record_deployment_event(
    session: AsyncSession,
    *,
    revision_id: int,
    event_type: str,
    actor: str,
    reason: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> DeploymentEvent:
    """Record an immutable event in the deployment hash chain for a specific revision."""
    last_event = (
        await session.execute(
            select(DeploymentEvent)
            .where(DeploymentEvent.revision_id == revision_id)
            .order_by(DeploymentEvent.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    previous_hash = last_event.event_hash if last_event else ("0" * 64)
    now = utc_now()
    event_hash = compute_event_hash(
        revision_id=revision_id,
        event_type=event_type,
        actor=actor,
        timestamp=now,
        previous_hash=previous_hash,
        payload=payload,
    )

    event = DeploymentEvent(
        revision_id=revision_id,
        event_type=event_type,
        actor=actor,
        event_hash=event_hash,
        previous_hash=previous_hash,
        reason=reason,
        payload=dict(payload or {}),
        created_at=now,
    )
    session.add(event)
    await session.flush()
    return event


def validate_manifest_safety(manifest: Mapping[str, Any]) -> None:
    """Safety checks to ensure runtime invariants before live activation."""
    strategy = manifest.get("strategy") or {}
    up_thresh = strategy.get("decision_threshold")
    down_thresh = strategy.get("decision_threshold_down")

    if up_thresh is not None and not (0.50 <= float(up_thresh) <= 0.99):
        raise AILabError(
            f"safety invariant violation: decision_threshold {up_thresh} outside [0.50, 0.99]"
        )
    if down_thresh is not None and not (0.01 <= float(down_thresh) <= 0.50):
        raise AILabError(
            f"safety invariant violation: decision_threshold_down {down_thresh} outside [0.01, 0.50]"
        )


async def get_or_create_default_permission(session: AsyncSession) -> AIPermission:
    """Ensure baseline security constraints exist in the database."""
    permission = (
        await session.execute(
            select(AIPermission).order_by(AIPermission.id.desc()).limit(1)
        )
    ).scalar_one_or_none()

    if permission:
        return permission

    permission = AIPermission(
        max_active_experiments=DEFAULT_MAX_ACTIVE_EXPERIMENTS,
        allowed_model_families=DEFAULT_ALLOWED_FAMILIES,
        allowed_assets=DEFAULT_ALLOWED_ASSETS,
        max_compute_budget_usd=DEFAULT_MAX_BUDGET_USD,
        requires_human_approval_for_live=True,
        can_modify_runtime_settings=False,
        can_place_live_orders=False,
    )
    session.add(permission)
    await session.flush()
    return permission


def validate_permission(
    permission: AIPermission | None,
    action: str,
    *,
    family: str | None = None,
    asset: str | None = None,
) -> None:
    if permission is None:
        raise AIPermissionError("No AI permissions snapshot found")

    action = action.upper()
    if action == "LIVE_DEPLOYMENT" and permission.requires_human_approval_for_live:
        raise AIPermissionError(
            "Live deployment requires explicit human approval policy"
        )
    if action == "MODIFY_RUNTIME" and not permission.can_modify_runtime_settings:
        raise AIPermissionError("Autonomous modification of RuntimeSettings is prohibited")
    if action == "LIVE_ORDER" and not permission.can_place_live_orders:
        raise AIPermissionError("Autonomous live order placement is prohibited")

    if family:
        allowed_families = set(permission.allowed_model_families or [])
        if allowed_families and family not in allowed_families:
            raise AIPermissionError(f"Model family '{family}' is not permitted")

    if asset:
        allowed_assets = set(permission.allowed_assets or [])
        if allowed_assets and asset not in allowed_assets:
            raise AIPermissionError(f"Asset '{asset}' is not permitted")


async def create_optimization_run(
    session: AsyncSession,
    *,
    objective: str,
    scope: Mapping[str, Any] | None = None,
    autonomy_level: str = "AUTONOMOUS_SHADOW",
    created_by: str = "system",
) -> AIOptimizationRun:
    permission = await get_or_create_default_permission(session)
    scope_data = dict(scope or {})

    family = scope_data.get("model_family")
    asset = scope_data.get("asset")
    validate_permission(permission, "CREATE_RUN", family=family, asset=asset)

    run = AIOptimizationRun(
        objective=objective,
        scope=scope_data,
        autonomy_level=autonomy_level,
        status="DRAFT",
        created_by=created_by,
        permission_id=permission.id,
    )
    session.add(run)
    await session.flush()
    return run


async def transition_run(
    session: AsyncSession,
    run: AIOptimizationRun,
    target_status: str,
    *,
    reason: str | None = None,
) -> AIOptimizationRun:
    target = target_status.upper()
    current = run.status.upper() if run.status else "DRAFT"

    if current in IMMUTABLE_TERMINAL_STATUSES and target != current:
        raise AILabError(
            f"illegal transition from terminal status '{current}' to '{target}'"
        )

    allowed = VALID_TRANSITIONS.get(current, set())
    if target not in allowed and target != current:
        raise AILabError(
            f"illegal status transition from '{current}' to '{target}'. Allowed: {sorted(allowed)}"
        )

    run.status = target
    now = utc_now()
    if target in IMMUTABLE_TERMINAL_STATUSES:
        run.completed_at = now

    audit = AIStepAuditLog(
        run_id=run.id,
        step_id=None,
        action="TRANSITION_STATUS",
        inputs={"from": current, "to": target},
        outputs={"status": target},
        decision_reason=reason or f"Transition to {target}",
        passed_checks=True,
    )
    session.add(audit)
    await session.flush()
    return run


async def create_run_step(
    session: AsyncSession,
    *,
    run_id: int,
    step_type: str,
    sequence: int,
    inputs: Mapping[str, Any] | None = None,
) -> AIRunStep:
    step = AIRunStep(
        run_id=run_id,
        step_type=step_type.upper(),
        sequence=sequence,
        status="PENDING",
        inputs=dict(inputs or {}),
    )
    session.add(step)
    await session.flush()
    return step


async def record_step_audit(
    session: AsyncSession,
    *,
    run_id: int,
    step_id: int | None,
    action: str,
    decision_reason: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    passed_checks: bool = True,
    guardrail_failures: list[str] | None = None,
) -> AIStepAuditLog:
    audit = AIStepAuditLog(
        run_id=run_id,
        step_id=step_id,
        action=action,
        decision_reason=decision_reason,
        inputs=dict(inputs or {}),
        outputs=dict(outputs or {}),
        passed_checks=passed_checks,
        guardrail_failures=list(guardrail_failures or []),
    )
    session.add(audit)
    await session.flush()
    return audit


async def record_experiment_config(
    session: AsyncSession,
    *,
    name: str,
    asset: str,
    regime: str,
    model_family: str,
    feature_set: str,
    feature_pipeline_version: str,
    model_params: Mapping[str, Any],
    strategy_params: Mapping[str, Any],
    backtest_params: Mapping[str, Any],
    config_hash: str,
    parent_config_id: int | None = None,
) -> AIExperimentConfig:
    config = AIExperimentConfig(
        name=name,
        asset=asset,
        regime=regime,
        model_family=model_family,
        feature_set=feature_set,
        feature_pipeline_version=feature_pipeline_version,
        model_params=dict(model_params),
        strategy_params=dict(strategy_params),
        backtest_params=dict(backtest_params),
        config_hash=config_hash,
        parent_config_id=parent_config_id,
    )
    session.add(config)
    await session.flush()
    return config


async def record_model_artifact(
    session: AsyncSession,
    *,
    config_id: int,
    artifact_uri: str,
    artifact_hash: str,
    schema_version: str,
    feature_pipeline_version: str,
    artifact_metadata: Mapping[str, Any] | None = None,
    model_registry_id: int | None = None,
    loadability_status: str = "PENDING",
) -> AIModelArtifact:
    artifact = AIModelArtifact(
        config_id=config_id,
        artifact_uri=artifact_uri,
        artifact_hash=artifact_hash,
        schema_version=schema_version,
        feature_pipeline_version=feature_pipeline_version,
        artifact_metadata=dict(artifact_metadata or {}),
        model_registry_id=model_registry_id,
        loadability_status=loadability_status.upper(),
    )
    session.add(artifact)
    await session.flush()
    return artifact


async def record_experiment_result(
    session: AsyncSession,
    *,
    config_id: int,
    run_id: int,
    metrics: Mapping[str, Any],
    validation_status: str,
    validation_failures: list[str] | None = None,
) -> ExperimentResult:
    result = ExperimentResult(
        config_id=config_id,
        run_id=run_id,
        metrics=dict(metrics),
        validation_status=validation_status.upper(),
        validation_failures=list(validation_failures or []),
    )
    session.add(result)
    await session.flush()
    return result


async def assign_shadow_candidate(
    session: AsyncSession,
    *,
    run_id: int,
    candidate_artifact_id: int,
    asset: str,
    baseline_model_id: int | None = None,
    shadow_config: Mapping[str, Any] | None = None,
) -> AIShadowAssignment:
    assignment = AIShadowAssignment(
        run_id=run_id,
        candidate_artifact_id=candidate_artifact_id,
        baseline_model_id=baseline_model_id,
        asset=asset,
        status="RUNNING",
        shadow_config=dict(shadow_config or {}),
    )
    session.add(assignment)
    await session.flush()
    return assignment


async def create_deployment_revision(
    session: AsyncSession,
    *,
    revision_key: str,
    manifest: Mapping[str, Any],
    status: str = "DRAFT",
    parent_id: int | None = None,
    description: str | None = None,
) -> DeploymentRevision:
    manifest_dict = dict(manifest)
    manifest_hash = compute_manifest_hash(manifest_dict)

    existing = (
        await session.execute(
            select(DeploymentRevision)
            .where(
                DeploymentRevision.manifest_hash == manifest_hash,
                DeploymentRevision.status.in_(["DRAFT", "SHADOW", "PENDING_APPROVAL"]),
            )
            .limit(1)
        )
    ).scalar_one_or_none()

    if existing:
        return existing

    revision = DeploymentRevision(
        revision_key=revision_key,
        manifest=manifest_dict,
        manifest_hash=manifest_hash,
        parent_id=parent_id,
        status=status.upper(),
        description=description,
    )
    session.add(revision)
    await session.flush()
    return revision


async def propose_live_deployment(
    session: AsyncSession,
    *,
    run_id: int,
    actor: str = "system",
    reason: str | None = None,
) -> tuple[AIApprovalRequest, DeploymentRevision]:
    """Propose a live deployment revision with rich diff computation."""
    run = (
        await session.execute(
            select(AIOptimizationRun)
            .where(AIOptimizationRun.id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()

    if run is None:
        raise AILabError(f"run {run_id} not found")

    if run.status not in {"SHADOW", "PENDING_APPROVAL"}:
        raise AILabError(
            f"propose_live_deployment requires run in SHADOW or PENDING_APPROVAL status, currently '{run.status}'"
        )

    existing_approval = (
        await session.execute(
            select(AIApprovalRequest)
            .where(
                AIApprovalRequest.run_id == run_id,
                AIApprovalRequest.status == "PENDING",
            )
            .order_by(AIApprovalRequest.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    if existing_approval:
        rev_id = int(existing_approval.target_id) if existing_approval.target_id.isdigit() else None
        rev = await session.get(DeploymentRevision, rev_id) if rev_id else None
        if rev:
            return existing_approval, rev

    active_rev = (
        await session.execute(
            select(DeploymentRevision)
            .where(DeploymentRevision.status == "ACTIVE")
            .order_by(DeploymentRevision.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    shadow = (
        await session.execute(
            select(AIShadowAssignment)
            .where(AIShadowAssignment.run_id == run_id)
            .order_by(AIShadowAssignment.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    candidate_artifact = None
    config = None
    if shadow and shadow.candidate_artifact_id:
        candidate_artifact = await session.get(
            AIModelArtifact, shadow.candidate_artifact_id
        )
        if candidate_artifact:
            config = await session.get(
                AIExperimentConfig, candidate_artifact.config_id
            )

    asset = (
        config.asset
        if config
        else (run.scope or {}).get("asset", "BTCUSDT")
    )

    baseline_model = (
        await session.execute(
            select(ModelRegistry)
            .where(ModelRegistry.asset == asset, ModelRegistry.is_active.is_(True))
            .limit(1)
        )
    ).scalar_one_or_none()

    manifest = build_deployment_manifest(
        asset=asset,
        candidate_artifact=candidate_artifact,
        config=config,
        baseline_model=baseline_model,
    )
    validate_manifest_safety(manifest)

    revision_key = f"rev-{asset.lower()}-{run.id}-{utc_now().strftime('%Y%m%d%H%M%S')}"
    revision = await create_deployment_revision(
        session,
        revision_key=revision_key,
        manifest=manifest,
        status="PENDING_APPROVAL",
        parent_id=active_rev.id if active_rev else None,
        description=f"Generated from run #{run.id} ({run.objective})",
    )

    await record_deployment_event(
        session,
        revision_id=revision.id,
        event_type="CREATED",
        actor=actor,
        reason=reason or f"Proposed deployment for run #{run.id}",
        payload={"run_id": run.id, "manifest_hash": revision.manifest_hash},
    )

    parsed_summary: dict[str, Any] = {}
    if run.summary:
        if isinstance(run.summary, dict):
            parsed_summary = run.summary
        elif isinstance(run.summary, str):
            try:
                parsed_summary = json.loads(run.summary)
            except Exception as e:
                logger.warning("ai_lab_diff_summary_parse_failed", run_id=run.id, error=str(e))
                parsed_summary = {}

    report_data = parsed_summary.get("report", parsed_summary)
    metrics_block = {}
    if isinstance(report_data, dict):
        metrics_block = {
            "median_pnl": report_data.get("median_pnl") or report_data.get("median_oot_pnl"),
            "total_trades": report_data.get("total_trades"),
            "max_drawdown": report_data.get("max_drawdown") or report_data.get("median_oot_drawdown"),
        }

    candidate_diff = {
        "config_id": config.id if config else None,
        "artifact_id": candidate_artifact.id if candidate_artifact else None,
        "model_family": config.model_family if config else None,
        "feature_set": config.feature_set if config else None,
        "feature_pipeline_version": config.feature_pipeline_version if config else "1.0",
        "decision_threshold": (config.strategy_params or {}).get("decision_threshold") if config else None,
        "decision_threshold_down": (config.strategy_params or {}).get("decision_threshold_down") if config else None,
        "model_params": config.model_params if config else {},
    }

    baseline_diff = {
        "model_registry_id": baseline_model.id if baseline_model else None,
        "version": baseline_model.version if baseline_model else None,
        "model_type": baseline_model.model_type if baseline_model else None,
        "features": baseline_model.features if baseline_model else None,
        "decision_threshold": baseline_model.decision_threshold if baseline_model else None,
        "decision_threshold_down": baseline_model.decision_threshold_down if baseline_model else None,
        "backtest_pnl": baseline_model.backtest_pnl if baseline_model else None,
        "backtest_trades": baseline_model.backtest_trades if baseline_model else None,
    }

    diff_payload = {
        "candidate": candidate_diff,
        "baseline": baseline_diff,
        "metrics": metrics_block,
    }

    approval = await request_approval(
        session,
        run_id=run.id,
        target_type="DEPLOYMENT_REVISION",
        target_id=str(revision.id),
        requested_action="ACTIVATE",
        diff=diff_payload,
    )

    if run.status != "PENDING_APPROVAL":
        await transition_run(
            session,
            run,
            "PENDING_APPROVAL",
            reason=f"Deployment revision {revision.id} proposed by {actor}",
        )

    await session.flush()
    return approval, revision


async def approve_and_activate_deployment(
    session: AsyncSession,
    *,
    approval_id: int,
    actor: str,
    reason: str | None = None,
) -> tuple[AIApprovalRequest, DeploymentRevision]:
    """Approve a proposed live deployment revision and activate it transactionally."""
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
    if revision is None:
        raise AILabError(f"deployment revision {revision_id} not found")
    if revision.status not in {"PENDING_APPROVAL", "SHADOW", "DRAFT"}:
        raise AILabError(
            f"revision {revision_id} cannot be activated from status '{revision.status}'"
        )

    manifest = revision.manifest or {}
    validate_manifest_safety(manifest)

    models_desc = manifest.get("models", [])
    if not models_desc:
        raise AILabError(
            f"revision {revision_id} has no candidate models declared in manifest"
        )

    models_to_activate: list[tuple[str, ModelRegistry]] = []
    seen_assets: set[str] = set()
    for model_desc in models_desc:
        asset = model_desc.get("asset")
        if not asset:
            continue
        if asset in seen_assets:
            raise AILabError(f"revision {revision_id} contains duplicate asset {asset!r}")
        seen_assets.add(asset)
        artifact_id = model_desc.get("artifact_id")
        if not artifact_id:
            raise AILabError(
                f"revision {revision_id} manifest model for asset '{asset}' has no artifact_id"
            )
        artifact = await session.get(AIModelArtifact, artifact_id)
        if artifact is None or artifact.model_registry_id is None:
            raise AILabError(
                f"candidate artifact {artifact_id} has no linked ModelRegistry entry"
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
        raise AILabError(f"revision {revision_id} has no valid model entries")

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

    active_revisions = (
        await session.execute(
            select(DeploymentRevision)
            .where(
                DeploymentRevision.status == "ACTIVE",
                DeploymentRevision.id != revision.id,
            )
            .with_for_update()
        )
    ).scalars().all()

    now = utc_now()
    for old_rev in active_revisions:
        old_rev.status = "SUPERSEDED"
        old_rev.superseded_at = now
        await record_deployment_event(
            session,
            revision_id=old_rev.id,
            event_type="SUPERSEDED",
            actor=actor,
            reason=f"Superseded by activation of revision {revision.id}",
            payload={"superseded_by_revision_id": revision.id},
        )

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
            await transition_run(
                session,
                run,
                "ACTIVE",
                reason=activation_reason,
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
    if revision and revision.status in {"PENDING_APPROVAL", "SHADOW", "DRAFT"}:
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

    approval = AIApprovalRequest(
        run_id=run_id,
        target_type=target_type,
        target_id=target_id,
        requested_action=requested_action,
        status="PENDING",
        diff=dict(diff),
    )
    session.add(approval)
    await session.flush()
    return approval


async def get_run_detail(
    session: AsyncSession,
    run_id: int,
) -> dict[str, Any] | None:
    run = await session.get(AIOptimizationRun, run_id)
    if not run:
        return None

    steps = (
        await session.execute(
            select(AIRunStep)
            .where(AIRunStep.run_id == run_id)
            .order_by(AIRunStep.sequence, AIRunStep.id)
        )
    ).scalars().all()

    results = (
        await session.execute(
            select(ExperimentResult)
            .where(ExperimentResult.run_id == run_id)
            .order_by(ExperimentResult.id.desc())
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


async def list_optimization_runs(
    session: AsyncSession,
    *,
    status: str | None = None,
    limit: int = 50,
) -> list[AIOptimizationRun]:
    stmt = select(AIOptimizationRun).order_by(desc(AIOptimizationRun.id)).limit(limit)
    if status:
        stmt = stmt.where(AIOptimizationRun.status == status.upper())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_active_shadow_assignment(
    session: AsyncSession,
    *,
    asset: str,
) -> AIShadowAssignment | None:
    return (
        await session.execute(
            select(AIShadowAssignment)
            .where(
                AIShadowAssignment.asset == asset,
                AIShadowAssignment.status == "RUNNING",
            )
            .order_by(desc(AIShadowAssignment.id))
            .limit(1)
        )
    ).scalar_one_or_none()


async def check_guardrails(
    session: AsyncSession,
    run: AIOptimizationRun,
    action: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if run.permission_id:
        perm = await session.get(AIPermission, run.permission_id)
        if perm:
            if action.upper() == "LIVE_ORDER" and not perm.can_place_live_orders:
                failures.append("Action LIVE_ORDER violates AI permissions")
            if action.upper() == "MODIFY_RUNTIME" and not perm.can_modify_runtime_settings:
                failures.append("Action MODIFY_RUNTIME violates AI permissions")

    active_runs_count = (
        await session.execute(
            select(func.count(AIOptimizationRun.id)).where(
                AIOptimizationRun.status.in_(["PLANNING", "DATA_PREP", "TRAINING", "EVALUATING", "SHADOW"])
            )
        )
    ).scalar() or 0

    if active_runs_count > DEFAULT_MAX_ACTIVE_EXPERIMENTS:
        failures.append(
            f"Active experiment count ({active_runs_count}) exceeds limit ({DEFAULT_MAX_ACTIVE_EXPERIMENTS})"
        )

    return len(failures) == 0, failures


async def ensure_run_action_allowed(
    session: AsyncSession,
    run_id: int,
    action: str,
) -> AIOptimizationRun:
    run = await session.get(AIOptimizationRun, run_id)
    if not run:
        raise AILabError(f"Optimization run {run_id} not found")
    if run.permission_id is None:
        raise AIPermissionError("run has no permission snapshot")
    permission = await session.get(AIPermission, run.permission_id)
    validate_permission(permission, action)
    return run
