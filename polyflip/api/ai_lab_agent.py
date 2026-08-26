"""Typed HTTP API for the independent ``ai_research_agent`` container.

The external agent has no database, shell or docker access: every mutation is
performed here inside the API process, guarded by a per-run lease token.
"""

from __future__ import annotations

import hmac
import uuid
from datetime import timedelta
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.llm import (
    AgentDecision,
    HypothesisProposal,
    OpenAIResponsesProvider,
)
from polyflip.ai_lab.orchestrator import plan_run
from polyflip.ai_lab.service import (
    AIRunTransitionError,
    append_step,
    transition_run,
    utc_now,
)
from polyflip.config import settings
from polyflip.db.connection import get_db_session
from polyflip.db.models import (
    AIExperimentConfig,
    AIOptimizationRun,
    AIRunStep,
    AIWorkerLease,
    ExperimentResult,
    ModelRegistry,
    TradeHistory,
)


# ---------------------------------------------------------------------------
# Authentication: dedicated agent token with API-key bootstrap fallback
# ---------------------------------------------------------------------------
async def verify_agent_token(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization.split(" ", 1)[1].strip()
    if not supplied and x_api_key:
        supplied = x_api_key.strip()
    expected = getattr(settings, "AI_LAB_AGENT_TOKEN", "") or getattr(
        settings, "API_KEY", ""
    )
    if not expected or not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid agent token")


router = APIRouter(
    prefix="/api/ai-lab/agent",
    tags=["ai-lab-agent"],
    dependencies=[Depends(verify_agent_token)],
)

logger = structlog.get_logger("polyflip.api.ai_lab_agent")


class AgentClaimRequest(BaseModel):
    worker_id: str = Field(default="external-ai-research-agent", max_length=128)


def _lease_ttl_seconds() -> int:
    return int(getattr(settings, "AI_LAB_AGENT_LEASE_TTL_SECONDS", 120) or 120)


def _as_utc(value):
    """Normalize DB datetimes to aware UTC (SQLite drops tzinfo)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=__import__("datetime").timezone.utc)
    return value.astimezone(__import__("datetime").timezone.utc)


def _normalize_asset(raw: Any) -> str | None:
    """Normalize asset symbol: strip, upper, remove USDT suffix."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    if not s:
        return None
    # Handle common suffixes like BTCUSDT, BTC/USDT, BTC-USDT
    s = s.replace("/", "").replace("-", "")
    if s.endswith("USDT"):
        s = s[:-4]
    return s or None


async def _acquire_lease(db: AsyncSession, run_id: int, worker_id: str) -> str | None:
    now = utc_now()
    expires_at = now + timedelta(seconds=_lease_ttl_seconds())
    lease = (
        await db.execute(select(AIWorkerLease).where(AIWorkerLease.run_id == run_id))
    ).scalar_one_or_none()
    token = f"agent-{uuid.uuid4().hex}"
    if lease is None:
        db.add(
            AIWorkerLease(
                run_id=run_id,
                worker_id=worker_id,
                owner_token=token,
                acquired_at=now,
                heartbeat_at=now,
                expires_at=expires_at,
            )
        )
        await db.flush()
        return token
    if _as_utc(lease.expires_at) > now:
        return None  # actively held by another worker
    lease.worker_id = worker_id
    lease.owner_token = token
    lease.acquired_at = now
    lease.heartbeat_at = now
    lease.expires_at = expires_at
    await db.flush()
    return token


async def _verify_lease(db: AsyncSession, run_id: int, lease_token: str | None) -> None:
    if not lease_token:
        raise HTTPException(status_code=409, detail="LEASE_LOST")
    now = utc_now()
    lease = (
        await db.execute(select(AIWorkerLease).where(AIWorkerLease.run_id == run_id))
    ).scalar_one_or_none()
    if (
        lease is None
        or lease.owner_token != lease_token
        or _as_utc(lease.expires_at) <= now
    ):
        raise HTTPException(status_code=409, detail="LEASE_LOST")


async def _renew_lease(db: AsyncSession, run_id: int, lease_token: str | None):
    await _verify_lease(db, run_id, lease_token)
    now = utc_now()
    lease = (
        await db.execute(select(AIWorkerLease).where(AIWorkerLease.run_id == run_id))
    ).scalar_one()
    lease.heartbeat_at = now
    lease.expires_at = now + timedelta(seconds=_lease_ttl_seconds())
    await db.flush()
    return _as_utc(lease.expires_at)


async def _release_lease(db: AsyncSession, run_id: int) -> None:
    lease = (
        await db.execute(select(AIWorkerLease).where(AIWorkerLease.run_id == run_id))
    ).scalar_one_or_none()
    if lease is not None:
        await db.delete(lease)


async def _require_run(db: AsyncSession, run_id: int) -> AIOptimizationRun:
    run = await db.get(AIOptimizationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


def _config_id_from_step(step: AIRunStep | None) -> int | None:
    if step is None:
        return None
    for payload in (step.output_payload, step.input_payload):
        if isinstance(payload, dict) and payload.get("config_id") is not None:
            try:
                return int(payload["config_id"])
            except (TypeError, ValueError):
                return None
    return None


def _decision_payload(step: AIRunStep | None) -> dict[str, Any] | None:
    if step is None:
        return None
    decision = None
    if isinstance(step.input_payload, dict):
        value = step.input_payload.get("decision")
        if isinstance(value, dict):
            decision = value
    return {
        "action": step.action,
        "decision": decision,
        "output_payload": step.output_payload,
        "step_id": step.id,
    }


async def _current_iteration(db: AsyncSession, run_id: int) -> dict[str, Any]:
    steps = (
        (
            await db.execute(
                select(AIRunStep)
                .where(AIRunStep.run_id == run_id)
                .order_by(AIRunStep.step_index)
            )
        )
        .scalars()
        .all()
    )
    boundary = max(
        (step for step in steps if step.step_type == "ITERATION_REQUEUED"),
        key=lambda step: step.step_index,
        default=None,
    )
    cutoff = boundary.step_index if boundary is not None else -1
    proposal = next(
        (
            step
            for step in reversed(steps)
            if step.step_type == "PROPOSAL" and step.step_index > cutoff
        ),
        None,
    )
    config_id = _config_id_from_step(proposal)
    result = None
    failed_result = None
    if config_id is not None:
        results = (
            (
                await db.execute(
                    select(ExperimentResult)
                    .where(
                        ExperimentResult.run_id == run_id,
                        ExperimentResult.config_id == config_id,
                        ExperimentResult.evaluation_kind == "POLYMARKET_OOT",
                    )
                    .order_by(ExperimentResult.id.desc())
                )
            )
            .scalars()
            .all()
        )
        result = next((row for row in results if row.status == "SUCCEEDED"), None)
        failed_result = next(
            (row for row in results if row.status in {"FAILED", "INSUFFICIENT_DATA"}),
            None,
        )
    current_decision = None
    if result is not None:
        current_decision = next(
            (
                step
                for step in reversed(steps)
                if step.step_type == "DECISION"
                and isinstance(step.output_payload, dict)
                and step.output_payload.get("result_id") == result.id
            ),
            None,
        )
    latest_decision = next(
        (step for step in reversed(steps) if step.step_type == "DECISION"),
        None,
    )
    pending = bool(
        config_id is not None
        and any(
            step.status in {"PENDING", "RUNNING"}
            and step.step_type
            in {"TRAIN_MODEL", "RUN_OOT_BACKTEST", "RUN_POLYMARKET_OOT"}
            and isinstance(step.input_payload, dict)
            and step.input_payload.get("config_id") == config_id
            for step in steps
        )
    )
    return {
        "steps": steps,
        "boundary": boundary,
        "proposal": proposal,
        "config_id": config_id,
        "pending": pending,
        "result": result,
        "failed_result": failed_result,
        "current_decision": current_decision,
        "latest_decision": latest_decision,
    }


async def _agent_phase(db, run_id: int) -> dict[str, Any]:
    iteration = await _current_iteration(db, run_id)
    proposal = iteration["proposal"]
    term = iteration["result"]
    decision = iteration["current_decision"]
    if proposal is None:
        phase = "NEEDS_PROPOSAL"
    elif iteration["pending"]:
        phase = "WAITING_RESULT"
    elif term is not None and decision is None:
        phase = "NEEDS_DECISION"
    elif decision is not None:
        phase = "NEEDS_COMPLETION"
    else:
        phase = "WAITING_RESULT"
    return {
        "phase": phase,
        "latest_config_id": iteration["config_id"],
        "latest_result_id": term.id if term else None,
        "latest_decision": _decision_payload(decision or iteration["latest_decision"]),
    }


@router.get("/runs/{run_id}/phase")
async def get_agent_phase(
    run_id: int,
    lease_token: str,
    db: AsyncSession = Depends(get_db_session),
):
    await _require_run(db, run_id)
    await _verify_lease(db, run_id, lease_token)
    return await _agent_phase(db, run_id)


def _claimed_run_payload(run: AIOptimizationRun, lease_token: str) -> dict[str, Any]:
    snap = (
        run.llm_snapshot if isinstance(getattr(run, "llm_snapshot", None), dict) else {}
    )
    # Normalize to new per-model snapshot shape.
    research = snap.get("research") if isinstance(snap.get("research"), dict) else None
    summary = snap.get("summary") if isinstance(snap.get("summary"), dict) else None
    if research is None:
        # Fallback to flat fields
        research = {
            "model_id": run.llm_research_model,
            "protocol": snap.get("protocol") or "responses",
        }
    if summary is None:
        summary = {
            "model_id": run.llm_summary_model,
            "protocol": snap.get("protocol") or "responses",
        }
    return {
        "id": run.id,
        "status": run.status,
        "objective": run.objective,
        "scope": run.scope or {},
        "mode": run.mode,
        "autonomy_level": run.autonomy_level,
        "budget_experiments": run.budget_experiments,
        "budget_seconds": run.budget_seconds,
        "experiments_completed": run.experiments_completed,
        "llm_provider": run.llm_provider,
        "llm_research_model": run.llm_research_model,
        "llm_summary_model": run.llm_summary_model,
        "llm_snapshot": snap,
        "llm_research": research,
        "llm_summary": summary,
        "lease_token": lease_token,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/claim")
async def claim_next_agent_run(
    payload: AgentClaimRequest | None = None,
    db: AsyncSession = Depends(get_db_session),
):
    worker_id = (payload.worker_id if payload else None) or "external-ai-research-agent"
    now = utc_now()
    existing = (
        await db.execute(
            select(AIWorkerLease)
            .where(
                AIWorkerLease.worker_id == worker_id,
                AIWorkerLease.expires_at > now,
            )
            .order_by(AIWorkerLease.heartbeat_at.desc(), AIWorkerLease.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        run = await db.get(AIOptimizationRun, existing.run_id)
        if run is not None and run.status in {
            "QUEUED",
            "PLANNING",
            "RUNNING",
            "EVALUATING",
        }:
            return {"run": _claimed_run_payload(run, existing.owner_token)}

    candidate = (
        await db.execute(
            select(AIOptimizationRun)
            .outerjoin(
                AIWorkerLease,
                AIWorkerLease.run_id == AIOptimizationRun.id,
            )
            .where(
                AIOptimizationRun.status.in_(
                    ["QUEUED", "PLANNING", "RUNNING", "EVALUATING"]
                ),
                or_(
                    AIWorkerLease.id.is_(None),
                    AIWorkerLease.expires_at <= now,
                ),
            )
            .order_by(AIOptimizationRun.id)
            .with_for_update(of=AIOptimizationRun, skip_locked=True)
            .limit(1)
        )
    ).scalar_one_or_none()
    if candidate is None:
        await db.rollback()
        return {"run": None}

    token = f"agent-{uuid.uuid4().hex}"
    expires_at = now + timedelta(seconds=_lease_ttl_seconds())
    lease = (
        await db.execute(
            select(AIWorkerLease).where(AIWorkerLease.run_id == candidate.id)
        )
    ).scalar_one_or_none()
    try:
        if lease is None:
            db.add(
                AIWorkerLease(
                    run_id=candidate.id,
                    worker_id=worker_id,
                    owner_token=token,
                    acquired_at=now,
                    heartbeat_at=now,
                    expires_at=expires_at,
                )
            )
        else:
            lease.worker_id = worker_id
            lease.owner_token = token
            lease.acquired_at = now
            lease.heartbeat_at = now
            lease.expires_at = expires_at
        await db.flush()
        if candidate.status == "QUEUED":
            await transition_run(
                db,
                candidate,
                "RUNNING",
                reason="agent claimed",
            )
        run_id = int(candidate.id)
        await db.commit()
    except (IntegrityError, AIRunTransitionError):
        await db.rollback()
        return {"run": None}

    run = await _require_run(db, run_id)
    logger.info("agent_run_claimed", run_id=run_id, worker=worker_id)
    return {"run": _claimed_run_payload(run, token)}


class HeartbeatRequest(BaseModel):
    run_id: int
    lease_token: str


@router.post("/heartbeat")
async def agent_heartbeat(
    payload: HeartbeatRequest, db: AsyncSession = Depends(get_db_session)
):
    await _require_run(db, payload.run_id)
    try:
        expires_at = await _renew_lease(db, payload.run_id, payload.lease_token)
        await db.commit()
    except HTTPException as exc:
        await db.rollback()
        raise exc
    # Return real expiry, not now.
    iso = (
        expires_at.isoformat().replace("+00:00", "Z")
        if expires_at
        else utc_now().isoformat().replace("+00:00", "Z")
    )
    return {
        "run_id": payload.run_id,
        "leased_until": iso,
    }


@router.get("/runs/{run_id}/context")
async def get_agent_context(
    run_id: int,
    lease_token: str,
    db: AsyncSession = Depends(get_db_session),
):
    run = await _require_run(db, run_id)
    await _verify_lease(db, run_id, lease_token)

    # Asset-aware filtering: scope asset normalized (BTC/BTCUSDT -> BTC)
    scope = run.scope if isinstance(run.scope, dict) else {}
    raw_asset = scope.get("asset")
    # Support alternative key naming
    if raw_asset is None:
        raw_asset = scope.get("scope_asset")
    normalized_asset = _normalize_asset(raw_asset)
    # ModelRegistry filtered by asset (same normalization)
    # Fetch then filter in python to handle USDT variants robustly.
    candidate_models = (
        (
            await db.execute(
                select(ModelRegistry)
                .where(ModelRegistry.is_active.is_(True))
                .order_by(ModelRegistry.id.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    if normalized_asset:
        filtered_models = [
            r for r in candidate_models if _normalize_asset(r.asset) == normalized_asset
        ]
        # Keep most recent 20 after filtering
        active_rows = filtered_models[:20]
    else:
        active_rows = candidate_models[:20]
    active_models = [
        {
            "asset": row.asset,
            "version": row.version,
            "model_type": row.model_type,
            "accuracy": row.accuracy,
            "ece": row.ece,
            "quality_gate_passed": row.quality_gate_passed,
        }
        for row in active_rows
    ]

    day_ago = utc_now() - __import__("datetime").timedelta(hours=24)
    # TradeHistory filtered by asset using same normalization.
    if normalized_asset:
        recent_trades = (
            await db.execute(
                select(TradeHistory.pnl, TradeHistory.asset).where(
                    TradeHistory.timestamp >= day_ago,
                    TradeHistory.status.in_(["FILLED", "PAPER_FILLED"]),
                )
            )
        ).all()
        filtered = [
            r for r in recent_trades if _normalize_asset(r[1]) == normalized_asset
        ]
        trades_24h = len(filtered)
        net_pnl_24h = float(sum(r[0] or 0.0 for r in filtered))
        wins_24h = int(sum(1 for r in filtered if (r[0] or 0) > 0))
    else:
        trade_rows = (
            await db.execute(
                select(
                    func.count(TradeHistory.id),
                    func.coalesce(func.sum(TradeHistory.pnl), 0.0),
                    func.sum(
                        __import__("sqlalchemy").case(
                            (TradeHistory.pnl > 0, 1), else_=0
                        )
                    ),
                ).where(
                    TradeHistory.timestamp >= day_ago,
                    TradeHistory.status.in_(["FILLED", "PAPER_FILLED"]),
                )
            )
        ).first()
        trades_24h = int(trade_rows[0] or 0) if trade_rows else 0
        net_pnl_24h = float(trade_rows[1] or 0.0) if trade_rows else 0.0
        wins_24h = int(trade_rows[2] or 0) if trade_rows else 0

    prior_rows = (
        (
            await db.execute(
                select(ExperimentResult)
                .where(ExperimentResult.run_id == run_id)
                .order_by(ExperimentResult.id.desc())
                .limit(10)
            )
        )
        .scalars()
        .all()
    )
    prior_experiments = [
        {
            "result_id": row.id,
            "config_id": row.config_id,
            "evaluation_kind": row.evaluation_kind,
            "status": row.status,
            "net_pnl": row.net_pnl,
            "trade_count": row.trade_count,
            "max_drawdown": row.max_drawdown,
            "median_oot_pnl": (row.metrics or {}).get("median_pnl")
            if isinstance(row.metrics, dict)
            else None,
            "summary": row.summary,
        }
        for row in prior_rows
    ]

    from polyflip.ai_lab.llm import ALLOWED_FEATURE_SETS

    # Quality gate from scope/settings, not hardcoded.
    qg: dict[str, Any] = {}
    if isinstance(scope.get("quality_gate"), dict):
        qg = dict(scope.get("quality_gate"))
    # Allow flat keys in scope to override.
    for key in ("min_trades", "max_ece", "min_positive_oot_windows"):
        if key in scope and key not in qg:
            qg[key] = scope[key]
    # Fallback to settings if available (e.g., runtime settings), else defaults.
    try:
        from polyflip.config import settings as _cfg

        qg.setdefault(
            "min_trades", int(getattr(_cfg, "AI_LAB_QUALITY_MIN_TRADES", 30) or 30)
        )
        qg.setdefault(
            "max_ece", float(getattr(_cfg, "AI_LAB_QUALITY_MAX_ECE", 0.15) or 0.15)
        )
        qg.setdefault(
            "min_positive_oot_windows",
            int(getattr(_cfg, "AI_LAB_QUALITY_MIN_POSITIVE_WINDOWS", 2) or 2),
        )
    except Exception:
        qg.setdefault("min_trades", 30)
        qg.setdefault("max_ece", 0.15)
        qg.setdefault("min_positive_oot_windows", 2)
    quality_gate = {
        "min_trades": int(qg.get("min_trades", 30)),
        "max_ece": float(qg.get("max_ece", 0.15)),
        "min_positive_oot_windows": int(qg.get("min_positive_oot_windows", 2)),
    }

    return {
        "run": {
            "id": run.id,
            "status": run.status,
            "objective": run.objective,
            "scope": run.scope or {},
            "autonomy_level": run.autonomy_level,
            "iteration": run.experiments_completed,
            "budget_remaining_steps": max(
                run.budget_experiments - run.experiments_completed, 0
            ),
        },
        "active_models": active_models,
        "recent_trade_statistics": {
            "trades_24h": trades_24h,
            "win_rate": (wins_24h / trades_24h) if trades_24h else None,
            "net_pnl_24h": net_pnl_24h,
        },
        "prior_experiments": prior_experiments,
        "available_feature_sets": sorted(ALLOWED_FEATURE_SETS - {"DEFAULT"}),
        "quality_gate": quality_gate,
    }


@router.get("/runs/{run_id}/result")
async def get_agent_latest_result(
    run_id: int,
    lease_token: str,
    db: AsyncSession = Depends(get_db_session),
):
    await _require_run(db, run_id)
    await _verify_lease(db, run_id, lease_token)
    iteration = await _current_iteration(db, run_id)
    term = iteration["result"]
    failed = iteration["failed_result"]
    selected = term or failed
    if selected is None:
        return {"state": "PENDING", "result": None}
    state = "READY" if term is not None else "FAILED"
    return {
        "state": state,
        "result": {
            "result_id": selected.id,
            "config_id": selected.config_id,
            "evaluation_kind": selected.evaluation_kind,
            "status": selected.status,
            "metrics": selected.metrics or {},
            "net_pnl": selected.net_pnl,
            "trade_count": selected.trade_count,
            "max_drawdown": selected.max_drawdown,
            "artifact_id": selected.artifact_id,
            "summary": selected.summary,
        },
    }


class ProposalRequest(BaseModel):
    lease_token: str
    client_request_id: str = Field(min_length=8, max_length=64)
    proposal: HypothesisProposal

    @field_validator("proposal", mode="before")
    @classmethod
    def normalize_proposal(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return OpenAIResponsesProvider._coerce_payload(dict(value))
        return value


class ProposalResponse(BaseModel):
    config_id: int
    step_ids: list[int]


class AgentHeartbeatResponse(BaseModel):
    run_id: int
    leased_until: str


class AgentContextResponse(BaseModel):
    run: dict[str, Any]
    active_models: list[dict[str, Any]]
    recent_trade_statistics: dict[str, Any]
    prior_experiments: list[dict[str, Any]]
    available_feature_sets: list[str]
    quality_gate: dict[str, Any]


class AgentResultResponse(BaseModel):
    result: dict[str, Any] | None = None


class DecisionResponse(BaseModel):
    accepted: bool
    step_id: int
    action: str


class AgentCompleteResponse(BaseModel):
    run_id: int
    status: str


class AgentClaimResponse(BaseModel):
    run: dict[str, Any] | None = None


@router.post("/runs/{run_id}/proposal", response_model=ProposalResponse)
async def submit_agent_proposal(
    run_id: int,
    payload: ProposalRequest,
    db: AsyncSession = Depends(get_db_session),
):
    # Lock run row to serialize proposals for the same run.
    locked_run = (
        await db.execute(
            select(AIOptimizationRun)
            .where(AIOptimizationRun.id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if locked_run is None:
        raise HTTPException(status_code=404, detail="run not found")
    run = locked_run
    await _verify_lease(db, run_id, payload.lease_token)
    # Idempotency: same client_request_id must not create a second config.
    existing = (
        await db.execute(
            select(AIRunStep).where(
                AIRunStep.run_id == run_id,
                AIRunStep.client_request_id == payload.client_request_id,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.step_type != "PROPOSAL":
            raise HTTPException(
                status_code=409,
                detail="CLIENT_REQUEST_ID_REUSED",
            )
        cfg_id = None
        if isinstance(existing.output_payload, dict):
            cfg_id = existing.output_payload.get("config_id")
        if cfg_id is None and isinstance(existing.input_payload, dict):
            cfg_id = existing.input_payload.get("config_id")
        try:
            cfg_id_int = int(cfg_id) if cfg_id is not None else None
        except Exception:
            cfg_id_int = None
        # Collect all steps belonging to this proposal iteration.
        existing_id = int(existing.id)
        all_steps = (
            (
                await db.execute(
                    select(AIRunStep)
                    .where(AIRunStep.run_id == run_id)
                    .order_by(AIRunStep.step_index)
                )
            )
            .scalars()
            .all()
        )
        step_ids: list[int] = []
        for s in all_steps:
            if s.client_request_id == payload.client_request_id:
                step_ids.append(int(s.id))
            elif (
                cfg_id_int is not None
                and isinstance(s.input_payload, dict)
                and s.input_payload.get("config_id") == cfg_id_int
            ):
                step_ids.append(int(s.id))
        # Fallback: at least return the proposal step itself.
        if not step_ids:
            step_ids = [existing_id]
        # Do not create anything; just return previous result.
        await db.rollback()
        return {
            "config_id": int(cfg_id_int) if cfg_id_int is not None else existing_id,
            "step_ids": step_ids,
        }
    proposal = payload.proposal
    config = await create_agent_config(db, run=run, proposal=proposal)
    # Compute next_index via func.max to avoid races on step_index.
    max_index = (
        await db.execute(
            select(func.max(AIRunStep.step_index)).where(AIRunStep.run_id == run_id)
        )
    ).scalar_one_or_none()
    next_index = int(max_index) + 1 if max_index is not None else 0
    proposal_step = await append_step(
        db,
        run.id,
        step_index=next_index,
        step_type="PROPOSAL",
        status="SUCCEEDED",
        hypothesis=proposal.hypothesis[:2000],
        action="CREATE_EXPERIMENT",
        input_payload={"proposal": payload.proposal.model_dump(mode="json")},
        output_payload={"config_id": config.id},
        summary=f"External agent hypothesis for {proposal.asset}",
    )
    # Persist client_request_id on the proposal step (append_step does not know it).
    proposal_step.client_request_id = payload.client_request_id  # type: ignore[attr-defined]
    await db.flush()
    # Append canonical TRAIN/OOT steps after the proposal.
    steps = await plan_run(db, run.id, [config.id])
    await db.commit()
    return {
        "config_id": config.id,
        "step_ids": [proposal_step.id] + [step.id for step in steps],
    }


async def create_agent_config(
    db: AsyncSession, *, run: AIOptimizationRun, proposal: HypothesisProposal
) -> AIExperimentConfig:
    name = f"agent-{run.id}-{uuid.uuid4().hex[:8]}"
    from polyflip.ai_lab.service import create_experiment_config

    return await create_experiment_config(
        db,
        name=name,
        model_family=proposal.model_family,
        feature_set=proposal.feature_set,
        feature_pipeline_version="agent-v1",
        model_params=dict(proposal.parameter_changes or {}),
        strategy_params=dict(proposal.strategy_parameter_changes or {}),
        backtest_params=dict(proposal.test_plan or {}),
        asset=(proposal.asset or "").strip() or None,
        description=proposal.hypothesis[:500],
        created_by="external-agent",
    )


class DecisionRequest(BaseModel):
    lease_token: str
    client_request_id: str = Field(min_length=8, max_length=64)
    decision: AgentDecision

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return OpenAIResponsesProvider._coerce_payload(dict(value))
        return value


@router.post("/runs/{run_id}/decision", response_model=DecisionResponse)
async def submit_agent_decision(
    run_id: int,
    payload: DecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    # Lock run row for decision serialization.
    locked_run = (
        await db.execute(
            select(AIOptimizationRun)
            .where(AIOptimizationRun.id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if locked_run is None:
        raise HTTPException(status_code=404, detail="run not found")
    run = locked_run
    await _verify_lease(db, run_id, payload.lease_token)
    iteration = await _current_iteration(db, run_id)
    term = iteration["result"]
    if term is None:
        raise HTTPException(status_code=409, detail="RESULT_NOT_READY")

    existing_request = (
        await db.execute(
            select(AIRunStep).where(
                AIRunStep.run_id == run_id,
                AIRunStep.client_request_id == payload.client_request_id,
            )
        )
    ).scalar_one_or_none()
    if existing_request is not None:
        existing_result_id = (existing_request.output_payload or {}).get("result_id")
        if existing_request.step_type != "DECISION" or existing_result_id != term.id:
            raise HTTPException(
                status_code=409,
                detail="CLIENT_REQUEST_ID_REUSED",
            )
        return {
            "accepted": True,
            "step_id": int(existing_request.id),
            "action": str(existing_request.action or payload.decision.action),
        }

    existing_for_result = iteration["current_decision"]
    if existing_for_result is not None:
        step_id = int(existing_for_result.id)
        action = str(existing_for_result.action or payload.decision.action)
        await db.rollback()
        return {
            "accepted": True,
            "step_id": step_id,
            "action": action,
        }
    # Verify result belongs to run (already filtered) and recommended_config_id belongs.
    decision = payload.decision
    rec_id = decision.recommended_config_id
    if rec_id is not None:
        cfg = await db.get(AIExperimentConfig, int(rec_id))
        if cfg is None:
            raise HTTPException(status_code=422, detail="unknown recommended_config_id")
        # Verify the config was proposed for this run (via steps or results).
        has_result = (
            await db.execute(
                select(ExperimentResult).where(
                    ExperimentResult.run_id == run_id,
                    ExperimentResult.config_id == int(rec_id),
                )
            )
        ).scalar_one_or_none()
        belongs_via_step = False
        if has_result is None:
            all_steps = (
                (await db.execute(select(AIRunStep).where(AIRunStep.run_id == run_id)))
                .scalars()
                .all()
            )
            for s in all_steps:
                if isinstance(s.input_payload, dict) and s.input_payload.get(
                    "config_id"
                ) == int(rec_id):
                    belongs_via_step = True
                    break
                if isinstance(s.output_payload, dict) and s.output_payload.get(
                    "config_id"
                ) == int(rec_id):
                    belongs_via_step = True
                    break
            if not belongs_via_step:
                raise HTTPException(
                    status_code=422,
                    detail="recommended_config_id does not belong to run",
                )
        # Also ensure the recommended config matches the terminal result when possible.
        # We allow any belonging config, but if term exists, we at least ensure term belongs.
        if term.run_id != run_id:
            raise HTTPException(status_code=422, detail="result does not belong to run")
    # Compute next index via func.max.
    max_index = (
        await db.execute(
            select(func.max(AIRunStep.step_index)).where(AIRunStep.run_id == run_id)
        )
    ).scalar_one_or_none()
    next_index = int(max_index) + 1 if max_index is not None else 0
    step = await append_step(
        db,
        run_id,
        step_index=next_index,
        step_type="DECISION",
        status="SUCCEEDED",
        action=decision.action,
        input_payload={
            "decision": payload.decision.model_dump(mode="json"),
            "result_id": term.id,
        },
        output_payload={
            "recommended_config_id": rec_id,
            "result_id": term.id,
            "config_id": rec_id,
        },
        summary=decision.rationale[:400],
    )
    step.client_request_id = payload.client_request_id  # type: ignore[attr-defined]
    await db.flush()
    # Increment experiments_completed exactly once per successful decision.
    run.experiments_completed = (run.experiments_completed or 0) + 1
    # Transition to EVALUATING if still RUNNING.
    if run.status == "RUNNING":
        try:
            await transition_run(db, run, "EVALUATING", reason="agent decision")
        except Exception:
            pass
    await db.commit()
    return {
        "accepted": True,
        "step_id": step.id,
        "action": decision.action,
    }


class CompleteRequest(BaseModel):
    action: Literal["COMPLETED", "FAILED", "REQUEUE"]
    reason: str = Field(default="", max_length=2000)
    lease_token: str


@router.post("/runs/{run_id}/complete")
async def complete_agent_run(
    run_id: int,
    payload: CompleteRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = (
        await db.execute(
            select(AIOptimizationRun)
            .where(AIOptimizationRun.id == run_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    await _verify_lease(db, run_id, payload.lease_token)
    if payload.action == "COMPLETED":
        if run.status != "EVALUATING":
            await transition_run(db, run, "EVALUATING", reason="agent finalizing")
        await transition_run(
            db,
            run,
            "COMPLETED",
            reason=payload.reason or "agent complete",
        )
    elif payload.action == "FAILED":
        await transition_run(
            db,
            run,
            "FAILED",
            reason=payload.reason or "agent failed",
        )
    else:
        iteration = await _current_iteration(db, run_id)
        decision = iteration["current_decision"] or iteration["latest_decision"]
        max_index = (
            await db.execute(
                select(func.max(AIRunStep.step_index)).where(AIRunStep.run_id == run_id)
            )
        ).scalar_one_or_none()
        await append_step(
            db,
            run_id,
            step_index=int(max_index) + 1 if max_index is not None else 0,
            step_type="ITERATION_REQUEUED",
            status="SUCCEEDED",
            action="REQUEUE",
            input_payload={
                "decision_step_id": decision.id if decision is not None else None,
                "decision_action": (decision.action if decision is not None else None),
            },
            output_payload={"next_iteration": run.experiments_completed},
            summary=payload.reason or "agent requeued",
        )
        await transition_run(db, run, "QUEUED", reason=payload.reason or "requeue")
    await _release_lease(db, run_id)
    await db.commit()
    return {"run_id": run_id, "status": run.status}
