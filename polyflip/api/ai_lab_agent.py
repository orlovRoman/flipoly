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
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.ai_lab.llm import (
    AgentDecision,
    HypothesisProposal,
    OpenAIResponsesProvider,
)
from polyflip.ai_lab.orchestrator import plan_run
from polyflip.ai_lab.service import append_step, transition_run, utc_now
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
    expected = (
        getattr(settings, "AI_LAB_AGENT_TOKEN", "")
        or getattr(settings, "API_KEY", "")
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


async def _acquire_lease(
    db: AsyncSession, run_id: int, worker_id: str
) -> str | None:
    now = utc_now()
    expires_at = now + timedelta(seconds=_lease_ttl_seconds())
    lease = (
        await db.execute(
            select(AIWorkerLease).where(AIWorkerLease.run_id == run_id)
        )
    ).scalar_one_or_none()
    token = f"agent-{uuid.uuid4().hex}"
    if lease is None:
        db.add(AIWorkerLease(
            run_id=run_id,
            worker_id=worker_id,
            owner_token=token,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        ))
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


async def _verify_lease(
    db: AsyncSession, run_id: int, lease_token: str | None
) -> None:
    if not lease_token:
        raise HTTPException(status_code=409, detail="LEASE_LOST")
    now = utc_now()
    lease = (
        await db.execute(
            select(AIWorkerLease).where(AIWorkerLease.run_id == run_id)
        )
    ).scalar_one_or_none()
    if (
        lease is None
        or lease.owner_token != lease_token
        or _as_utc(lease.expires_at) <= now
    ):
        raise HTTPException(status_code=409, detail="LEASE_LOST")


async def _renew_lease(db: AsyncSession, run_id: int, lease_token: str | None) -> None:
    await _verify_lease(db, run_id, lease_token)
    now = utc_now()
    lease = (
        await db.execute(
            select(AIWorkerLease).where(AIWorkerLease.run_id == run_id)
        )
    ).scalar_one()
    lease.heartbeat_at = now
    lease.expires_at = now + timedelta(seconds=_lease_ttl_seconds())
    await db.flush()


async def _release_lease(db: AsyncSession, run_id: int) -> None:
    lease = (
        await db.execute(
            select(AIWorkerLease).where(AIWorkerLease.run_id == run_id)
        )
    ).scalar_one_or_none()
    if lease is not None:
        await db.delete(lease)


async def _require_run(db: AsyncSession, run_id: int) -> AIOptimizationRun:
    run = await db.get(AIOptimizationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


async def _agent_phase(db, run_id: int) -> dict[str, Any]:
    from polyflip.db.models import AIRunStep
    steps = (await db.execute(select(AIRunStep).where(AIRunStep.run_id == run_id).order_by(AIRunStep.step_index))).scalars().all()
    has_proposal = any(s.step_type == "PROPOSAL" for s in steps)
    pending = any(s.status == "PENDING" and s.step_type in {"TRAIN_MODEL", "RUN_OOT_BACKTEST", "RUN_POLYMARKET_OOT"} for s in steps)
    has_decision = any(s.step_type == "DECISION" for s in steps)
    # terminal OOT result
    term = (await db.execute(select(ExperimentResult).where(ExperimentResult.run_id == run_id, ExperimentResult.evaluation_kind == "POLYMARKET_OOT", ExperimentResult.status == "SUCCEEDED").order_by(ExperimentResult.id.desc()).limit(1))).scalar_one_or_none()
    if not has_proposal:
        phase = "NEEDS_PROPOSAL"
    elif pending:
        phase = "WAITING_RESULT"
    elif term is not None and not has_decision:
        phase = "NEEDS_DECISION"
    elif has_decision:
        phase = "NEEDS_COMPLETION"
    else:
        phase = "NEEDS_PROPOSAL"
    latest_cfg = next((s.input_payload.get("config_id") for s in reversed(steps) if s.input_payload and s.input_payload.get("config_id")), None)
    return {"phase": phase, "latest_config_id": latest_cfg, "latest_result_id": term.id if term else None, "latest_decision": has_decision}


def _claimed_run_payload(run: AIOptimizationRun, lease_token: str) -> dict[str, Any]:
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
    # Idempotent: active lease held by this worker is returned as-is.
    now = utc_now()
    existing_lease = (
        await db.execute(
            select(AIWorkerLease).where(
                AIWorkerLease.worker_id == worker_id,
                AIWorkerLease.expires_at > now,
            )
        )
    ).scalar_one_or_none()
    if existing_lease is not None:
        run = await db.get(AIOptimizationRun, existing_lease.run_id)
        if run is not None and run.status in {
            "QUEUED",
            "PLANNING",
            "RUNNING",
            "EVALUATING",
        }:
            return {"run": _claimed_run_payload(run, existing_lease.owner_token)}
    candidate_ids = (
        await db.execute(
            select(AIOptimizationRun.id)
            .where(AIOptimizationRun.status == "QUEUED")
            .order_by(AIOptimizationRun.id)
            .limit(10)
        )
    ).scalars().all()
    for run_id in candidate_ids:
        token = await _acquire_lease(db, run_id, worker_id)
        if token is None:
            continue
        result = await db.execute(
            update(AIOptimizationRun)
            .where(
                AIOptimizationRun.id == run_id,
                AIOptimizationRun.status == "QUEUED",
            )
            .values(status="RUNNING", started_at=utc_now(), agent_type="EXTERNAL_AGENT")
        )
        if result.rowcount != 1:
            await db.rollback()
            continue
        run = await _require_run(db, run_id)
        await db.commit()
        logger.info("agent_run_claimed", run_id=run_id, worker=worker_id)
        return {"run": _claimed_run_payload(run, token)}
    await db.rollback()
    return {"run": None}


class HeartbeatRequest(BaseModel):
    run_id: int
    lease_token: str


@router.post("/heartbeat")
async def agent_heartbeat(payload: HeartbeatRequest, db: AsyncSession = Depends(get_db_session)):
    await _require_run(db, payload.run_id)
    try:
        await _renew_lease(db, payload.run_id, payload.lease_token)
        await db.commit()
    except HTTPException as exc:
        await db.rollback()
        raise exc
    return {
        "run_id": payload.run_id,
        "leased_until": (
            utc_now().isoformat().replace("+00:00", "Z")
        ),
    }


@router.get("/runs/{run_id}/context")
async def get_agent_context(
    run_id: int,
    lease_token: str,
    db: AsyncSession = Depends(get_db_session),
):
    run = await _require_run(db, run_id)
    await _verify_lease(db, run_id, lease_token)

    active_rows = (
        await db.execute(
            select(ModelRegistry)
            .where(ModelRegistry.is_active.is_(True))
            .order_by(ModelRegistry.id.desc())
            .limit(20)
        )
    ).scalars().all()
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
    trades_24h = int(trade_rows[0] or 0)
    net_pnl_24h = float(trade_rows[1] or 0.0)
    wins_24h = int(trade_rows[2] or 0)

    prior_rows = (
        await db.execute(
            select(ExperimentResult)
            .where(ExperimentResult.run_id == run_id)
            .order_by(ExperimentResult.id.desc())
            .limit(10)
        )
    ).scalars().all()
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
        "quality_gate": {
            "min_trades": 30,
            "max_ece": 0.15,
            "min_positive_oot_windows": 2,
        },
    }


@router.get("/runs/{run_id}/result")
async def get_agent_latest_result(
    run_id: int,
    lease_token: str,
    db: AsyncSession = Depends(get_db_session),
):
    await _require_run(db, run_id)
    await _verify_lease(db, run_id, lease_token)
    row = (
        await db.execute(
            select(ExperimentResult)
            .where(ExperimentResult.run_id == run_id)
            .order_by(ExperimentResult.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        return {"result": None}
    return {
        "result": {
            "result_id": row.id,
            "config_id": row.config_id,
            "evaluation_kind": row.evaluation_kind,
            "status": row.status,
            "metrics": row.metrics or {},
            "net_pnl": row.net_pnl,
            "trade_count": row.trade_count,
            "max_drawdown": row.max_drawdown,
            "artifact_id": row.artifact_id,
            "summary": row.summary,
        }
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
        all_steps = (
            await db.execute(
                select(AIRunStep)
                .where(AIRunStep.run_id == run_id)
                .order_by(AIRunStep.step_index)
            )
        ).scalars().all()
        step_ids: list[int] = []
        for s in all_steps:
            if s.client_request_id == payload.client_request_id:
                step_ids.append(int(s.id))
            elif cfg_id_int is not None and isinstance(s.input_payload, dict) and s.input_payload.get("config_id") == cfg_id_int:
                step_ids.append(int(s.id))
        # Fallback: at least return the proposal step itself.
        if not step_ids:
            step_ids = [int(existing.id)]
        # Do not create anything; just return previous result.
        await db.rollback()
        return {"config_id": int(cfg_id_int) if cfg_id_int is not None else int(existing.id), "step_ids": step_ids}
    proposal = payload.proposal
    config = await create_agent_config(db, run=run, proposal=proposal)
    # Compute next_index via func.max to avoid races on step_index.
    max_index = (
        await db.execute(select(func.max(AIRunStep.step_index)).where(AIRunStep.run_id == run_id))
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
    decision: AgentDecision

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return OpenAIResponsesProvider._coerce_payload(dict(value))
        return value


@router.post(
    "/runs/{run_id}/decision", response_model=DecisionResponse
)
async def submit_agent_decision(
    run_id: int,
    payload: DecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = await _require_run(db, run_id)
    await _verify_lease(db, run_id, payload.lease_token)
    decision = payload.decision
    latest_step = (
        await db.execute(
            select(AIRunStep.step_index)
            .where(AIRunStep.run_id == run_id)
            .order_by(AIRunStep.step_index.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    step = await append_step(
        db,
        run_id,
        step_index=(latest_step + 1) if latest_step is not None else 0,
        step_type="DECISION",
        status="SUCCEEDED",
        action=decision.action,
        input_payload={"decision": payload.decision.model_dump(mode="json")},
        output_payload={"recommended_config_id": decision.recommended_config_id},
        summary=decision.rationale[:400],
    )
    run.experiments_completed = (run.experiments_completed or 0) + 1
    await db.commit()
    return {
        "accepted": True,
        "step_id": step.id,
        "action": decision.action,
    }


class CompleteRequest(BaseModel):
    action: Literal["COMPLETED", "FAILED", "REQUEUE"]
    reason: str = Field(default="", max_length=2000)
    lease_token: str | None = None


@router.post("/runs/{run_id}/complete")
async def complete_agent_run(
    run_id: int,
    payload: CompleteRequest,
    db: AsyncSession = Depends(get_db_session),
):
    run = await _require_run(db, run_id)
    if payload.action != "REQUEUE":
        # Terminal actions require the live lease; REQUEUE may also come from
        # an agent whose lease just expired to avoid zombie RUNNING runs.
        await _verify_lease(db, run_id, payload.lease_token)
    elif payload.lease_token:
        lease = (
            await db.execute(
                select(AIWorkerLease).where(AIWorkerLease.run_id == run_id)
            )
        ).scalar_one_or_none()
        if lease is not None and lease.owner_token == payload.lease_token:
            await _release_lease(db, run_id)
    if payload.action == "COMPLETED":
        # Canonical state machine: RUNNING -> EVALUATING -> COMPLETED.
        await transition_run(db, run, "EVALUATING", reason="agent finalizing")
        await transition_run(db, run, "COMPLETED", reason=payload.reason or "agent complete")
    elif payload.action == "FAILED":
        await transition_run(db, run, "FAILED", reason=payload.reason or "agent failed")
    else:
        run.status = "QUEUED"
        run.started_at = None
        run.error = None
    await _release_lease(db, run_id)
    await db.commit()
    return {"run_id": run_id, "status": run.status}
