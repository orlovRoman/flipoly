"""Typed payloads exchanged with the AI Lab agent API.

These are lightweight client-side models for typing and sanity only. The
authoritative validation of proposals/decisions happens inside the API using
the canonical ``HypothesisProposal`` / ``AgentDecision`` schemas.
"""
from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class ClaimedRun(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    status: str
    objective: str = ""
    scope: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: str = "EXPERIMENT"
    budget_experiments: int = 0
    budget_seconds: int = Field(default=0, alias="budget_seconds")
    experiments_completed: int = 0
    lease_token: str | None = None
    llm_provider: str | None = None
    llm_research_model: str | None = None
    llm_summary_model: str | None = None


class AgentContext(BaseModel):
    run: dict[str, Any] = Field(default_factory=dict)
    active_models: list[dict[str, Any]] = Field(default_factory=list)
    recent_trade_statistics: dict[str, Any] = Field(default_factory=dict)
    prior_experiments: list[dict[str, Any]] = Field(default_factory=list)
    available_feature_sets: list[str] = Field(default_factory=list)
    quality_gate: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    config_id: int | None = None
    evaluation_kind: str | None = None
    status: str = "UNKNOWN"
    metrics: dict[str, Any] = Field(default_factory=dict)
    net_pnl: float | None = None
    trade_count: int | None = None
    max_drawdown: float | None = None
    summary: str | None = None


CompleteAction: TypeAlias = Literal["COMPLETED", "FAILED", "REQUEUE"]
