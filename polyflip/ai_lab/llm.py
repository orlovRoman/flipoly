"""LLM Provider abstraction and structured hypothesis schemas for AI Lab Phase 10.

Integrates with OpenAI Responses API (store=False) and provides deterministic MockLLMProvider.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field, field_validator
import structlog

logger = structlog.get_logger("polyflip.ai_lab.llm")

ALLOWED_MODEL_FAMILIES = {"LOGREG", "LIGHTGBM", "LogisticRegression", "LightGBM"}
ALLOWED_FEATURE_SETS = {"FS_D0", "FS_D1", "FS_D2", "FS_D3", "FS_D4", "FS_D5", "DEFAULT"}
ALLOWED_MARKET_ROLES = {"FAVORITE", "OUTSIDER", "COMBINED", "DIRECTION_ONLY", "ALL"}
ALLOWED_ASSETS = {
    "BTC", "ETH", "SOL", "XRP", "DOGE",
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
}


# ---------------------------------------------------------------------------
LLM_PROVIDERS = ("mock", "openai", "opencode", "openrouter")
DEFAULT_OPENCODE_ENDPOINT = "https://opencode.ai/zen/v1/responses"
DEFAULT_OPENCODE_CHAT_ENDPOINT = "https://opencode.ai/zen/v1/chat/completions"
DEFAULT_OPENCODE_GO_RESPONSES_ENDPOINT = "https://opencode.ai/zen/go/v1/responses"
DEFAULT_OPENCODE_GO_CHAT_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
DEFAULT_OPENCODE_GO_MESSAGES_ENDPOINT = "https://opencode.ai/zen/go/v1/messages"
DEFAULT_OPENCODE_GO_MODEL_SPECS = (
    ("grok-4.6", "Grok 4.6", "responses"),
    ("glm-5.3-flash", "GLM-5.3-Flash", "chat_completions"),
    ("glm-5.3", "GLM-5.3", "chat_completions"),
    ("glm-5.2", "GLM-5.2", "chat_completions"),
    ("glm-5.1", "GLM-5.1", "chat_completions"),
    ("gpt-5.6-luna", "GPT 5.6 Luna", "responses"),
    ("kimi-k3", "Kimi K3", "chat_completions"),
    ("kimi-k2.7-code", "Kimi K2.7 Code", "chat_completions"),
    ("kimi-k2.6", "Kimi K2.6", "chat_completions"),
    ("longcat-2.0", "LongCat-2.0", "chat_completions"),
    ("mimo-v2.5", "MiMo-V2.5", "chat_completions"),
    ("mimo-v2.5-pro", "MiMo-V2.5-Pro", "chat_completions"),
    ("minimax-m3", "MiniMax M3", "messages"),
    ("minimax-m2.7", "MiniMax M2.7", "messages"),
    ("muse-spark-1.2-contributor", "Muse Spark 1.2 Contributor", "responses"),
    ("qwen3.8-max", "Qwen3.8 Max", "messages"),
    ("qwen3.8-flash", "Qwen3.8 Flash", "messages"),
    ("qwen3.7-max", "Qwen3.7 Max", "messages"),
    ("qwen3.7-plus", "Qwen3.7 Plus", "messages"),
    ("qwen3.6-plus", "Qwen3.6 Plus", "messages"),
    ("deepseek-v4-pro", "DeepSeek V4 Pro", "chat_completions"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash", "chat_completions"),
    ("deepseek-v4-flash-vision-exp", "DeepSeek V4 Flash Vision Exp", "chat_completions"),
    ("hy3", "Hy3", "chat_completions"),
)
DEFAULT_OPENCODE_FREE_MODEL_SPECS = (
    ("big-pickle", "Big Pickle", "chat_completions"),
    ("x-preview-f-free", "Ox Alpha Free", "chat_completions"),
    ("mimo-v2.5-free", "MiMo-V2.5 Free", "chat_completions"),
    ("hy3-free", "Hy3 Free", "chat_completions"),
    ("nemotron-3-ultra-free", "Nemotron 3 Ultra Free", "chat_completions"),
    ("nemotron-3.5-lightning-free", "Nemotron 3.5 Lightning Free", "chat_completions"),
    ("muse-spark-1.2-contributor-free", "Muse Spark 1.2 Contributor Free", "responses"),
)
OPENCODE_MODEL_SPECS = {
    model_id: {
        "label": label,
        "protocol": protocol,
        "is_go": model_id in {item[0] for item in DEFAULT_OPENCODE_GO_MODEL_SPECS},
        "supports_structured_output": True,
    }
    for model_id, label, protocol in (
        DEFAULT_OPENCODE_GO_MODEL_SPECS + DEFAULT_OPENCODE_FREE_MODEL_SPECS
    )
}
OPENCODE_GO_MODELS = frozenset(
    item[0] for item in DEFAULT_OPENCODE_GO_MODEL_SPECS
)
OPENCODE_FREE_MODELS = frozenset(
    item[0] for item in DEFAULT_OPENCODE_FREE_MODEL_SPECS
)
DEFAULT_OPENCODE_RESPONSES_MODELS = tuple(
    model_id
    for model_id, _label, protocol in (
        DEFAULT_OPENCODE_GO_MODEL_SPECS + DEFAULT_OPENCODE_FREE_MODEL_SPECS
    )
    if protocol == "responses"
)
DEFAULT_OPENCODE_CHAT_MODELS = tuple(
    model_id
    for model_id, _label, protocol in (
        DEFAULT_OPENCODE_GO_MODEL_SPECS + DEFAULT_OPENCODE_FREE_MODEL_SPECS
    )
    if protocol == "chat_completions"
)
DEFAULT_OPENCODE_MESSAGES_MODELS = tuple(
    model_id
    for model_id, _label, protocol in DEFAULT_OPENCODE_GO_MODEL_SPECS
    if protocol == "messages"
)
DEFAULT_OPENCODE_MODELS = tuple(
    item[0] for item in DEFAULT_OPENCODE_GO_MODEL_SPECS + DEFAULT_OPENCODE_FREE_MODEL_SPECS
)
DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
# OpenRouter's GO catalogue (canonical model slugs, not display names).
# Models without response_format support remain visible but are rejected when
# selected for a structured AI Lab run.
OPENROUTER_GO_MODEL_CATALOG = (
    ("x-ai/grok-4.6", "Grok 4.6", True),
    ("openai/gpt-5.6-luna", "GPT 5.6 Luna", True),
    ("z-ai/glm-5.3-flash", "GLM-5.3-Flash", True),
    ("z-ai/glm-5.3", "GLM-5.3", True),
    ("z-ai/glm-5.2", "GLM-5.2", True),
    ("z-ai/glm-5.1", "GLM-5.1", True),
    ("moonshotai/kimi-k3", "Kimi K3", True),
    ("moonshotai/kimi-k2.7-code", "Kimi K2.7 Code", True),
    ("moonshotai/kimi-k2.6", "Kimi K2.6", True),
    ("meituan/longcat-2.0", "LongCat-2.0", False),
    ("xiaomi/mimo-v2.5", "MiMo-V2.5", True),
    ("xiaomi/mimo-v2.5-pro", "MiMo-V2.5-Pro", True),
    ("minimax/minimax-m3", "MiniMax M3", True),
    ("minimax/minimax-m2.7", "MiniMax M2.7", True),
    ("meta/muse-spark-1.2-contributor", "Muse Spark 1.2 Contributor", True),
    ("qwen/qwen3.8-max", "Qwen3.8 Max", True),
    ("qwen/qwen3.7-max", "Qwen3.7 Max", True),
    ("qwen/qwen3.7-plus", "Qwen3.7 Plus", True),
    ("qwen/qwen3.6-plus", "Qwen3.6 Plus", True),
    ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", True),
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", True),
    ("deepseek/deepseek-v4-flash-vision-exp", "DeepSeek V4 Flash Vision Exp", True),
    ("tencent/hy3", "Hy3", True),
)
DEFAULT_OPENROUTER_MODELS = tuple(item[0] for item in OPENROUTER_GO_MODEL_CATALOG)
OPENROUTER_MODEL_LABELS = {item[0]: item[1] for item in OPENROUTER_GO_MODEL_CATALOG}
OPENCODE_MODEL_LABELS = {
    model_id: label
    for model_id, label, _protocol in (
        DEFAULT_OPENCODE_GO_MODEL_SPECS + DEFAULT_OPENCODE_FREE_MODEL_SPECS
    )
}


def _csv_values(value: Any) -> list[str]:
    """Return a normalized comma-separated list without exposing secrets."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _settings_value(settings_obj: Any, name: str, default: Any = "") -> Any:
    return getattr(settings_obj, name, default)


def llm_provider_configured(provider_name: str, settings_obj: Any | None = None) -> bool:
    """Report whether a provider has credentials configured, never returning them."""
    if settings_obj is None:
        from polyflip.config import settings as settings_obj
    provider = str(provider_name or "").strip().lower()
    if provider == "mock":
        return True
    key_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "AI_LAB_LLM_API_KEY"
    fallback_key = (
        _settings_value(settings_obj, "AI_LAB_LLM_API_KEY", "")
        or _settings_value(settings_obj, "OPENAI_API_KEY", "")
    )
    return bool(str(_settings_value(settings_obj, key_name, fallback_key) or "").strip())


def get_llm_model_catalog(provider_name: str | None = None) -> dict[str, Any]:
    """Return the safe provider/model catalog used by the AI Lab UI."""
    from polyflip.config import settings

    selected = str(provider_name or "").strip().lower() or None
    configured_models = _csv_values(_settings_value(settings, "AI_LAB_ALLOWED_MODELS", ""))
    research_default = str(_settings_value(settings, "AI_LAB_MODEL_RESEARCH", "gpt-5.6"))
    summary_default = str(_settings_value(settings, "AI_LAB_MODEL_SUMMARY", "gpt-5.6-mini"))
    available = [p.lower() for p in _csv_values(
        _settings_value(settings, "AI_LAB_LLM_AVAILABLE_PROVIDERS", ",".join(LLM_PROVIDERS))
    )]
    available = [p for p in available if p in LLM_PROVIDERS] or ["mock"]

    def models_for(provider: str) -> list[str]:
        if provider == "mock":
            return ["mock-gpt-5"]
        if provider == "opencode":
            models = list(DEFAULT_OPENCODE_MODELS)
            if configured_models:
                # The OpenCode menu is intentionally limited to the documented
                # Go catalogue and the explicitly free Zen models.  Ignore
                # stale/unknown values from AI_LAB_ALLOWED_MODELS instead of
                # re-introducing arbitrary provider models.
                matching = [m for m in configured_models if m in models]
                if matching:
                    models = matching
            return models
        if provider == "openrouter":
            models = list(DEFAULT_OPENROUTER_MODELS)
            provider_models = _csv_values(
                _settings_value(settings, "AI_LAB_OPENROUTER_MODELS", "")
            )
            if provider_models:
                models = [m for m in provider_models if m in models] or provider_models
            elif configured_models:
                matching = [m for m in configured_models if m in models]
                if matching:
                    models = matching
            return models
        models = [research_default, summary_default]
        if configured_models:
            models = [m for m in configured_models if m] or models
        return list(dict.fromkeys(models))

    providers = [{
        "id": provider,
        "label": {"mock": "Mock (offline)", "openai": "OpenAI", "opencode": "OpenCode", "openrouter": "OpenRouter"}[provider],
        "configured": llm_provider_configured(provider, settings),
    } for provider in available]
    if selected and selected not in available:
        raise ValueError(f"Unsupported or disabled AI Lab provider: {selected}")
    target = selected or str(_settings_value(settings, "AI_LAB_LLM_PROVIDER", "mock")).lower()
    if target not in available:
        target = available[0]
    models = models_for(target)
    effective_research = (
        models[0] if target in {"mock", "opencode", "openrouter"} else research_default
    )
    effective_summary = (
        models[-1] if target in {"mock", "opencode", "openrouter"} else summary_default
    )
    model_specs = {
        model_id: {
            "label": label,
            "protocol": "chat_completions",
            "supports_structured_output": supports,
        }
        for model_id, label, supports in OPENROUTER_GO_MODEL_CATALOG
    }
    return {
        "provider": target,
        "providers": providers,
        "models": [{
            "id": model,
            "label": (
                OPENCODE_MODEL_LABELS.get(model, model)
                if target == "opencode"
                else model_specs.get(model, {}).get("label", model)
                if target == "openrouter"
                else model
            ),
            "protocol": (
                model_specs.get(model, {}).get("protocol")
                if target == "openrouter"
                else (
                    OPENCODE_MODEL_SPECS.get(model, {}).get("protocol", "responses")
                    if target == "opencode"
                    else "responses"
                )
            ),
            "supports_structured_output": (
                bool(model_specs.get(model, {}).get("supports_structured_output", True))
                if target == "openrouter"
                else bool(OPENCODE_MODEL_SPECS.get(model, {}).get("supports_structured_output", True))
                if target == "opencode"
                else True
            ),
            "default_research": model == effective_research,
            "default_summary": model == effective_summary,
        } for model in models],
        "defaults": {
            "research_model": effective_research,
            "summary_model": effective_summary,
        },
    }


def normalize_llm_selection(
    provider_name: str | None,
    model_research: str | None,
    model_summary: str | None,
) -> tuple[str, str, str]:
    """Validate and fill a run's immutable provider/model selection."""
    catalog = get_llm_model_catalog(provider_name)
    provider = catalog["provider"]
    allowed = {str(item["id"]) for item in catalog["models"]}
    defaults = catalog["defaults"]
    research = str(model_research or defaults["research_model"])
    summary = str(model_summary or defaults["summary_model"])
    if research not in allowed or summary not in allowed:
        raise ValueError(
            f"Unknown model for provider {provider}: research={research!r}, summary={summary!r}"
        )
    return provider, research, summary

# Structured Output Schemas
# ---------------------------------------------------------------------------
class HypothesisProposal(BaseModel):
    """Structured hypothesis and experiment plan generated by the LLM."""

    hypothesis: str = Field(..., min_length=10, max_length=2000, description="Clear statement of market hypothesis")
    asset: str = Field(..., description="Target trading asset (e.g. BTC, ETH, SOL)")
    market_role: str = Field(default="ALL", description="Target market role: FAVORITE, OUTSIDER, COMBINED, ALL")
    model_family: str = Field(..., description="ML algorithm: LOGREG or LIGHTGBM")
    feature_set: str = Field(..., description="Feature pipeline identifier, e.g. FS_D0, FS_D1, FS_D2")
    parameter_changes: dict[str, Any] = Field(default_factory=dict, description="Hyperparameters for model training")
    strategy_parameter_changes: dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters, e.g. decision_threshold, min_edge"
    )
    expected_effect: dict[str, Any] = Field(
        default_factory=lambda: {"metric": "median_oot_pnl", "direction": "increase"},
        description="Expected quantitative improvement",
    )
    reasoning: list[str] = Field(default_factory=list, description="List of logical arguments supporting the hypothesis")
    risks: list[str] = Field(default_factory=list, description="Identified risks or selection bias concerns")
    test_plan: dict[str, Any] = Field(
        default_factory=lambda: {"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"},
        description="Evaluation protocol parameters",
    )

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        clean = v.upper().replace("USDT", "")
        if clean not in {"BTC", "ETH", "SOL", "XRP", "DOGE"}:
            raise ValueError(
                f"Unsupported asset: '{v}'. Must be one of BTC, ETH, SOL, XRP, DOGE."
            )
        return clean + "USDT"

    @field_validator("model_family")
    @classmethod
    def validate_family(cls, v: str) -> str:
        if v not in ALLOWED_MODEL_FAMILIES:
            raise ValueError(f"Disallowed model family '{v}'. Must be in {ALLOWED_MODEL_FAMILIES}")
        if v.upper().startswith("LOG"):
            return "LogisticRegression"
        return "LightGBM"

    @field_validator("feature_set")
    @classmethod
    def validate_feature_set(cls, v: str) -> str:
        v_upper = v.upper()
        if not (v_upper.startswith("FS_") or v_upper in ALLOWED_FEATURE_SETS or v.isalnum() or "_" in v):
            raise ValueError(f"Invalid feature set name '{v}'.")
        return v_upper

    @field_validator("market_role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in ALLOWED_MARKET_ROLES:
            raise ValueError(f"Invalid market role '{v}'. Must be in {ALLOWED_MARKET_ROLES}")
        return v_upper


class AgentDecision(BaseModel):
    """Structured decision after analyzing experiment results."""

    action: str = Field(
        ...,
        description="Action to take: CONTINUE_RESEARCH, MUTATE_HYPOTHESIS, RECOMMEND_SHADOW, FINALIZE_NO_WINNER, APPLY_OVERLAY, REQUEST_LIVE_APPROVAL",
    )
    rationale: str = Field(..., min_length=10, description="Detailed explanation of the analytical reasoning")
    key_findings: list[str] = Field(default_factory=list, description="Key empirical observations from OOT metrics")
    recommended_config_id: int | None = Field(default=None, description="Winning experiment config id if eligible")
    proposed_overlay: dict[str, Any] | None = Field(
        default=None, description="Proposed temporary runtime settings overlay"
    )
    next_step_focus: str | None = Field(default=None, description="Focus area for the next iteration if continuing")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {
            "CONTINUE_RESEARCH",
            "MUTATE_HYPOTHESIS",
            "RECOMMEND_SHADOW",
            "FINALIZE_NO_WINNER",
            "APPLY_OVERLAY",
            "REQUEST_LIVE_APPROVAL",
            "STOP_BUDGET_EXHAUSTED",
        }
        if v.upper() not in allowed:
            raise ValueError(f"Invalid agent action '{v}'. Must be one of {sorted(allowed)}")
        return v.upper()


class LLMUsageStats(BaseModel):
    """Token usage and cost tracking for LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0
    provider: str = "openai"
    model: str = "gpt-5.6"
    prompt_hash: str = ""
    response_hash: str = ""


# ---------------------------------------------------------------------------
# Context Dataclasses
# ---------------------------------------------------------------------------
class AgentContext(BaseModel):
    """Snapshot context provided to the LLM for hypothesis formation."""

    run_id: int
    asset: str
    autonomy_level: str
    budget_remaining_steps: int
    current_active_model: dict[str, Any] | None = None
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    feature_sets_available: list[str] = Field(default_factory=lambda: ["FS_D0", "FS_D1", "FS_D2"])
    previous_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    failed_experiments: list[dict[str, Any]] = Field(default_factory=list)
    market_statistics: dict[str, Any] = Field(default_factory=dict)


class AnalysisContext(BaseModel):
    """Context provided to the LLM after experiment execution."""

    run_id: int
    hypothesis: HypothesisProposal
    config_id: int
    metrics: dict[str, Any]
    baseline_comparison: dict[str, Any]
    finalization_gate: dict[str, Any]
    iteration: int
    budget_remaining_steps: int


# ---------------------------------------------------------------------------
# LLM Provider Protocol & Implementations
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM reasoning backends."""

    async def propose_hypothesis(
        self,
        context: AgentContext,
    ) -> tuple[HypothesisProposal, LLMUsageStats]:
        ...

    async def analyze_experiment(
        self,
        context: AnalysisContext,
    ) -> tuple[AgentDecision, LLMUsageStats]:
        ...

    async def summarize_step(
        self,
        step_name: str,
        details: Mapping[str, Any],
    ) -> tuple[str, LLMUsageStats]:
        ...


class MockLLMProvider:
    """Deterministic Mock LLM provider for unit tests and local offline development."""

    def __init__(self, model_name: str = "mock-gpt-5") -> None:
        self.model_name = model_name
        self.call_count = 0

    async def propose_hypothesis(
        self,
        context: AgentContext,
    ) -> tuple[HypothesisProposal, LLMUsageStats]:
        self.call_count += 1
        asset = context.asset or "BTCUSDT"
        clean_asset = asset.replace("USDT", "")

        # Alternate between LogReg and LightGBM across iterations
        family = "LogisticRegression" if (self.call_count % 2 == 1) else "LightGBM"
        feature_set = f"FS_D{self.call_count % 3}"

        proposal = HypothesisProposal(
            hypothesis=f"Iter {self.call_count}: Calibrated {family} on {feature_set} reduces outsider mispricing on {clean_asset}",
            asset=clean_asset,
            market_role="FAVORITE" if self.call_count % 2 == 1 else "OUTSIDER",
            model_family=family,
            feature_set=feature_set,
            parameter_changes={"C": 0.5, "max_iter": 200} if family == "LogisticRegression" else {"n_estimators": 50, "learning_rate": 0.05},
            strategy_parameter_changes={"decision_threshold": 0.58, "min_edge": 0.03},
            expected_effect={"metric": "median_oot_pnl", "direction": "increase", "target_gain": 0.05},
            reasoning=[
                f"Historical OOT evaluations indicate stability for {clean_asset} under {feature_set}",
                "Regularization dampens overconfidence in late-phase trading windows",
            ],
            risks=["Small sample size in extreme market volatility", "Slippage on execution"],
            test_plan={"oot_windows": 3, "min_markets": 50, "execution_mode": "PAPER_REALISTIC"},
        )
        stats = LLMUsageStats(
            prompt_tokens=250,
            completion_tokens=180,
            total_tokens=430,
            estimated_cost_usd=0.001,
            latency_ms=45,
            provider="mock",
            model=self.model_name,
            prompt_hash=hashlib.sha256(b"mock_prompt").hexdigest(),
            response_hash=hashlib.sha256(proposal.hypothesis.encode("utf-8")).hexdigest(),
        )
        return proposal, stats

    async def analyze_experiment(
        self,
        context: AnalysisContext,
    ) -> tuple[AgentDecision, LLMUsageStats]:
        self.call_count += 1
        gate_passed = context.finalization_gate.get(
            "gate_passed",
            context.finalization_gate.get("passed", False),
        )
        median_pnl = context.metrics.get("median_pnl", 0.0)

        if gate_passed and median_pnl > 0:
            action = "RECOMMEND_SHADOW"
            rationale = f"Candidate {context.config_id} passed all strict finalization criteria with positive median PnL ({median_pnl:.4f} USDC)."
        elif context.budget_remaining_steps <= 1:
            action = "FINALIZE_NO_WINNER"
            rationale = f"Budget exhausted after {context.iteration} iterations. No candidate decisively beat baseline with gate clearance."
        else:
            action = "CONTINUE_RESEARCH"
            rationale = f"Experiment {context.config_id} showed promising direction (PnL: {median_pnl:.4f}), mutating regularization for next window."

        decision = AgentDecision(
            action=action,
            rationale=rationale,
            key_findings=[
                f"OOT Net PnL: {median_pnl:.4f} USDC across windows",
                f"Gate status: {'PASSED' if gate_passed else 'REJECTED'}",
            ],
            recommended_config_id=context.config_id if gate_passed else None,
            proposed_overlay={"MIN_EDGE": 0.035} if gate_passed else None,
            next_step_focus="Refine feature set to FS_D2 with tighter outlier clamping" if action == "CONTINUE_RESEARCH" else None,
        )
        stats = LLMUsageStats(
            prompt_tokens=320,
            completion_tokens=150,
            total_tokens=470,
            estimated_cost_usd=0.0012,
            latency_ms=50,
            provider="mock",
            model=self.model_name,
            prompt_hash=hashlib.sha256(b"mock_analysis_prompt").hexdigest(),
            response_hash=hashlib.sha256(decision.rationale.encode("utf-8")).hexdigest(),
        )
        return decision, stats

    async def summarize_step(
        self,
        step_name: str,
        details: Mapping[str, Any],
    ) -> tuple[str, LLMUsageStats]:
        summary_text = f"Step '{step_name}' completed with status: {details.get('status', 'SUCCESS')}. Details: {json.dumps(dict(details), default=str)[:200]}"
        stats = LLMUsageStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.0003,
            latency_ms=20,
            provider="mock",
            model=self.model_name,
        )
        return summary_text, stats


class OpenAIResponsesProvider:
    """OpenAI Responses API provider with structured outputs and zero persistence."""

    def __init__(
        self,
        api_key: str,
        model_research: str = "gpt-5.6",
        model_summary: str = "gpt-5.6-mini",
        store: bool = False,
        timeout_seconds: float = 60.0,
        endpoint_url: str = "https://api.openai.com/v1/responses",
        provider_name: str = "openai",
        route_opencode_models: bool = False,
        opencode_go_responses_endpoint: str = DEFAULT_OPENCODE_GO_RESPONSES_ENDPOINT,
        opencode_go_chat_endpoint: str = DEFAULT_OPENCODE_GO_CHAT_ENDPOINT,
        opencode_go_messages_endpoint: str = DEFAULT_OPENCODE_GO_MESSAGES_ENDPOINT,
    ) -> None:
        self.api_key = api_key
        self.model_research = model_research
        self.model_summary = model_summary
        self.store = store
        self.timeout_seconds = timeout_seconds
        self.endpoint_url = endpoint_url
        self.provider_name = provider_name
        self.route_opencode_models = route_opencode_models
        self.opencode_go_responses_endpoint = opencode_go_responses_endpoint
        self.opencode_go_chat_endpoint = opencode_go_chat_endpoint
        self.opencode_go_messages_endpoint = opencode_go_messages_endpoint

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        # Approximate pricing per 1M tokens
        if "mini" in model.lower():
            return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0
        return (prompt_tokens * 2.50 + completion_tokens * 10.00) / 1_000_000.0


    @staticmethod
    def _kv_items_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": ["string", "number", "boolean", "null"]},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        }

    @classmethod
    def _schema_for(cls, kind: str) -> dict[str, Any]:
        kv = {"type": "array", "items": cls._kv_items_schema()}
        if kind == "hypothesis":
            return {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "asset": {"type": "string"},
                    "market_role": {"type": "string"},
                    "model_family": {"type": "string"},
                    "feature_set": {"type": "string"},
                    "parameter_changes": kv,
                    "strategy_parameter_changes": kv,
                    "expected_effect": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "direction": {"type": "string"},
                            "target_gain": {"type": ["number", "null"]},
                        },
                        "required": ["metric", "direction", "target_gain"],
                        "additionalProperties": False,
                    },
                    "reasoning": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "test_plan": {
                        "type": "object",
                        "properties": {
                            "oot_windows": {"type": "integer"},
                            "min_markets": {"type": "integer"},
                            "execution_mode": {"type": "string"},
                        },
                        "required": ["oot_windows", "min_markets", "execution_mode"],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "hypothesis", "asset", "market_role", "model_family",
                    "feature_set", "parameter_changes",
                    "strategy_parameter_changes", "expected_effect",
                    "reasoning", "risks", "test_plan",
                ],
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "rationale": {"type": "string"},
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "recommended_config_id": {"type": ["integer", "null"]},
                "proposed_overlay": {"type": ["array", "null"], "items": cls._kv_items_schema()},
                "next_step_focus": {"type": ["string", "null"]},
            },
            "required": [
                "action", "rationale", "key_findings",
                "recommended_config_id", "proposed_overlay", "next_step_focus",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        for key in ("parameter_changes", "strategy_parameter_changes", "proposed_overlay"):
            value = result.get(key)
            if isinstance(value, list):
                result[key] = {
                    str(item["key"]): item.get("value")
                    for item in value
                    if isinstance(item, dict) and isinstance(item.get("key"), str)
                }
        return result

    def _endpoint_for_model(self, model: str) -> str:
        """Select the OpenCode transport required by the selected model.

        The default OpenCode catalog contains both Responses and Chat Completions
        models. A custom endpoint is always respected and is never rewritten.
        """
        if self.route_opencode_models and self.provider_name == "opencode":
            spec = OPENCODE_MODEL_SPECS.get(model)
            if spec and self.endpoint_url == DEFAULT_OPENCODE_ENDPOINT:
                protocol = str(spec.get("protocol") or "responses")
                if spec.get("is_go"):
                    return {
                        "responses": self.opencode_go_responses_endpoint,
                        "chat_completions": self.opencode_go_chat_endpoint,
                        "messages": self.opencode_go_messages_endpoint,
                    }.get(protocol, self.endpoint_url)
                if protocol == "chat_completions":
                    return DEFAULT_OPENCODE_CHAT_ENDPOINT
        return self.endpoint_url

    @staticmethod
    def _chat_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, Mapping):
                    value = item.get("text") or item.get("content") or item.get("value")
                    if value:
                        parts.append(str(value))
            return "".join(parts)
        if isinstance(content, Mapping):
            value = content.get("text") or content.get("content") or content.get("value")
            return str(value) if value else ""
        return ""

    @staticmethod
    def _messages_content_text(data: Mapping[str, Any]) -> str:
        for item in data.get("content", []) or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "tool_use" and isinstance(item.get("input"), Mapping):
                return json.dumps(dict(item["input"]), separators=(",", ":"))
            if item.get("type") == "text" and item.get("text"):
                return str(item["text"])
        return ""

    @classmethod
    def _response_text(cls, data: Mapping[str, Any], *, is_chat_completion: bool) -> str:
        if is_chat_completion:
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], Mapping):
                message = choices[0].get("message") or {}
                if isinstance(message, Mapping):
                    parsed = message.get("parsed")
                    if isinstance(parsed, Mapping):
                        return json.dumps(dict(parsed), separators=(",", ":"))
                    text = cls._chat_content_text(message.get("content"))
                    if text:
                        return text
                    refusal = message.get("refusal")
                    if refusal:
                        return str(refusal)
            # A few OpenAI-compatible gateways return output_text even for
            # Chat Completions responses.
            raw = data.get("output_text") or ""
            return str(raw) if raw else ""
        raw = data.get("output_text") or ""
        if raw:
            return str(raw)
        for item in data.get("output", []):
            if not isinstance(item, Mapping):
                continue
            for content in item.get("content", []):
                if not isinstance(content, Mapping):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    raw = content.get("text") or content.get("value")
                    if raw:
                        return str(raw)
        return ""

    async def _responses_json(
        self,
        *,
        model: str,
        instructions: str,
        context: Mapping[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import httpx

        request_endpoint = self._endpoint_for_model(model)
        is_chat_completion = request_endpoint.rstrip("/").endswith("/chat/completions")
        is_messages = request_endpoint.rstrip("/").endswith("/messages")
        if is_messages:
            body = {
                "model": model,
                "max_tokens": 2048,
                "system": instructions,
                "messages": [{
                    "role": "user",
                    "content": json.dumps(dict(context), indent=2, default=str),
                }],
                "tools": [{
                    "name": schema_name,
                    "description": "Return the requested structured result.",
                    "input_schema": schema,
                }],
                "tool_choice": {"type": "tool", "name": schema_name},
            }
        elif is_chat_completion:
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(dict(context), indent=2, default=str)},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
        else:
            body = {
                "model": model,
                "input": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(dict(context), indent=2, default=str)},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
                "store": False,
            }
        if temperature is not None and not model.lower().startswith("gpt-5"):
            body["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if is_messages:
            headers["anthropic-version"] = "2023-06-01"
        started = time.time()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                request_endpoint,
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
        raw = (
            self._messages_content_text(data)
            if is_messages
            else self._response_text(data, is_chat_completion=is_chat_completion)
        )
        if not raw:
            provider_api = (
                "Anthropic Messages"
                if is_messages
                else "Chat Completions"
                if is_chat_completion
                else "Responses"
            )
            raise ValueError(
                f"{self.provider_name.title()} {provider_api} API returned no structured output"
            )
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        completion_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        return self._coerce_payload(json.loads(raw)), {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens) or prompt_tokens + completion_tokens),
            "latency_ms": int((time.time() - started) * 1000),
            "response_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }

    def _stats(self, *, model: str, prompt: Mapping[str, Any], usage: Mapping[str, Any]) -> LLMUsageStats:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return LLMUsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
            estimated_cost_usd=self._compute_cost(prompt_tokens, completion_tokens, model),
            latency_ms=int(usage.get("latency_ms", 0)),
            provider=self.provider_name,
            model=model,
            prompt_hash=hashlib.sha256(json.dumps(dict(prompt), sort_keys=True, default=str).encode("utf-8")).hexdigest(),
            response_hash=str(usage.get("response_hash", "")),
        )

    async def propose_hypothesis(self, context: AgentContext) -> tuple[HypothesisProposal, LLMUsageStats]:
        instructions = (
            "You are an autonomous quant researcher for Polymarket crypto binary markets. "
            "Formulate one testable hypothesis for model architecture, feature set and "
            "strategy parameters. Never propose shell commands, external network calls or direct LIVE trades."
        )
        prompt = {"instructions": instructions, "context": context.model_dump()}
        payload, usage = await self._responses_json(
            model=self.model_research,
            instructions=instructions,
            context={"context": context.model_dump()},
            schema_name="hypothesis_proposal",
            schema=self._schema_for("hypothesis"),
        )
        return HypothesisProposal.model_validate(payload), self._stats(
            model=self.model_research, prompt=prompt, usage=usage
        )

    async def analyze_experiment(self, context: AnalysisContext) -> tuple[AgentDecision, LLMUsageStats]:
        instructions = (
            "Analyze Polymarket-OOT results, compare to baseline, obey the strict finalization "
            "gate, and choose exactly one next action. Request live approval only after the "
            "policy gate; never activate LIVE directly."
        )
        prompt = {"instructions": instructions, "context": context.model_dump()}
        payload, usage = await self._responses_json(
            model=self.model_research,
            instructions=instructions,
            context={"context": context.model_dump()},
            schema_name="agent_decision",
            schema=self._schema_for("decision"),
        )
        return AgentDecision.model_validate(payload), self._stats(
            model=self.model_research, prompt=prompt, usage=usage
        )

    async def summarize_step(self, step_name: str, details: Mapping[str, Any]) -> tuple[str, LLMUsageStats]:
        import httpx

        request_endpoint = self._endpoint_for_model(self.model_summary)
        is_chat_completion = request_endpoint.rstrip("/").endswith("/chat/completions")
        is_messages = request_endpoint.rstrip("/").endswith("/messages")
        system_prompt = "Summarize one execution step in 1-2 concise Russian sentences. Do not invent metrics."
        user_prompt = f"Step: {step_name}\nDetails: {json.dumps(dict(details), default=str)}"
        if is_messages:
            body = {
                "model": self.model_summary,
                "max_tokens": 512,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        elif is_chat_completion:
            body: dict[str, Any] = {
                "model": self.model_summary,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        else:
            body = {
                "model": self.model_summary,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "store": False,
            }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if is_messages:
            headers["anthropic-version"] = "2023-06-01"
        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    request_endpoint,
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()
                data = response.json()
            text = (
                self._messages_content_text(data)
                if is_messages
                else self._response_text(data, is_chat_completion=is_chat_completion)
            ).strip()
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
            return text or f"Шаг {step_name} завершён.", self._stats(
                model=self.model_summary,
                prompt={"step": step_name, "details": dict(details)},
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens", 0),
                    "latency_ms": int((time.time() - started) * 1000),
                    "response_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
            )
        except Exception as exc:
            logger.warning("llm_summary_failed", provider=self.provider_name, error=str(exc))
            return (
                f"Шаг {step_name} завершён. Статус: {details.get('status', 'OK')}",
                LLMUsageStats(provider="fallback", model=self.model_summary),
            )


def get_llm_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    model_research: str | None = None,
    model_summary: str | None = None,
    endpoint_url: str | None = None,
) -> LLMProvider:
    """Factory for Mock, OpenAI and OpenCode-compatible Responses providers."""
    from polyflip.config import settings

    provider = (provider_name or settings.AI_LAB_LLM_PROVIDER or "mock").lower()
    provider_key = (
        getattr(settings, "OPENROUTER_API_KEY", "")
        if provider == "openrouter"
        else getattr(settings, "AI_LAB_LLM_API_KEY", "")
    )
    key = api_key or provider_key or getattr(settings, "AI_LAB_LLM_API_KEY", "") or getattr(settings, "OPENAI_API_KEY", "")
    if provider == "mock":
        return MockLLMProvider(model_name=model_research or "mock-gpt-5")
    if provider in {"openai", "opencode", "openrouter"}:
        if not key:
            key_name = (
                "OPENAI_API_KEY"
                if provider == "openai"
                else "OPENROUTER_API_KEY"
                if provider == "openrouter"
                else "AI_LAB_LLM_API_KEY"
            )
            raise RuntimeError(
                f"AI_LAB_LLM_PROVIDER={provider} requires {key_name}; "
                "set AI_LAB_LLM_PROVIDER=mock explicitly for offline tests"
            )
        default_endpoint = {"openai": "https://api.openai.com/v1/responses", "opencode": DEFAULT_OPENCODE_ENDPOINT, "openrouter": DEFAULT_OPENROUTER_ENDPOINT}[provider]
        configured_endpoint = endpoint_url or getattr(settings, "AI_LAB_LLM_ENDPOINT", "") or ""
        return OpenAIResponsesProvider(
            api_key=key,
            model_research=model_research or getattr(settings, "AI_LAB_MODEL_RESEARCH", "gpt-5.6"),
            model_summary=model_summary or getattr(settings, "AI_LAB_MODEL_SUMMARY", "gpt-5.6-mini"),
            store=False,
            endpoint_url=configured_endpoint or default_endpoint,
            provider_name=provider,
            route_opencode_models=(provider == "opencode" and not configured_endpoint),
            opencode_go_responses_endpoint=str(
                getattr(settings, "AI_LAB_OPENCODE_GO_RESPONSES_ENDPOINT", DEFAULT_OPENCODE_GO_RESPONSES_ENDPOINT)
                or DEFAULT_OPENCODE_GO_RESPONSES_ENDPOINT
            ),
            opencode_go_chat_endpoint=str(
                getattr(settings, "AI_LAB_OPENCODE_GO_CHAT_ENDPOINT", DEFAULT_OPENCODE_GO_CHAT_ENDPOINT)
                or DEFAULT_OPENCODE_GO_CHAT_ENDPOINT
            ),
            opencode_go_messages_endpoint=str(
                getattr(settings, "AI_LAB_OPENCODE_GO_MESSAGES_ENDPOINT", DEFAULT_OPENCODE_GO_MESSAGES_ENDPOINT)
                or DEFAULT_OPENCODE_GO_MESSAGES_ENDPOINT
            ),
        )
    raise ValueError(f"Unsupported AI_LAB_LLM_PROVIDER: {provider}")
