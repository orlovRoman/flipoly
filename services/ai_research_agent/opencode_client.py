"""OpenCode LLM client for the independent research agent.

Mirrors the structured-output behavior of ``polyflip.ai_lab.llm`` without
importing platform code: the same JSON schemas are sent over either the
Responses or Chat Completions transport, selected per model.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

DEFAULT_OPENCODE_ENDPOINT = "https://opencode.ai/zen/v1/responses"
DEFAULT_OPENCODE_CHAT_ENDPOINT = "https://opencode.ai/zen/v1/chat/completions"
DEFAULT_OPENCODE_GO_RESPONSES_ENDPOINT = "https://opencode.ai/zen/go/v1/responses"
DEFAULT_OPENCODE_GO_CHAT_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
DEFAULT_OPENCODE_GO_MESSAGES_ENDPOINT = "https://opencode.ai/zen/go/v1/messages"
DEFAULT_LLM_TIMEOUT_SECONDS = 180.0
MIN_LLM_TIMEOUT_SECONDS = 5.0
MAX_LLM_TIMEOUT_SECONDS = 900.0


def _configured_timeout_seconds() -> float:
    raw = os.getenv("AI_LAB_LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = DEFAULT_LLM_TIMEOUT_SECONDS
    return min(max(value, MIN_LLM_TIMEOUT_SECONDS), MAX_LLM_TIMEOUT_SECONDS)

_OPENCODE_GO_MODEL_PROTOCOLS = {
    "grok-4.6": "responses",
    "glm-5.3-flash": "chat_completions",
    "glm-5.3": "chat_completions",
    "glm-5.2": "chat_completions",
    "glm-5.1": "chat_completions",
    "gpt-5.6-luna": "responses",
    "kimi-k3": "chat_completions",
    "kimi-k2.7-code": "chat_completions",
    "kimi-k2.6": "chat_completions",
    "longcat-2.0": "chat_completions",
    "mimo-v2.5": "chat_completions",
    "mimo-v2.5-pro": "chat_completions",
    "minimax-m3": "messages",
    "minimax-m2.7": "messages",
    "muse-spark-1.2-contributor": "responses",
    "qwen3.8-max": "messages",
    "qwen3.8-flash": "messages",
    "qwen3.7-max": "messages",
    "qwen3.7-plus": "messages",
    "qwen3.6-plus": "messages",
    "deepseek-v4-pro": "chat_completions",
    "deepseek-v4-flash": "chat_completions",
    "deepseek-v4-flash-vision-exp": "chat_completions",
    "hy3": "chat_completions",
}
_OPENCODE_FREE_MODEL_PROTOCOLS = {
    "big-pickle": "chat_completions",
    "x-preview-f-free": "chat_completions",
    "mimo-v2.5-free": "chat_completions",
    "hy3-free": "chat_completions",
    "nemotron-3-ultra-free": "chat_completions",
    "nemotron-3.5-lightning-free": "chat_completions",
    "muse-spark-1.2-contributor-free": "responses",
}
OPENCODE_GO_MODELS = frozenset(_OPENCODE_GO_MODEL_PROTOCOLS)
OPENCODE_MODEL_SPECS = {
    model_id: {"protocol": protocol, "is_go": True}
    for model_id, protocol in _OPENCODE_GO_MODEL_PROTOCOLS.items()
}
OPENCODE_MODEL_SPECS.update(
    {
        model_id: {"protocol": protocol, "is_go": False}
        for model_id, protocol in _OPENCODE_FREE_MODEL_PROTOCOLS.items()
    }
)
DEFAULT_OPENCODE_CHAT_MODELS = {
    model_id
    for model_id, spec in OPENCODE_MODEL_SPECS.items()
    if spec["protocol"] == "chat_completions"
}
DEFAULT_RESPONSES_ENDPOINT = DEFAULT_OPENCODE_ENDPOINT
DEFAULT_CHAT_ENDPOINT = DEFAULT_OPENCODE_CHAT_ENDPOINT
DEFAULT_CHAT_MODELS = set(DEFAULT_OPENCODE_CHAT_MODELS)
DEFAULT_OPENROUTER_CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODELS = {
    "x-ai/grok-4.6",
    "openai/gpt-5.6-luna",
    "z-ai/glm-5.3-flash",
    "z-ai/glm-5.3",
    "z-ai/glm-5.2",
    "z-ai/glm-5.1",
    "moonshotai/kimi-k3",
    "moonshotai/kimi-k2.7-code",
    "moonshotai/kimi-k2.6",
    "meituan/longcat-2.0",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2.5-pro",
    "minimax/minimax-m3",
    "minimax/minimax-m2.7",
    "meta/muse-spark-1.2-contributor",
    "qwen/qwen3.8-max",
    "qwen/qwen3.7-max",
    "qwen/qwen3.7-plus",
    "qwen/qwen3.6-plus",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-vision-exp",
    "tencent/hy3",
}
ALLOWED_MARKET_ROLES = ("FAVORITE", "OUTSIDER", "COMBINED", "DIRECTION_ONLY", "ALL")
_MARKET_ROLE_ALIASES = {"TAKER": "OUTSIDER"}
ALLOWED_AGENT_ACTIONS = (
    "CONTINUE_RESEARCH",
    "MUTATE_HYPOTHESIS",
    "RECOMMEND_SHADOW",
    "FINALIZE_NO_WINNER",
    "APPLY_OVERLAY",
    "REQUEST_LIVE_APPROVAL",
    "STOP_BUDGET_EXHAUSTED",
)
AGENT_ACTION_ALIASES = {
    # Historical model wording: explicitly paper-only, so it maps to the
    # existing action that requeues research when budget remains.
    "HOLD_LIVE_RETRY_PAPER_TRAIN_ONLY": "CONTINUE_RESEARCH",
}


def _kv_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": ["string", "number", "boolean", "null"]},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    }


def _hypothesis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "asset": {"type": "string"},
            "market_role": {"type": "string", "enum": list(ALLOWED_MARKET_ROLES)},
            "model_family": {"type": "string"},
            "feature_set": {"type": "string"},
            "parameter_changes": _kv_schema(),
            "strategy_parameter_changes": _kv_schema(),
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
            "hypothesis",
            "asset",
            "market_role",
            "model_family",
            "feature_set",
            "parameter_changes",
            "strategy_parameter_changes",
            "expected_effect",
            "reasoning",
            "risks",
            "test_plan",
        ],
        "additionalProperties": False,
    }


def _decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ALLOWED_AGENT_ACTIONS)},
            "rationale": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "recommended_config_id": {"type": ["integer", "null"]},
            "proposed_overlay": {
                "type": ["array", "null"],
                "items": _kv_schema()["items"],
            },
            "next_step_focus": {"type": ["string", "null"]},
        },
        "required": [
            "action",
            "rationale",
            "key_findings",
            "recommended_config_id",
            "proposed_overlay",
            "next_step_focus",
        ],
        "additionalProperties": False,
    }


def _coerce_kv_lists(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("parameter_changes", "strategy_parameter_changes", "proposed_overlay"):
        value = result.get(key)
        if isinstance(value, list):
            result[key] = {
                str(item["key"]): item.get("value")
                for item in value
                if isinstance(item, dict) and isinstance(item.get("key"), str)
            }
    role = result.get("market_role")
    if isinstance(role, str):
        normalized = role.strip().upper()
        result["market_role"] = _MARKET_ROLE_ALIASES.get(normalized, normalized)
    return result


def _coerce_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = _coerce_kv_lists(payload)
    action = result.get("action")
    if isinstance(action, str):
        normalized = action.strip().upper()
        result["action"] = AGENT_ACTION_ALIASES.get(normalized, normalized)
    return result


def _usage_telemetry(data: dict[str, Any], latency_ms: int, *, model: str) -> dict[str, Any]:
    usage = data.get("usage") if isinstance(data, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total = usage.get("total_tokens")
    try:
        prompt = int(prompt or 0)
    except (TypeError, ValueError):
        prompt = 0
    try:
        completion = int(completion or 0)
    except (TypeError, ValueError):
        completion = 0
    try:
        total = int(total) if total is not None else prompt + completion
    except (TypeError, ValueError):
        total = prompt + completion
    return {
        "latency_ms": int(latency_ms),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "model": model,
    }

def _model_selection(
    context: dict[str, Any],
    role: str,
    *,
    fallback_role: str | None = None,
) -> tuple[str, str | None]:
    """Resolve the immutable model/protocol selection for one agent role."""
    selected = context.get(role)
    model = None
    protocol = None
    if isinstance(selected, dict):
        model = selected.get("model_id") or selected.get("model") or selected.get("id")
        protocol = selected.get("protocol")
    if not model:
        model = context.get(f"{role}_model")
    if not model and fallback_role:
        fallback = context.get(fallback_role)
        if isinstance(fallback, dict):
            model = (
                fallback.get("model_id")
                or fallback.get("model")
                or fallback.get("id")
            )
            protocol = protocol or fallback.get("protocol")
        if not model:
            model = context.get(f"{fallback_role}_model")
    if not model:
        model = "gpt-5.6"
    if not protocol:
        protocol = context.get(f"{role}_protocol")
    if not protocol and fallback_role:
        protocol = context.get(f"{fallback_role}_protocol")
    if not protocol:
        protocol = context.get("protocol")
    return str(model), (str(protocol) if protocol else None)


class OpenCodeClient:
    def __init__(self) -> None:
        self.provider = os.getenv("AI_LAB_LLM_PROVIDER", "opencode").strip().lower()
        self.api_key = os.getenv("AI_LAB_LLM_API_KEY", "")
        self.request_timeout_seconds = _configured_timeout_seconds()
        if self.provider == "openrouter" and not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        if self.provider == "openrouter":
            self.responses_endpoint = os.getenv(
                "AI_LAB_OPENROUTER_ENDPOINT", DEFAULT_OPENROUTER_CHAT_ENDPOINT
            )
            self.chat_endpoint = self.responses_endpoint
            default_chat_models = DEFAULT_OPENROUTER_MODELS
            chat_models_csv = os.getenv("AI_LAB_OPENROUTER_MODELS", "")
        else:
            self.responses_endpoint = os.getenv(
                "AI_LAB_OPENCODE_RESPONSES_ENDPOINT", DEFAULT_RESPONSES_ENDPOINT
            )
            self.chat_endpoint = os.getenv(
                "AI_LAB_OPENCODE_CHAT_ENDPOINT", DEFAULT_CHAT_ENDPOINT
            )
            self.go_responses_endpoint = os.getenv(
                "AI_LAB_OPENCODE_GO_RESPONSES_ENDPOINT",
                DEFAULT_OPENCODE_GO_RESPONSES_ENDPOINT,
            )
            self.go_chat_endpoint = os.getenv(
                "AI_LAB_OPENCODE_GO_CHAT_ENDPOINT",
                DEFAULT_OPENCODE_GO_CHAT_ENDPOINT,
            )
            self.go_messages_endpoint = os.getenv(
                "AI_LAB_OPENCODE_GO_MESSAGES_ENDPOINT",
                DEFAULT_OPENCODE_GO_MESSAGES_ENDPOINT,
            )
            default_chat_models = DEFAULT_CHAT_MODELS | {
                model_id
                for model_id, spec in OPENCODE_MODEL_SPECS.items()
                if spec["protocol"] == "chat_completions"
            }
            chat_models_csv = os.getenv("AI_LAB_OPENCODE_CHAT_MODELS", "")
        self.chat_models = {
            item.strip() for item in chat_models_csv.split(",") if item.strip()
        } or set(default_chat_models)

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            referer = os.getenv("AI_LAB_OPENROUTER_HTTP_REFERER", "").strip()
            title = os.getenv("AI_LAB_OPENROUTER_X_TITLE", "").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
        return headers

    def _endpoint_for(self, model: str) -> tuple[str, bool]:
        spec = OPENCODE_MODEL_SPECS.get(model)
        if self.provider == "opencode" and spec:
            return self._endpoint_for_protocol(spec["protocol"], model)
        is_chat = model in self.chat_models
        return (
            (self.chat_endpoint, True) if is_chat else (self.responses_endpoint, False)
        )

    def _endpoint_for_protocol(
        self,
        protocol: str | None,
        model: str | None = None,
    ) -> tuple[str, bool]:
        if self.provider == "openrouter":
            # OpenRouter exposes this catalogue through Chat Completions.
            return (self.chat_endpoint, True)
        if self.provider == "opencode" and model in OPENCODE_GO_MODELS:
            if protocol == "messages":
                return (self.go_messages_endpoint, False)
            if protocol == "chat_completions":
                return (self.go_chat_endpoint, True)
            return (self.go_responses_endpoint, False)
        if protocol == "messages":
            return (self.chat_endpoint, False)
        if protocol == "chat_completions":
            return (self.chat_endpoint, True)
        if protocol == "responses":
            return (self.responses_endpoint, False)
        # Fallback to responses for unknown or mock
        return (self.responses_endpoint, False)

    @staticmethod
    def _mock_payload(schema_name: str, context: dict[str, Any]) -> dict[str, Any]:
        root = context.get("context") if isinstance(context, dict) else context
        root = root if isinstance(root, dict) else {}
        scope = root.get("scope") if isinstance(root.get("scope"), dict) else {}
        asset = str(scope.get("asset") or "BTC").upper().replace("USDT", "")
        if asset not in {"BTC", "ETH", "SOL", "XRP", "DOGE"}:
            asset = "BTC"
        if schema_name == "hypothesis_proposal":
            return {
                "hypothesis": f"Deterministic mock baseline for {asset} outsider markets",
                "asset": asset,
                "market_role": "OUTSIDER",
                "model_family": "LOGREG",
                "feature_set": "FS_D0",
                "parameter_changes": {},
                "strategy_parameter_changes": {},
                "expected_effect": {
                    "metric": "median_oot_pnl",
                    "direction": "increase",
                    "target_gain": 0.0,
                },
                "reasoning": ["mock provider is deterministic and offline"],
                "risks": ["synthetic output is not evidence of performance"],
                "test_plan": {
                    "oot_windows": 3,
                    "min_markets": 50,
                    "execution_mode": "PAPER_REALISTIC",
                },
            }
        if schema_name == "agent_decision":
            return {
                "action": "FINALIZE_NO_WINNER",
                "rationale": "Deterministic mock provider completed without promotion.",
                "key_findings": ["mock output must be replaced by a real model"],
                "recommended_config_id": None,
                "proposed_overlay": None,
                "next_step_focus": None,
            }
        raise ValueError(f"unsupported mock schema: {schema_name}")
    async def _structured_json(
        self,
        *,
        model: str,
        instructions: str,
        context: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        protocol: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.provider == "mock":
            payload = self._mock_payload(schema_name, context)
            return payload, _usage_telemetry({}, 0, model=model)

        # Use explicit protocol when provided (snapshot-provided), else use
        # the curated model specification to select transport and endpoint.
        selected_protocol = protocol or OPENCODE_MODEL_SPECS.get(model, {}).get("protocol")
        if selected_protocol:
            endpoint, is_chat = self._endpoint_for_protocol(selected_protocol, model)
        else:
            endpoint, is_chat = self._endpoint_for(model)
        is_messages = selected_protocol == "messages"
        user_content = json.dumps(context, indent=2, default=str)
        if is_messages:
            body: dict[str, Any] = {
                "model": model,
                "max_tokens": 4096,
                "system": instructions,
                "messages": [{"role": "user", "content": user_content}],
                "tools": [{
                    "name": schema_name,
                    "description": "Return the requested structured result.",
                    "input_schema": schema,
                }],
                "tool_choice": {"type": "tool", "name": schema_name},
            }
        elif is_chat:
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_content},
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
                    {"role": "user", "content": user_content},
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
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            headers = self._request_headers()
            if is_messages:
                headers["anthropic-version"] = "2023-06-01"
            response = await client.post(
                endpoint,
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
        latency_ms = int((time.monotonic() - started) * 1000)
        telemetry = _usage_telemetry(data, latency_ms, model=model)
        if is_messages:
            text = ""
            for part in data.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool_use" and isinstance(part.get("input"), dict):
                    text = json.dumps(part["input"])
                    break
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    text = part["text"]
                    break
        elif is_chat:
            choices = data.get("choices") or []
            text = ""
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                if isinstance(message, dict):
                    parsed = message.get("parsed")
                    if isinstance(parsed, dict):
                        text = json.dumps(parsed)
                    else:
                        content = message.get("content")
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            text = "".join(
                                str(part.get("text") or part.get("content") or "")
                                for part in content
                                if isinstance(part, dict)
                            )
                        if not text and message.get("refusal"):
                            text = str(message["refusal"])
            if not text and isinstance(data.get("output_text"), str):
                text = data["output_text"]
        else:
            text = data.get("output_text") or ""
            if not text:
                for item in data.get("output", []):
                    if not isinstance(item, dict):
                        continue
                    for part in item.get("content", []):
                        if (
                            isinstance(part, dict)
                            and part.get("type") in {"output_text", "text"}
                            and part.get("text")
                        ):
                            text = str(part["text"])
                            break
                    if text:
                        break
        if not text:
            raise ValueError(f"{schema_name}: empty structured output from {model} ({self.provider})")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{schema_name}: structured output must be an object")
        if schema_name == "agent_decision":
            payload = _coerce_decision_payload(payload)
        else:
            payload = _coerce_kv_lists(payload)
        return payload, telemetry

    async def propose_hypothesis(self, context: dict[str, Any]) -> dict[str, Any]:
        # Snapshot provides explicit per-model protocol; use it when available.
        model, protocol = _model_selection(context, "research")
        payload, telemetry = await self._structured_json(
            model=model,
            protocol=protocol,
            instructions=(
                "You are an autonomous quant researcher for Polymarket crypto "
                "binary markets. Formulate one testable hypothesis for model "
                "architecture, feature set and strategy parameters. Never "
                "propose shell commands, external network calls or LIVE trades."
            ),
            context={"context": context},
            schema_name="hypothesis_proposal",
            schema=_hypothesis_schema(),
        )
        return {"proposal": payload, "telemetry": telemetry, **telemetry}

    async def decide(
        self,
        *,
        context: dict[str, Any],
        proposal: dict[str, Any],
        result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        model, protocol = _model_selection(
            context, "summary", fallback_role="research"
        )
        payload, telemetry = await self._structured_json(
            model=model,
            protocol=protocol,
            instructions=(
                "Analyze Polymarket-OOT results versus baseline and choose one "
                "exactly one action from this list: CONTINUE_RESEARCH, "
                "MUTATE_HYPOTHESIS, RECOMMEND_SHADOW, FINALIZE_NO_WINNER, "
                "APPLY_OVERLAY, REQUEST_LIVE_APPROVAL, or "
                "STOP_BUDGET_EXHAUSTED. Use the exact uppercase token. Never "
                "request direct LIVE activation."
            ),
            context={
                "context": context,
                "proposal": proposal,
                "result": result,
            },
            schema_name="agent_decision",
            schema=_decision_schema(),
        )
        return {"decision": payload, "telemetry": telemetry, **telemetry}
