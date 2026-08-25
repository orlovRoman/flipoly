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

DEFAULT_RESPONSES_ENDPOINT = "https://opencode.ai/zen/v1/responses"
DEFAULT_CHAT_ENDPOINT = "https://opencode.ai/zen/v1/chat/completions"
DEFAULT_CHAT_MODELS = {"big-pickle", "nemotron-3-ultra-free"}


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
            "market_role": {"type": "string"},
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
            "hypothesis", "asset", "market_role", "model_family", "feature_set",
            "parameter_changes", "strategy_parameter_changes", "expected_effect",
            "reasoning", "risks", "test_plan",
        ],
        "additionalProperties": False,
    }


def _decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "rationale": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "recommended_config_id": {"type": ["integer", "null"]},
            "proposed_overlay": {
                "type": ["array", "null"], "items": _kv_schema()
            },
            "next_step_focus": {"type": ["string", "null"]},
        },
        "required": [
            "action", "rationale", "key_findings", "recommended_config_id",
            "proposed_overlay", "next_step_focus",
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
    return result


class OpenCodeClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("AI_LAB_LLM_API_KEY", "")
        self.responses_endpoint = os.getenv(
            "AI_LAB_OPENCODE_RESPONSES_ENDPOINT", DEFAULT_RESPONSES_ENDPOINT
        )
        self.chat_endpoint = os.getenv(
            "AI_LAB_OPENCODE_CHAT_ENDPOINT", DEFAULT_CHAT_ENDPOINT
        )
        chat_models_csv = os.getenv("AI_LAB_OPENCODE_CHAT_MODELS", "")
        self.chat_models = (
            {item.strip() for item in chat_models_csv.split(",") if item.strip()}
            or set(DEFAULT_CHAT_MODELS)
        )

    def _endpoint_for(self, model: str) -> tuple[str, bool]:
        is_chat = model in self.chat_models
        return (
            (self.chat_endpoint, True)
            if is_chat
            else (self.responses_endpoint, False)
        )

    def _endpoint_for_protocol(self, protocol: str | None) -> tuple[str, bool]:
        if protocol == "chat_completions":
            return (self.chat_endpoint, True)
        if protocol == "responses":
            return (self.responses_endpoint, False)
        # Fallback to responses for unknown or mock
        return (self.responses_endpoint, False)

    async def _structured_json(
        self,
        *,
        model: str,
        instructions: str,
        context: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        protocol: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        # Use explicit protocol when provided (snapshot-provided), else guess via model.
        if protocol:
            endpoint, is_chat = self._endpoint_for_protocol(protocol)
        else:
            endpoint, is_chat = self._endpoint_for(model)
        user_content = json.dumps(context, indent=2, default=str)
        if is_chat:
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_content},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name, "strict": True, "schema": schema,
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            data = response.json()
        latency_ms = int((time.monotonic() - started) * 1000)
        if is_chat:
            choices = data.get("choices") or []
            text = ""
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    text = content
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
            raise ValueError(f"{schema_name}: empty structured output from {model}")
        return _coerce_kv_lists(json.loads(text)), latency_ms

    async def propose_hypothesis(self, context: dict[str, Any]) -> dict[str, Any]:
        # Snapshot provides explicit per-model protocol; use it when available.
        research = context.get("research")
        if isinstance(research, dict):
            model = str(research.get("model_id") or context.get("research_model") or "gpt-5.6")
            protocol = str(research.get("protocol") or "")
            if not protocol:
                protocol = None
        else:
            model = str(context.get("research_model") or "gpt-5.6")
            protocol = str(context.get("research_protocol") or context.get("protocol") or "")
            protocol = protocol or None
        payload, latency_ms = await self._structured_json(
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
        return {"proposal": payload, "latency_ms": latency_ms}

    async def decide(
        self,
        *,
        context: dict[str, Any],
        proposal: dict[str, Any],
        result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        research = context.get("research")
        if isinstance(research, dict):
            model = str(research.get("model_id") or context.get("research_model") or "gpt-5.6")
            protocol = str(research.get("protocol") or "")
            protocol = protocol or None
        else:
            model = str(context.get("research_model") or "gpt-5.6")
            protocol = str(context.get("research_protocol") or context.get("protocol") or "")
            protocol = protocol or None
        payload, latency_ms = await self._structured_json(
            model=model,
            protocol=protocol,
            instructions=(
                "Analyze Polymarket-OOT results versus baseline and choose one "
                "action. Never request direct LIVE activation."
            ),
            context={
                "context": context,
                "proposal": proposal,
                "result": result,
            },
            schema_name="agent_decision",
            schema=_decision_schema(),
        )
        return {"decision": payload, "latency_ms": latency_ms}
