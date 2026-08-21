"""Codex thread lifecycle for the AI Lab.

This module deliberately does not make the existing LLM provider pretend to be
a Codex thread.  Thread support is optional: an absent official SDK is a
configuration state, not a reason to use the offline mock.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol


class ThreadProviderStatus(str, Enum):
    OK = "OK"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ThreadResult:
    status: ThreadProviderStatus
    thread_id: str | None = None
    response: str | None = None
    error: str | None = None


class AgentThreadProvider(Protocol):
    """Small provider contract so the service boundary owns persistence."""

    async def create(self, prompt: str) -> ThreadResult: ...

    async def resume(self, thread_id: str, prompt: str) -> ThreadResult: ...


class MockThreadProvider:
    """Explicit offline provider used by tests; never selected implicitly."""

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []
        self._next_id = 1

    async def create(self, prompt: str) -> ThreadResult:
        thread_id = f"mock-thread-{self._next_id}"
        self._next_id += 1
        self.prompts.append((thread_id, prompt))
        return ThreadResult(ThreadProviderStatus.OK, thread_id, "mock")

    async def resume(self, thread_id: str, prompt: str) -> ThreadResult:
        self.prompts.append((thread_id, prompt))
        return ThreadResult(ThreadProviderStatus.OK, thread_id, "mock")


class NotConfiguredThreadProvider:
    """Sentinel provider: optional Codex support is unavailable."""

    async def create(self, prompt: str) -> ThreadResult:
        return ThreadResult(ThreadProviderStatus.NOT_CONFIGURED, error="Codex official SDK is not installed")

    async def resume(self, thread_id: str, prompt: str) -> ThreadResult:
        return ThreadResult(ThreadProviderStatus.NOT_CONFIGURED, thread_id=thread_id, error="Codex official SDK is not installed")

_SDK_NAMES = ("codex_sdk", "openai_codex", "codex")


def official_sdk_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _SDK_NAMES)


class CodexThreadProvider:
    """Thin adapter around an installed official Codex SDK.

    SDK releases expose either ``client.threads`` or a module-level client;
    keeping the adapter duck-typed avoids adding an optional dependency to the
    trading application.  Unsupported SDK shapes return ERROR rather than
    falling back to MockThreadProvider.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            name = next((n for n in _SDK_NAMES if importlib.util.find_spec(n)), None)
            if not name:
                raise RuntimeError("Codex official SDK is not installed")
            client = importlib.import_module(name)
        self.client = client

    async def _call(self, operation: str, *args: Any, **kwargs: Any) -> ThreadResult:
        try:
            threads = getattr(self.client, "threads", self.client)
            method = getattr(threads, operation)
            value = method(*args, **kwargs)
            if hasattr(value, "__await__"):
                value = await value
            thread_id = getattr(value, "id", None) or (value.get("id") if isinstance(value, dict) else None)
            if not thread_id and operation == "resume":
                thread_id = args[0] if args else kwargs.get("thread_id")
            return ThreadResult(ThreadProviderStatus.OK, str(thread_id) if thread_id else None, str(getattr(value, "text", "")) or None)
        except Exception as exc:
            return ThreadResult(ThreadProviderStatus.ERROR, error=str(exc))

    async def create(self, prompt: str) -> ThreadResult:
        return await self._call("create", prompt=prompt)

    async def resume(self, thread_id: str, prompt: str) -> ThreadResult:
        return await self._call("resume", thread_id=thread_id, prompt=prompt)


def get_thread_provider(provider_name: str | None = None) -> AgentThreadProvider:
    """Return an explicitly configured provider; unavailable Codex is explicit."""
    name = (provider_name or "codex").strip().lower()
    if name == "mock":
        return MockThreadProvider()
    if name != "codex" or not official_sdk_available():
        return NotConfiguredThreadProvider()
    return CodexThreadProvider()


_SECRET_KEY = re.compile(r"(api[_-]?key|secret|token|password|private[_-]?key|database[_-]?url|credential)", re.I)
# Only raw event collections are sensitive.  Do not redact aggregate research
# metrics such as ``trade_count`` or ``total_trades`` from the agent context.
_RAW_TRADING_KEY = re.compile(
    r"^(?:raw[_-]?(?:trades?|orders?|fills?|positions?|executions?|data)|"
    r"(?:trade|order|fill|position|execution)[_-](?:records?|history|events?|snapshots?)|"
    r"order(?:book)?|market[_-]?data)$",
    re.I,
)
_SECRET_VALUE = re.compile(r"(?i)\b(sk-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._-]+)\b")


def redact_prompt(value: Any, *, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED_SECRET]"
    if _RAW_TRADING_KEY.search(key):
        return "[REDACTED_TRADING_DATA]"
    if isinstance(value, Mapping):
        return {str(k): redact_prompt(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_prompt(v, key=key) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED_SECRET]", value)
    return value


def build_safe_prompt(objective: str, context: Mapping[str, Any] | None = None) -> str:
    safe = redact_prompt(dict(context or {}))
    return f"Objective: {redact_prompt(objective)}\nContext: {json.dumps(safe, sort_keys=True, default=str)}"


async def ensure_agent_thread(session: Any, run: Any, provider: AgentThreadProvider, prompt: str) -> ThreadResult:
    """Create once, then resume the persisted thread on subsequent iterations."""
    safe_prompt = build_safe_prompt(prompt)
    result = await (provider.resume(run.agent_thread_id, safe_prompt) if run.agent_thread_id else provider.create(safe_prompt))
    if result.status is ThreadProviderStatus.OK and result.thread_id and not run.agent_thread_id:
        run.agent_thread_id = result.thread_id
        await session.flush()
    return result
