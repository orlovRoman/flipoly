import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from polyflip.ai_lab.thread_provider import (
    MockThreadProvider,
    ThreadProviderStatus,
    build_safe_prompt,
    ensure_agent_thread,
    get_thread_provider,
)


class _Session:
    def __init__(self):
        self.flushes = 0

    async def flush(self):
        self.flushes += 1


def test_mock_thread_is_explicitly_offline_and_resume_keeps_id():
    async def run():
        provider = MockThreadProvider()
        first = await provider.create("hello")
        second = await provider.resume(first.thread_id, "again")
        assert first.status is ThreadProviderStatus.OK
        assert second.thread_id == first.thread_id
        assert len(provider.prompts) == 2

    asyncio.run(run())


def test_thread_id_is_persisted_only_after_create_and_resume_uses_same_thread():
    async def run():
        session = _Session()
        run_row = SimpleNamespace(agent_thread_id=None)
        provider = MockThreadProvider()
        created = await ensure_agent_thread(session, run_row, provider, "objective")
        resumed = await ensure_agent_thread(session, run_row, provider, "objective")
        assert run_row.agent_thread_id == created.thread_id == resumed.thread_id
        assert session.flushes == 1

    asyncio.run(run())


def test_missing_codex_sdk_is_not_configured_not_mock():
    with patch("polyflip.ai_lab.thread_provider.official_sdk_available", return_value=False):
        provider = get_thread_provider("codex")

    result = asyncio.run(provider.create("hello"))
    assert result.status is ThreadProviderStatus.NOT_CONFIGURED
    assert not isinstance(provider, MockThreadProvider)


def test_prompt_redaction_removes_secrets_and_raw_trading_data():
    prompt = build_safe_prompt(
        "use sk-testsecret",
        {
            "api_key": "sk-live-secret",
            "order": {"price": 0.5, "token": "abc"},
            "safe_note": "keep this",
        },
    )
    assert "sk-live-secret" not in prompt
    assert "price" not in prompt
    assert "keep this" in prompt
    assert "REDACTED_SECRET" in prompt
    assert "REDACTED_TRADING_DATA" in prompt
