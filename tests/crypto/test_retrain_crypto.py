from unittest.mock import AsyncMock

import pytest

from polyflip.scripts import retrain_crypto


class _SessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, traceback):
        return False


@pytest.mark.asyncio
async def test_retrain_continues_after_failure_and_reports_truthful_status(monkeypatch):
    outcomes = {"BTCUSDT": True, "ETHUSDT": False, "SOLUSDT": RuntimeError("boom")}
    calls = []

    class FakeTrainer:
        def __init__(self, session):
            self.session = session

        async def train(self, symbol, interval):
            calls.append((symbol, interval))
            outcome = outcomes[symbol]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    sleep = AsyncMock()
    monkeypatch.setattr(retrain_crypto, "async_session", _SessionContext)
    monkeypatch.setattr(retrain_crypto, "CryptoModelTrainer", FakeTrainer)
    monkeypatch.setattr(retrain_crypto.asyncio, "sleep", sleep)

    results = await retrain_crypto.run_sequential_retrain(
        tuple(outcomes),
        pause_seconds=0.25,
    )

    assert calls == [
        ("BTCUSDT", "15m"),
        ("ETHUSDT", "15m"),
        ("SOLUSDT", "15m"),
    ]
    assert results == {
        "BTCUSDT": "COMPLETED",
        "ETHUSDT": "FAILED_OR_EMPTY",
        "SOLUSDT": "ERROR: boom",
    }
    assert sleep.await_count == 2
    assert retrain_crypto.exit_code_for(results) == 1


@pytest.mark.asyncio
async def test_single_symbol_does_not_sleep(monkeypatch):
    class FakeTrainer:
        def __init__(self, session):
            self.session = session

        async def train(self, symbol, interval):
            return True

    sleep = AsyncMock()
    monkeypatch.setattr(retrain_crypto, "async_session", _SessionContext)
    monkeypatch.setattr(retrain_crypto, "CryptoModelTrainer", FakeTrainer)
    monkeypatch.setattr(retrain_crypto.asyncio, "sleep", sleep)

    results = await retrain_crypto.run_sequential_retrain(
        ("BTCUSDT",), pause_seconds=0.25,
    )

    assert results == {"BTCUSDT": "COMPLETED"}
    sleep.assert_not_awaited()


def test_empty_symbol_list_does_not_succeed():
    assert retrain_crypto.exit_code_for({}) == 1


def test_exit_code_requires_at_least_one_successful_result():
    assert retrain_crypto.exit_code_for({"BTCUSDT": "COMPLETED"}) == 0
    assert retrain_crypto.exit_code_for({}) == 1
