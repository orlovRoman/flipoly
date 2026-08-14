import asyncio

import pytest

from polyflip.ai_lab import executor
from polyflip.crypto.polymarket_backtest import (
    aggregate_stored_polymarket_backtests,
)


def test_legacy_polymarket_summary_keeps_default_invested_capital():
    summary = aggregate_stored_polymarket_backtests(
        [
            {
                "n_markets": 4,
                "n_quotes": 4,
                "n_oof": 4,
                "n_eligible": 2,
                "n_trades": 2,
                "net_profit": 1.0,
                "total_invested": 0.0,
                "avg_edge": 0.1,
                "avg_net_edge": 0.08,
                "win_rate": 0.5,
                "equity_curve": [],
                "slices": [],
            }
        ],
        strategy_branch="COMBINED",
    )

    assert summary["total_invested"] == pytest.approx(2.0)
    assert summary["roi_pct"] == pytest.approx(50.0)


class _AuditSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, value):
        self.added.append(value)

    async def get(self, _model, _identifier):
        return SimpleNamespace(objective="audit", scope={})

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None


class SimpleNamespace:
    def __init__(self, **values):
        self.__dict__.update(values)


def test_invalid_step_audit_preserves_json_null_for_missing_config_id(monkeypatch):
    step = SimpleNamespace(
        id=101,
        action="PLACE_ORDER",
        input_payload={},
        status="RUNNING",
        finished_at=None,
        summary=None,
        error_code=None,
        error_message=None,
    )
    session = _AuditSession()

    async def claim(_session, _run_id):
        return step

    monkeypatch.setattr(executor, "claim_next_step", claim)

    result = asyncio.run(
        executor.execute_next_step(session, 1, executor.AdapterRegistry())
    )

    assert result.error_code == "INVALID_STEP_INPUT"
    assert session.added[0].payload["raw_config_id"] is None


@pytest.mark.asyncio
async def test_in_memory_training_slot_returns_already_running():
    from polyflip.api.crypto_dashboard import _active_trainings, crypto_train

    symbol = "BTCUSDT"
    _active_trainings.pop(symbol, None)
    try:
        first = await crypto_train(
            symbol=symbol,
            interval="15m",
            feature_set="A",
            db=None,
        )
        second = await crypto_train(
            symbol=symbol,
            interval="15m",
            feature_set="A",
            db=None,
        )
        assert first["status"] == "started"
        assert second["status"] == "already_running"
    finally:
        _active_trainings.pop(symbol, None)
