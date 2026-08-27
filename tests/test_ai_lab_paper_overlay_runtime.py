"""Runtime PAPER-overlay metrics regression tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from polyflip.ai_lab.paper_overlay import (
    get_paper_overlay_runtime_summary,
    resolve_paper_runtime_settings,
)
from polyflip.ai_lab.service import create_permission, create_run
from polyflip.db.models import AIConfigOverlay, TradeHistory
from polyflip.trading.trade_recorder import save_or_update_skipped_trade


async def _make_run(db_session, *, asset="BTC"):
    permission = await create_permission(
        db_session,
        profile_name="overlay-runtime-test",
        allowed_actions=["CREATE_EXPERIMENT"],
        scope={},
        limits={},
        updated_by="test",
    )
    return await create_run(
        db_session,
        objective="overlay runtime",
        scope={"asset": asset},
        autonomy_level="EXPERIMENT",
        budget_experiments=1,
        permission=permission,
        llm_provider="mock",
    )


@pytest.mark.asyncio
async def test_runtime_summary_reports_before_after_pnl_and_coverage(db_session):
    run = await _make_run(db_session)
    now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
    overlay = AIConfigOverlay(
        run_id=run.id,
        scope={"target": "PAPER", "asset": "BTC"},
        changes={"DEAD_ZONE_WIDTH": 0.06},
        base_settings_hash="a" * 64,
        resulting_settings_hash="b" * 64,
        status="APPLIED",
        created_by="test",
        created_at=now - timedelta(hours=1),
        expires_at=now + timedelta(hours=1),
    )
    before = TradeHistory(
        market_id="before",
        asset="BTCUSDT",
        outcome_bought="YES",
        amount_usdc=10.0,
        executed_price=0.5,
        predicted_flip_prob=0.7,
        active_features="FS_D0",
        status="FILLED",
        mode="PAPER",
        pnl=1.25,
        created_at=now - timedelta(hours=2),
        timestamp=now - timedelta(hours=2),
    )
    after = TradeHistory(
        market_id="after",
        asset="BTC",
        outcome_bought="NO",
        amount_usdc=10.0,
        executed_price=0.5,
        predicted_flip_prob=0.6,
        active_features="FS_D0",
        status="FILLED",
        mode="PAPER",
        pnl=2.5,
        created_at=now - timedelta(minutes=10),
        timestamp=now - timedelta(minutes=10),
    )
    db_session.add(overlay)
    await db_session.flush()
    after.ai_lab_overlay_ids = [overlay.id]
    db_session.add_all([before, after])
    await db_session.flush()

    rows = await get_paper_overlay_runtime_summary(
        db_session, run_id=run.id, now=now
    )

    assert len(rows) == 1
    metrics = rows[0]["metrics"]
    assert metrics["before"] == {"trade_count": 1, "pnl": 1.25}
    assert metrics["after"]["trade_count"] == 1
    assert metrics["after"]["pnl"] == 2.5
    assert metrics["after"]["coverage"] == 0.5
    assert metrics["paper_trade_count"] == 2


@pytest.mark.asyncio
async def test_overlay_scope_normalizes_asset_symbols(db_session):
    run = await _make_run(db_session)
    now = datetime.now(timezone.utc)
    btc = AIConfigOverlay(
        run_id=run.id,
        scope={"target": "PAPER", "asset": "BTCUSDT"},
        changes={"TRADE_BET_SIZE_USDC": 20},
        base_settings_hash="a" * 64,
        resulting_settings_hash="b" * 64,
        status="APPLIED",
        expires_at=now + timedelta(minutes=5),
    )
    eth = AIConfigOverlay(
        run_id=run.id,
        scope={"target": "PAPER", "asset": "ETH"},
        changes={"TRADE_BET_SIZE_USDC": 30},
        base_settings_hash="c" * 64,
        resulting_settings_hash="d" * 64,
        status="APPLIED",
        expires_at=now + timedelta(minutes=5),
    )
    db_session.add_all([btc, eth])
    await db_session.flush()

    effective, ids = await resolve_paper_runtime_settings(
        db_session,
        {"TRADE_BET_SIZE_USDC": "10"},
        now=now,
        asset="BTC",
    )

    assert effective["TRADE_BET_SIZE_USDC"] == "20"
    assert ids == [btc.id]

@pytest.mark.asyncio
async def test_regime_scoped_overlay_requires_matching_regime(db_session):
    run = await _make_run(db_session)
    now = datetime.now(timezone.utc)
    overlay = AIConfigOverlay(
        run_id=run.id,
        scope={"target": "PAPER", "asset": "BTC", "regime": "trend"},
        changes={"TRADE_BET_SIZE_USDC": 20},
        base_settings_hash="a" * 64,
        resulting_settings_hash="b" * 64,
        status="APPLIED",
        expires_at=now + timedelta(minutes=5),
    )
    db_session.add(overlay)
    await db_session.flush()

    unclassified, unclassified_ids = await resolve_paper_runtime_settings(
        db_session, {"TRADE_BET_SIZE_USDC": "10"}, now=now, asset="BTC"
    )
    classified, classified_ids = await resolve_paper_runtime_settings(
        db_session,
        {"TRADE_BET_SIZE_USDC": "10"},
        now=now,
        asset="BTC",
        regime="trend",
    )

    assert unclassified["TRADE_BET_SIZE_USDC"] == "10"
    assert unclassified_ids == []
    assert classified["TRADE_BET_SIZE_USDC"] == "20"
    assert classified_ids == [overlay.id]


@pytest.mark.asyncio
async def test_existing_skipped_trade_receives_overlay_trace(db_session):
    market = SimpleNamespace(market_id="skip-market", asset="BTC")
    now = datetime.now(timezone.utc)
    await save_or_update_skipped_trade(
        db_session, market, "not enough edge", 0.0, None, now
    )
    row = (
        await db_session.execute(
            sa.select(TradeHistory).where(TradeHistory.market_id == "skip-market")
        )
    ).scalar_one()
    assert row.ai_lab_overlay_ids in (None, [])

    await save_or_update_skipped_trade(
        db_session,
        market,
        "not enough edge",
        0.0,
        None,
        now + timedelta(seconds=1),
        existing_skipped=row,
        decision_details={"ai_lab_overlay_ids": [42]},
    )
    assert row.ai_lab_overlay_ids == [42]


@pytest.mark.asyncio
async def test_live_engine_path_never_resolves_paper_overlays(db_session, monkeypatch):
    from polyflip.trading import engine
    from polyflip.trading import ml_inference

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    cfg = SimpleNamespace(
        trading_enabled=True,
        trading_mode="COMBINED",
        outs_min_edge=0.0,
        trade_max_price=1.0,
        active_features_str="FS_D0",
    )
    market = SimpleNamespace(
        market_id="live-market",
        asset="BTC",
        end_time_est=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    guard = SimpleNamespace(
        passed=False,
        skip_reason="guard: Time left <= 0",
        existing_skipped=None,
    )

    async def value_async(value):
        return value

    monkeypatch.setattr(engine, "load_trading_settings", value_async)
    monkeypatch.setattr(engine, "parse_trading_settings", lambda _raw: cfg)
    monkeypatch.setattr(engine, "load_eligible_markets", lambda *_args: value_async([market]))
    monkeypatch.setattr(engine, "check_market_guards", lambda *_args: value_async(guard))
    monkeypatch.setattr(ml_inference, "populate_models_cache", value_async)

    async def fail_resolver(*_args, **_kwargs):
        raise AssertionError("PAPER overlay resolver called in LIVE mode")

    monkeypatch.setattr(
        "polyflip.ai_lab.paper_overlay.resolve_paper_runtime_settings",
        fail_resolver,
    )
    engine._ACTIVE_MARKETS.clear()

    await engine.trade_worker_cycle(db_session, None)
