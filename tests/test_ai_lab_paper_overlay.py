"""PAPER overlay resolution and scope filtering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from polyflip.ai_lab.paper_overlay import resolve_paper_runtime_settings
from polyflip.ai_lab.service import create_permission, create_run
from polyflip.db.models import AIConfigOverlay


@pytest.mark.asyncio
async def test_paper_overlay_filters_expired_and_non_paper_rows(db_session):
    permission = await create_permission(
        db_session,
        profile_name="paper-overlay-test",
        allowed_actions=["CREATE_EXPERIMENT"],
        scope={},
        limits={},
        updated_by="test",
    )
    run = await create_run(
        db_session,
        objective="paper overlay",
        scope={"asset": "BTC"},
        autonomy_level="EXPERIMENT",
        budget_experiments=1,
        permission=permission,
        llm_provider="mock",
    )
    now = datetime.now(timezone.utc)
    active = AIConfigOverlay(
        run_id=run.id,
        scope={"target": "SHADOW_SIMULATION"},
        changes={"TRADE_BET_SIZE_USDC": 20, "TRADING_ENABLED": False},
        base_settings_hash="a" * 64,
        resulting_settings_hash="b" * 64,
        status="APPLIED",
        created_by="test",
        expires_at=now + timedelta(minutes=5),
    )
    expired = AIConfigOverlay(
        run_id=run.id,
        scope={"target": "PAPER"},
        changes={"TRADE_BET_SIZE_USDC": 999},
        base_settings_hash="c" * 64,
        resulting_settings_hash="d" * 64,
        status="APPLIED",
        created_by="test",
        expires_at=now - timedelta(seconds=1),
    )
    live_only = AIConfigOverlay(
        run_id=run.id,
        scope={"target": "LIVE"},
        changes={"TRADE_BET_SIZE_USDC": 777},
        base_settings_hash="e" * 64,
        resulting_settings_hash="f" * 64,
        status="APPLIED",
        created_by="test",
        expires_at=now + timedelta(minutes=5),
    )
    db_session.add_all([active, expired, live_only])
    await db_session.flush()

    effective, overlay_ids = await resolve_paper_runtime_settings(
        db_session,
        {"TRADE_BET_SIZE_USDC": "10", "TRADING_ENABLED": "true"},
        now=now,
    )

    assert effective["TRADE_BET_SIZE_USDC"] == "20"
    assert effective["TRADING_ENABLED"] == "False"
    assert overlay_ids == [active.id]
    assert expired.status == "EXPIRED"
